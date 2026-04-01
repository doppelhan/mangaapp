from flask import Blueprint, request, jsonify, send_file
import io
import base64
from PIL import Image

from utils.session import (
    generate_session_id,
    save_uploaded_image,
    get_original_image_path,
    save_processed_image,
    get_processed_image_path
)
from processing.pipeline import process_image_pipeline

api_bp = Blueprint('api', __name__)

def image_to_base64(img):
    """Converts PIL Image to base64 string."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@api_bp.route('/upload', methods=['POST'])
def upload():
    """Handles image upload and initiates a session."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        session_id = generate_session_id()
        save_uploaded_image(file, session_id)

        # We need to return the base64 of the original so the UI can display it
        # (Alternatively, could serve it statically, but keeping with original flow for now)
        img = Image.open(get_original_image_path(session_id))
        img_str = image_to_base64(img)

        return jsonify({
            'session_id': session_id,
            'original_image': img_str
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/process', methods=['POST'])
def process():
    """Processes the image based on parameters."""
    params = request.json
    session_id = params.get('session_id')

    if not session_id:
        return jsonify({'status': 'error', 'message': 'Missing session_id'}), 400

    original_path = get_original_image_path(session_id)
    if not original_path:
        return jsonify({'status': 'error', 'message': 'Image not found for session'}), 404

    try:
        # Load image
        img = Image.open(original_path)

        # Process
        processed_pil = process_image_pipeline(img, params)

        # Save output
        save_processed_image(processed_pil, session_id)

        # Return base64 for preview
        img_str = image_to_base64(processed_pil)

        return jsonify({
            'status': 'done',
            'image': img_str,
            'session_id': session_id
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@api_bp.route('/save_image')
def save_image():
    """Downloads the processed image for a given session."""
    session_id = request.args.get('session_id')
    if not session_id:
        return 'Missing session_id', 400

    processed_path = get_processed_image_path(session_id)
    if processed_path:
        return send_file(processed_path, mimetype='image/png', as_attachment=True,
                         download_name='processed_image.png')
    else:
        return 'No image to save.', 404
