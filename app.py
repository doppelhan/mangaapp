import os
from flask import Flask, jsonify, request, render_template

import config
from routes.web import web_bp
from routes.api import api_bp
from utils.session import get_original_image_path, generate_session_id, save_uploaded_image
import io
import base64
from PIL import Image

app = Flask(__name__)
app.config.from_object('config')

# Register blueprints
app.register_blueprint(web_bp)
app.register_blueprint(api_bp, url_prefix='/api')

def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Retrofit the index route to handle POST for backward compatibility
# In a modern rewrite, we'd use Ajax for uploads, but let's keep the form submission
# if that's what the original frontend expects
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename != '':
            try:
                session_id = generate_session_id()
                save_uploaded_image(file, session_id)
                img = Image.open(get_original_image_path(session_id))
                original_img_str = image_to_base64(img)

                return render_template('index.html',
                                       original_image=original_img_str,
                                       session_id=session_id, # We need to pass this to the template
                                       threshold_methods=config.THRESHOLD_METHODS,
                                       noise_reduction_methods=config.NOISE_REDUCTION_METHODS,
                                       screen_tone_patterns=config.SCREEN_TONE_PATTERNS,
                                       screen_tone_area_patterns=config.SCREEN_TONE_AREA_PATTERNS,
                                       pencil_shading_styles=config.PENCIL_SHADING_STYLES)
            except Exception as e:
                return str(e)

    return render_template('index.html',
                           threshold_methods=config.THRESHOLD_METHODS,
                           noise_reduction_methods=config.NOISE_REDUCTION_METHODS,
                           screen_tone_patterns=config.SCREEN_TONE_PATTERNS,
                           screen_tone_area_patterns=config.SCREEN_TONE_AREA_PATTERNS,
                           pencil_shading_styles=config.PENCIL_SHADING_STYLES)

# Retrofit the process route
@app.route('/process', methods=['POST'])
def process():
    params = request.json
    session_id = params.get('session_id')

    # If no session ID is found (e.g. older UI), fall back to original logic behavior (mocked here)
    if not session_id:
        return jsonify({'status': 'error', 'message': 'Session ID is required.'})

    original_path = get_original_image_path(session_id)
    if not original_path:
         return jsonify({'status': 'error', 'message': 'Image not found.'})

    try:
        from processing.pipeline import process_image_pipeline
        from utils.session import save_processed_image

        img = Image.open(original_path)
        processed_pil = process_image_pipeline(img, params)
        save_processed_image(processed_pil, session_id)

        img_str = image_to_base64(processed_pil)
        return jsonify({'status': 'done', 'image': img_str})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/save_image')
def save_image():
    from utils.session import get_processed_image_path
    from flask import send_file

    session_id = request.args.get('session_id')
    if not session_id:
        return 'Missing session_id', 400

    processed_path = get_processed_image_path(session_id)
    if processed_path:
        return send_file(processed_path, mimetype='image/png', as_attachment=True,
                         download_name='processed_image.png')
    else:
        return 'No image to save.', 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
