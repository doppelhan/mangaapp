import os
import uuid
import uuid
from PIL import Image, ImageOps
from config import UPLOAD_FOLDER, PROCESSED_FOLDER

def generate_session_id():
    """Generates a unique session ID for the user."""
    return str(uuid.uuid4())

def save_uploaded_image(file_storage, session_id):
    """Saves the uploaded file to disk using the session ID."""
    filename = f"{session_id}_original.png"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # Process original image
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)
    img = img.convert('RGB')

    # Save the original format in a sidecar or memory-light way
    # (for now, we'll standardize to PNG to simplify processing)
    img.save(filepath, format="PNG")

    return filepath

def get_original_image_path(session_id):
    """Returns the path to the original uploaded image."""
    filepath = os.path.join(UPLOAD_FOLDER, f"{session_id}_original.png")
    if os.path.exists(filepath):
        return filepath
    return None

def save_processed_image(pil_image, session_id):
    """Saves the processed image to disk."""
    filename = f"{session_id}_processed.png"
    filepath = os.path.join(PROCESSED_FOLDER, filename)
    pil_image.save(filepath, format="PNG")
    return filepath

def get_processed_image_path(session_id):
    """Returns the path to the processed image."""
    filepath = os.path.join(PROCESSED_FOLDER, f"{session_id}_processed.png")
    if os.path.exists(filepath):
        return filepath
    return None
