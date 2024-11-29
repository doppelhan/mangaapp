import os
import uuid
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import cv2
import numpy as np
import io
import base64
import logging
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

# ตั้งค่าขนาดสูงสุดของไฟล์ที่สามารถอัปโหลดได้ (100MB)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# ตั้งค่า Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory สำหรับเก็บไฟล์ชั่วคราว
TEMP_DIR = os.path.join(os.getcwd(), 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)

# Global dictionary เพื่อเก็บข้อมูลภาพ
images = {}

# Lists of parameters
threshold_methods = [
    "Global Thresholding",
    "Adaptive Mean Thresholding",
    "Adaptive Gaussian Thresholding",
    "Special Adaptive Thresholding"
]

noise_reduction_methods = [
    "None", "Median Filter", "Bilateral Filter", "Gaussian Blur", "Non-local Means Denoising"
]

screen_tone_patterns = [
    "None",
    "Dots",
    "Hatching",
    "Cross-Hatching",
    "Lines",
    "Vertical Lines",
    "Horizontal Lines",
    "Checkerboard",
    "Diagonal Stripes",
    "Spiral",
    "Pencil Shading",
    "Halftone Circles",
    "Scratch Lines",
    "Noise",
    "Fish Scale",
    "Scary Classic"
]

screen_tone_area_patterns = [
    "Global Darkness",
    "Shadow Regions",
    "Edge Detection",
    "Background Only",
    "Mid-tone Regions",
    "Highlight Regions",
    "High Contrast Edges",
    "Gradient Magnitude",
    "Texture Complexity",
    "Dark Edges",
    "Light Edges",
    "Uniform Regions",
    "High Texture Regions",
    "Foreground Objects",
    "Contours",
    "High Frequency Regions",
    "Blob Detection",
    "Brightest Regions"
]

pencil_shading_styles = [
    "Light",
    "Medium",
    "Dark",
    "Hatched",
    "Cross-hatched",
    "Manga Style"
]

# Function to adjust gamma
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** invGamma * 255
    table = np.clip(table, 0, 255).astype("uint8")
    return cv2.LUT(image, table)

# Function to generate mask
def generate_mask(gray_image, pattern):
    if pattern == "Global Darkness":
        mask = gray_image < 100

    elif pattern == "Shadow Regions":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        shadow = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
        shadow = cv2.GaussianBlur(shadow, (5,5), 0)
        mask = shadow < 100

    elif pattern == "Edge Detection":
        edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)
        mask = edges > 0

    elif pattern == "Background Only":
        blurred = cv2.GaussianBlur(gray_image, (7,7), 0)
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        mask = thresh == 255

    elif pattern == "Mid-tone Regions":
        mask = (gray_image >= 100) & (gray_image <= 200)

    elif pattern == "Highlight Regions":
        mask = gray_image > 200

    elif pattern == "High Contrast Edges":
        grad_x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0)
        grad_y = cv2.Sobel(gray_image, cv2.CV_16S, 0, 1)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad_mag = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        mask = grad_mag > 50

    elif pattern == "Gradient Magnitude":
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        abs_laplacian = np.absolute(laplacian)
        mask = abs_laplacian > 30

    elif pattern == "Texture Complexity":
        kernel_size = 9
        mean = cv2.blur(gray_image.astype(np.float32), (kernel_size, kernel_size))
        sqr_mean = cv2.blur(gray_image.astype(np.float32)**2, (kernel_size, kernel_size))
        variance = sqr_mean - mean**2
        mask = variance > 500

    elif pattern == "Dark Edges":
        edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)
        dark_pixels = gray_image < 100
        mask = (edges > 0) & dark_pixels

    elif pattern == "Light Edges":
        edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)
        bright_pixels = gray_image > 150
        mask = (edges > 0) & bright_pixels

    elif pattern == "Uniform Regions":
        kernel_size = 7
        mean = cv2.blur(gray_image.astype(np.float32), (kernel_size, kernel_size))
        sqr_mean = cv2.blur(gray_image.astype(np.float32)**2, (kernel_size, kernel_size))
        variance = sqr_mean - mean**2
        mask = variance < 50

    elif pattern == "High Texture Regions":
        kernel_size = 7
        mean = cv2.blur(gray_image.astype(np.float32), (kernel_size, kernel_size))
        sqr_mean = cv2.blur(gray_image.astype(np.float32)**2, (kernel_size, kernel_size))
        variance = sqr_mean - mean**2
        mask = variance > 100

    elif pattern == "Foreground Objects":
        _, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = mask == 255

    elif pattern == "Contours":
        _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray_image, dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        mask = mask > 0

    elif pattern == "High Frequency Regions":
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        abs_laplacian = np.absolute(laplacian)
        mask = abs_laplacian > np.mean(abs_laplacian)
        mask = mask.astype(np.uint8)

    elif pattern == "Blob Detection":
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 100
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray_image)
        mask = np.zeros_like(gray_image, dtype=np.uint8)
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            radius = int(kp.size / 2)
            cv2.circle(mask, (x, y), radius, 255, -1)
        mask = mask > 0

    elif pattern == "Brightest Regions":
        threshold_value = np.percentile(gray_image, 95)
        mask = gray_image >= threshold_value

    else:
        mask = gray_image < 100

    return mask

# Function to apply screen tone
def apply_screen_tone(image, size=5, pattern="None", mask=None, gray_image=None, color=(50, 50, 50), density=50, pencil_style="Light"):
    if pattern == "None" or mask is None:
        return image  # Do not apply screen tone

    draw = ImageDraw.Draw(image, 'RGBA')  # Use RGBA mode to support transparency
    width, height = image.size

    # Create a mask image for processing
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
    mask_image = mask_image.resize((width, height), Image.Resampling.NEAREST)
    mask_array = np.array(mask_image)

    # Prepare brightness data
    if gray_image is not None:
        gray_image_resized = cv2.resize(gray_image, (width, height), interpolation=cv2.INTER_NEAREST)
        brightness_array = 255 - gray_image_resized  # Higher value means darker
        brightness_array = brightness_array / 255.0  # Scale to [0,1]
    else:
        brightness_array = np.ones((height, width))

    # Choose pattern
    if pattern == "Dots":
        max_radius = size
        min_radius = 1
        step = size * 2

        for y in range(0, height, step):
            for x in range(0, width, step):
                if mask_array[y, x]:
                    brightness = brightness_array[y, x]
                    radius = min_radius + brightness * (max_radius - min_radius)
                    radius = int(radius)
                    if radius > 0:
                        dot_color = color + (int(255 * brightness),)
                        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=dot_color)

    elif pattern == "Hatching":
        min_spacing = max(2, 15 - size)
        max_spacing = max(5, 25 - size)
        spacing = int(max_spacing - (density / 100) * (max_spacing - min_spacing))
        spacing = max(spacing, 2)

        for y in range(0, height, spacing):
            for x in range(0, width, spacing):
                if mask_array[y, x]:
                    brightness = brightness_array[y, x]
                    line_color = color + (int(255 * brightness),)
                    length = int(size + brightness * size)
                    draw.line((x, y, x + length, y), fill=line_color, width=1)
                    draw.line((x, y, x + length, y + length), fill=line_color, width=1)

    elif pattern == "Cross-Hatching":
        min_spacing = max(2, 15 - size)
        max_spacing = max(5, 25 - size)
        spacing = int(max_spacing - (density / 100) * (max_spacing - min_spacing))
        spacing = max(spacing, 2)

        for y in range(0, height, spacing):
            for x in range(0, width, spacing):
                if mask_array[y, x]:
                    brightness = brightness_array[y, x]
                    line_color = color + (int(255 * brightness),)
                    length = int(size + brightness * size * 2)
                    draw.line((x, y, x + length, y + length), fill=line_color, width=1)
                    draw.line((x + length, y, x, y + length), fill=line_color, width=1)

    elif pattern == "Manga Style":
        # Manga style pencil shading
        line_spacing = size
        line_width = 1
        angles = [np.pi / 6, -np.pi / 6]
        for angle in angles:
            for y in range(0, height, line_spacing):
                for x in range(0, width, line_spacing):
                    if mask_array[y, x]:
                        brightness = brightness_array[y, x]
                        opacity = int(80 + brightness * 175)
                        line_color = color + (opacity,)
                        length = int(size * 2 + brightness * size * 2)
                        dx = length * np.cos(angle)
                        dy = length * np.sin(angle)
                        x_end = x + dx
                        y_end = y + dy
                        draw.line((x - dx, y - dy, x_end, y_end), fill=line_color, width=line_width)

    elif pattern == "Halftone Circles":
        max_radius = size
        min_radius = 1
        step = size * 2
        for y in range(0, height, step):
            for x in range(0, width, step):
                if mask_array[y, x]:
                    brightness = brightness_array[y, x]
                    radius = min_radius + brightness * (max_radius - min_radius)
                    radius = int(radius)
                    if radius > 0:
                        dot_color = color + (int(255 * brightness),)
                        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=dot_color)

    elif pattern == "Scratch Lines":
        num_lines = int((width + height) * density / 1000)
        for _ in range(num_lines):
            x_start = np.random.randint(0, width)
            y_start = np.random.randint(0, height)
            if mask_array[y_start, x_start]:
                brightness = brightness_array[y_start, x_start]
                line_color = color + (int(255 * brightness * 0.5),)
                length = np.random.randint(size * 5, size * 20)
                angle = np.random.uniform(0, 2 * np.pi)
                x_end = int(x_start + length * np.cos(angle))
                y_end = int(y_start + length * np.sin(angle))
                draw.line((x_start, y_start, x_end, y_end), fill=line_color, width=1)

    elif pattern == "Noise":
        num_dots = int(width * height * (density / 1000))
        for _ in range(num_dots):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            if mask_array[y, x]:
                brightness = brightness_array[y, x]
                dot_color = color + (int(255 * brightness * 0.3),)
                draw.point((x, y), fill=dot_color)

    elif pattern == "Fish Scale":
        step = size * 2
        radius = size
        for y in range(0, height, int(step * 0.75)):
            offset = 0 if (y // step) % 2 == 0 else radius
            for x in range(-radius, width + radius, step):
                if 0 <= y < height and 0 <= x + offset < width and mask_array[y, x + offset]:
                    brightness = brightness_array[y, x + offset]
                    fill_color = color + (int(255 * brightness),)
                    bbox = (x + offset - radius, y - radius, x + offset + radius, y + radius)
                    draw.pieslice(bbox, start=0, end=180, fill=fill_color)

    elif pattern == "Scary Classic":
        num_shadows = int(width * height * density / 100000)
        for _ in range(num_shadows):
            x_center = np.random.randint(0, width)
            y_center = np.random.randint(0, height)
            if mask_array[y_center, x_center]:
                brightness = brightness_array[y_center, x_center]
                shadow_color = color + (int(255 * brightness * 0.1),)
                radius = np.random.randint(size * 5, size * 15)
                draw.ellipse((x_center - radius, y_center - radius, x_center + radius, y_center + radius), fill=shadow_color)

        num_lines = int((width + height) * density / 500)
        for _ in range(num_lines):
            x_start = np.random.randint(0, width)
            y_start = np.random.randint(0, height)
            if mask_array[y_start, x_start]:
                brightness = brightness_array[y_start, x_start]
                line_color = color + (int(255 * brightness * 0.2),)
                length = np.random.randint(size * 10, size * 30)
                angle = np.random.uniform(0, 2 * np.pi)
                x_end = int(x_start + length * np.cos(angle))
                y_end = int(y_start + length * np.sin(angle))
                draw.line((x_start, y_start, x_end, y_end), fill=line_color, width=1)

    else:
        return image

    return image

# Function to convert image to base64
def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Function to process image
def process_image(image_path, params, is_preview=True):
    # Extract parameters from params
    threshold_value = int(params.get('threshold', 128))
    contrast_value = float(params.get('contrast', 1.0))
    brightness_value = float(params.get('brightness', 1.0))
    gamma_value = float(params.get('gamma', 1.0))
    exposure_value = float(params.get('exposure', 1.0))
    method = params.get('method', 'Special Adaptive Thresholding')
    block_size = int(params.get('block_size', 11))
    c_value = int(params.get('c_value', 6))  # Default C Value set to 6
    invert = params.get('invert', False)
    noise_reduction_method = params.get('noise_reduction', 'None')
    sharpen = params.get('sharpen', False)
    edge_enhance = params.get('edge_enhance', False)
    hist_eq = params.get('hist_eq', False)
    clahe = params.get('clahe', False)
    clip_limit = float(params.get('clip_limit', 2.0))
    tile_grid_size = int(params.get('tile_grid_size', 8))
    local_contrast = params.get('local_contrast', False)
    kernel_size = int(params.get('kernel_size', 9))

    # Screen Tone Layer 1
    screen_tone_1 = params.get('screen_tone_1', False)
    screen_tone_pattern_1 = params.get('screen_tone_pattern_1', 'None')
    screen_tone_size_1 = int(params.get('screen_tone_size_1', 2))  # Default Size set to 2
    screen_tone_density_1 = int(params.get('screen_tone_density_1', 50))  # Default Density set to 50
    screen_tone_area_pattern_1 = params.get('screen_tone_area_pattern_1', 'Global Darkness')
    pencil_shading_style_1 = params.get('pencil_shading_style_1', 'Light')
    screen_tone_color_1 = params.get('screen_tone_color_1', '#323232')
    screen_tone_color_1 = tuple(int(screen_tone_color_1.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    # Screen Tone Layer 2
    screen_tone_2 = params.get('screen_tone_2', False)
    screen_tone_pattern_2 = params.get('screen_tone_pattern_2', 'None')
    screen_tone_size_2 = int(params.get('screen_tone_size_2', 2))  # Default Size set to 2
    screen_tone_density_2 = int(params.get('screen_tone_density_2', 50))  # Default Density set to 50
    screen_tone_area_pattern_2 = params.get('screen_tone_area_pattern_2', 'Shadow Regions')
    pencil_shading_style_2 = params.get('pencil_shading_style_2', 'Medium')
    screen_tone_color_2 = params.get('screen_tone_color_2', '#505050')
    screen_tone_color_2 = tuple(int(screen_tone_color_2.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    # Set threshold_type to THRESH_BINARY
    threshold_type = cv2.THRESH_BINARY

    # Load the image
    img = Image.open(image_path).convert('RGB')
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Adjust Contrast and Brightness
    img_cv = cv2.convertScaleAbs(img_cv, alpha=contrast_value, beta=(brightness_value - 1) * 255)

    # Adjust Gamma Correction
    img_cv = adjust_gamma(img_cv, gamma=gamma_value)

    # Convert image to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Exposure Compensation
    gray = cv2.convertScaleAbs(gray, alpha=exposure_value, beta=0)

    # Histogram Equalization
    if hist_eq:
        gray = cv2.equalizeHist(gray)

    # Adaptive Histogram Equalization (CLAHE)
    if clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        gray = clahe_obj.apply(gray)

    # Local Contrast Enhancement
    if local_contrast:
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size <= 1:
            kernel_size = 3
        gaussian = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        gray = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)

    # Noise Reduction
    if noise_reduction_method == "Median Filter":
        denoised = cv2.medianBlur(gray, 3)
    elif noise_reduction_method == "Bilateral Filter":
        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    elif noise_reduction_method == "Gaussian Blur":
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    elif noise_reduction_method == "Non-local Means Denoising":
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    else:
        denoised = gray

    # Sharpen the image
    if sharpen:
        gaussian = cv2.GaussianBlur(denoised, (0, 0), sigmaX=3)
        denoised = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)

    # Edge Enhancement
    if edge_enhance:
        edges = cv2.Canny(denoised, threshold1=50, threshold2=150)
        denoised = cv2.bitwise_or(denoised, edges)

    # Thresholding
    if method == "Global Thresholding":
        _, thresh = cv2.threshold(denoised, threshold_value, 255, threshold_type)
    elif method == "Adaptive Mean Thresholding":
        if block_size % 2 == 0:
            block_size += 1
        if block_size <= 1:
            block_size = 3
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, threshold_type, block_size, c_value
        )
    elif method == "Adaptive Gaussian Thresholding":
        if block_size % 2 == 0:
            block_size += 1
        if block_size <= 1:
            block_size = 3
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type, block_size, c_value
        )
    elif method == "Special Adaptive Thresholding":
        if block_size % 2 == 0:
            block_size += 1
        if block_size <= 1:
            block_size = 3
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_value
        )
        black_mask = gray < 50
        thresh[black_mask] = 0
    else:
        _, thresh = cv2.threshold(denoised, threshold_value, 255, threshold_type)

    # Invert colors
    if invert:
        thresh = cv2.bitwise_not(thresh)

    # Convert back to RGB
    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    # Convert to PIL Image
    processed_pil = Image.fromarray(thresh_rgb)

    # Apply Screen Tone Layer 1
    if screen_tone_1 and screen_tone_pattern_1 != "None":
        mask1 = generate_mask(gray, screen_tone_area_pattern_1)
        processed_pil = apply_screen_tone(
            processed_pil,
            size=screen_tone_size_1,
            pattern=screen_tone_pattern_1,
            mask=mask1,
            gray_image=gray,
            color=screen_tone_color_1,
            density=screen_tone_density_1,
            pencil_style=pencil_shading_style_1
        )

    # Apply Screen Tone Layer 2
    if screen_tone_2 and screen_tone_pattern_2 != "None":
        mask2 = generate_mask(gray, screen_tone_area_pattern_2)
        processed_pil = apply_screen_tone(
            processed_pil,
            size=screen_tone_size_2,
            pattern=screen_tone_pattern_2,
            mask=mask2,
            gray_image=gray,
            color=screen_tone_color_2,
            density=screen_tone_density_2,
            pencil_style=pencil_shading_style_2
        )

    return processed_pil

@app.route('/', methods=['GET', 'POST'])
def index():
    global images

    if request.method == 'POST':
        # Receive image file from user
        file = request.files['image']
        if file:
            try:
                # Generate a unique ID for the image
                image_id = str(uuid.uuid4())

                # Save original image
                original_path = os.path.join(TEMP_DIR, f'original_{image_id}.png')
                img = Image.open(file.stream).convert('RGB')
                img.save(original_path)

                # Create and save preview image (e.g., max dimension 800px)
                preview_path = os.path.join(TEMP_DIR, f'preview_{image_id}.png')
                img.thumbnail((800, 800), Image.Resampling.LANCZOS)
                img.save(preview_path)

                # Process the preview image
                default_params = {
                    'method': 'Special Adaptive Thresholding',
                    'c_value': 6,
                    'screen_tone_size_1': 2,
                    'screen_tone_density_1': 50,
                    'screen_tone_size_2': 2,
                    'screen_tone_density_2': 50
                }
                processed_preview = process_image(preview_path, default_params, is_preview=True)
                processed_preview_path = os.path.join(TEMP_DIR, f'processed_preview_{image_id}.png')
                processed_preview.save(processed_preview_path)

                # Store paths in the images dictionary
                images[image_id] = {
                    'original': original_path,
                    'preview': preview_path,
                    'processed_preview': processed_preview_path
                }

                # Convert images to base64
                original_img = Image.open(original_path)
                original_img_str = image_to_base64(original_img)

                processed_preview_img = Image.open(processed_preview_path)
                processed_preview_img_str = image_to_base64(processed_preview_img)

                # Get sizes
                original_size = original_img.size
                processed_preview_size = processed_preview_img.size

                return render_template('index.html',
                                       image_id=image_id,
                                       original_image=original_img_str,
                                       processed_image=processed_preview_img_str,
                                       threshold_methods=threshold_methods,
                                       noise_reduction_methods=noise_reduction_methods,
                                       screen_tone_patterns=screen_tone_patterns,
                                       screen_tone_area_patterns=screen_tone_area_patterns,
                                       pencil_shading_styles=pencil_shading_styles,
                                       default_params=default_params,
                                       original_size=original_size,
                                       processed_size=processed_preview_size)
            except Exception as e:
                logger.error(f"Error processing uploaded image: {e}")
                return "Error processing the uploaded image.", 500
    else:
        return render_template('index.html',
                               threshold_methods=threshold_methods,
                               noise_reduction_methods=noise_reduction_methods,
                               screen_tone_patterns=screen_tone_patterns,
                               screen_tone_area_patterns=screen_tone_area_patterns,
                               pencil_shading_styles=pencil_shading_styles,
                               default_params={
                                   'method': 'Special Adaptive Thresholding',
                                   'c_value': 6,
                                   'screen_tone_size_1': 2,
                                   'screen_tone_density_1': 50,
                                   'screen_tone_size_2': 2,
                                   'screen_tone_density_2': 50
                               })

@app.route('/process', methods=['POST'])
def process_route():
    global images
    data = request.json
    image_id = data.get('image_id')
    if not image_id or image_id not in images:
        return jsonify({'status': 'error', 'message': 'Invalid image ID.'}), 400

    try:
        # Get parameters
        params = {
            'threshold': data.get('threshold', 128),
            'contrast': data.get('contrast', 1.0),
            'brightness': data.get('brightness', 1.0),
            'gamma': data.get('gamma', 1.0),
            'exposure': data.get('exposure', 1.0),
            'method': data.get('method', 'Special Adaptive Thresholding'),
            'block_size': data.get('block_size', 11),
            'c_value': data.get('c_value', 6),
            'invert': data.get('invert', False),
            'noise_reduction': data.get('noise_reduction', 'None'),
            'sharpen': data.get('sharpen', False),
            'edge_enhance': data.get('edge_enhance', False),
            'hist_eq': data.get('hist_eq', False),
            'clahe': data.get('clahe', False),
            'clip_limit': data.get('clip_limit', 2.0),
            'tile_grid_size': data.get('tile_grid_size', 8),
            'local_contrast': data.get('local_contrast', False),
            'kernel_size': data.get('kernel_size', 9),
            'screen_tone_1': data.get('screen_tone_1', False),
            'screen_tone_pattern_1': data.get('screen_tone_pattern_1', 'None'),
            'screen_tone_size_1': data.get('screen_tone_size_1', 2),
            'screen_tone_density_1': data.get('screen_tone_density_1', 50),
            'screen_tone_area_pattern_1': data.get('screen_tone_area_pattern_1', 'Global Darkness'),
            'pencil_shading_style_1': data.get('pencil_shading_style_1', 'Light'),
            'screen_tone_color_1': data.get('screen_tone_color_1', '#323232'),
            'screen_tone_2': data.get('screen_tone_2', False),
            'screen_tone_pattern_2': data.get('screen_tone_pattern_2', 'None'),
            'screen_tone_size_2': data.get('screen_tone_size_2', 2),
            'screen_tone_density_2': data.get('screen_tone_density_2', 50),
            'screen_tone_area_pattern_2': data.get('screen_tone_area_pattern_2', 'Shadow Regions'),
            'pencil_shading_style_2': data.get('pencil_shading_style_2', 'Medium'),
            'screen_tone_color_2': data.get('screen_tone_color_2', '#505050'),
        }

        # Process the preview image
        preview_path = images[image_id]['preview']
        processed_preview = process_image(preview_path, params, is_preview=True)
        processed_preview_path = images[image_id]['processed_preview']
        processed_preview.save(processed_preview_path)

        # Convert to base64
        processed_preview_img = Image.open(processed_preview_path)
        processed_preview_img_str = image_to_base64(processed_preview_img)

        # Get size
        processed_preview_size = processed_preview_img.size

        return jsonify({'status': 'done', 'image': processed_preview_img_str, 'size': processed_preview_size})
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({'status': 'error', 'message': 'Error processing image.'}), 500

@app.route('/save', methods=['POST'])
def save_image_route():
    global images
    data = request.json
    image_id = data.get('image_id')
    if not image_id or image_id not in images:
        return jsonify({'status': 'error', 'message': 'Invalid image ID.'}), 400

    try:
        # Get parameters
        params = {
            'threshold': data.get('threshold', 128),
            'contrast': data.get('contrast', 1.0),
            'brightness': data.get('brightness', 1.0),
            'gamma': data.get('gamma', 1.0),
            'exposure': data.get('exposure', 1.0),
            'method': data.get('method', 'Special Adaptive Thresholding'),
            'block_size': data.get('block_size', 11),
            'c_value': data.get('c_value', 6),
            'invert': data.get('invert', False),
            'noise_reduction': data.get('noise_reduction', 'None'),
            'sharpen': data.get('sharpen', False),
            'edge_enhance': data.get('edge_enhance', False),
            'hist_eq': data.get('hist_eq', False),
            'clahe': data.get('clahe', False),
            'clip_limit': data.get('clip_limit', 2.0),
            'tile_grid_size': data.get('tile_grid_size', 8),
            'local_contrast': data.get('local_contrast', False),
            'kernel_size': data.get('kernel_size', 9),
            'screen_tone_1': data.get('screen_tone_1', False),
            'screen_tone_pattern_1': data.get('screen_tone_pattern_1', 'None'),
            'screen_tone_size_1': data.get('screen_tone_size_1', 2),
            'screen_tone_density_1': data.get('screen_tone_density_1', 50),
            'screen_tone_area_pattern_1': data.get('screen_tone_area_pattern_1', 'Global Darkness'),
            'pencil_shading_style_1': data.get('pencil_shading_style_1', 'Light'),
            'screen_tone_color_1': data.get('screen_tone_color_1', '#323232'),
            'screen_tone_2': data.get('screen_tone_2', False),
            'screen_tone_pattern_2': data.get('screen_tone_pattern_2', 'None'),
            'screen_tone_size_2': data.get('screen_tone_size_2', 2),
            'screen_tone_density_2': data.get('screen_tone_density_2', 50),
            'screen_tone_area_pattern_2': data.get('screen_tone_area_pattern_2', 'Shadow Regions'),
            'pencil_shading_style_2': data.get('pencil_shading_style_2', 'Medium'),
            'screen_tone_color_2': data.get('screen_tone_color_2', '#505050'),
        }

        # Process the original image
        original_path = images[image_id]['original']
        processed_full = process_image(original_path, params, is_preview=False)

        # Save processed full image
        processed_full_path = os.path.join(TEMP_DIR, f'processed_full_{image_id}.png')
        processed_full.save(processed_full_path)

        # Send the file for download
        return send_file(processed_full_path, mimetype='image/png',
                         as_attachment=True, download_name='processed_image.png')
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return jsonify({'status': 'error', 'message': 'Error saving image.'}), 500

# เพิ่ม Error Handler เพื่อจับข้อผิดพลาดทั่วไป
@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return jsonify(error=e.description), e.code
    else:
        # Log ข้อผิดพลาดลงใน Logs
        logger.error(f"Unhandled Exception: {e}")
        return jsonify(error="Internal Server Error"), 500

# เพิ่ม Error Handler สำหรับ 413 Request Entity Too Large
@app.errorhandler(413)
def request_entity_too_large(error):
    return "ไฟล์ที่คุณอัปโหลดมีขนาดใหญ่เกินไป. โปรดลองอัปโหลดไฟล์ที่มีขนาดไม่เกิน 100MB.", 413

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
