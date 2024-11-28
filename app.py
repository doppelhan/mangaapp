import os
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import io
import base64
import logging

app = Flask(__name__)

# ตั้งค่าขนาดสูงสุดของไฟล์ที่สามารถอัปโหลดได้ (เช่น 50MB)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# ตั้งค่า Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables to store images and related data
original_image = None
processed_image = None
original_format = None
original_info = None

# Default screen tone colors
screen_tone_color = (50, 50, 50)
screen_tone_color2 = (80, 80, 80)

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
        laplacian = cv2.Laplacian(gray_image, cv2.CV_16S)
        abs_laplacian = cv2.convertScaleAbs(laplacian)
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
    mask_image = mask_image.resize((width, height), Image.NEAREST)
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
                    length = int(size + brightness * size)
                    draw.line((x, y, x + length, y + length), fill=line_color, width=1)
                    draw.line((x + length, y, x, y + length), fill=line_color, width=1)

    elif pattern == "Lines":
        min_spacing = max(2, 15 - size)
        max_spacing = max(5, 25 - size)
        spacing = int(max_spacing - (density / 100) * (max_spacing - min_spacing))
        spacing = max(spacing, 2)

        for y in range(0, height, spacing):
            if mask_array[y, :].any():
                brightness = brightness_array[y, :].mean()
                line_color = color + (int(255 * brightness),)
                draw.line((0, y, width, y), fill=line_color, width=1)

    elif pattern == "Vertical Lines":
        min_spacing = max(2, 15 - size)
        max_spacing = max(5, 25 - size)
        spacing = int(max_spacing - (density / 100) * (max_spacing - min_spacing))
        spacing = max(spacing, 2)

        for x in range(0, width, spacing):
            if mask_array[:, x].any():
                brightness = brightness_array[:, x].mean()
                line_color = color + (int(255 * brightness),)
                draw.line((x, 0, x, height), fill=line_color, width=1)

    elif pattern == "Horizontal Lines":
        min_spacing = max(2, 15 - size)
        max_spacing = max(5, 25 - size)
        spacing = int(max_spacing - (density / 100) * (max_spacing - min_spacing))
        spacing = max(spacing, 2)

        for y in range(0, height, spacing):
            if mask_array[y, :].any():
                brightness = brightness_array[y, :].mean()
                line_color = color + (int(255 * brightness),)
                draw.line((0, y, width, y), fill=line_color, width=1)

    elif pattern == "Checkerboard":
        step = size * 2
        for y in range(0, height, step):
            for x in range(0, width, step):
                if mask_array[y:y+step, x:x+step].any():
                    brightness = brightness_array[y:y+step, x:x+step].mean()
                    fill_color = color + (int(255 * brightness),)
                    if (x // step + y // step) % 2 == 0:
                        box = (x, y, x + size, y + size)
                        draw.rectangle(box, fill=fill_color)
                    else:
                        box = (x + size, y, x + step, y + size)
                        draw.rectangle(box, fill=fill_color)

    elif pattern == "Diagonal Stripes":
        spacing = size * 2
        for i in range(-height, width, spacing):
            brightness = brightness_array.mean()
            line_color = color + (int(255 * brightness),)
            draw.line([(i, 0), (i + height, height)], fill=line_color, width=1)

    elif pattern == "Spiral":
        center_x, center_y = width // 2, height // 2
        max_radius = min(center_x, center_y)
        for angle in np.linspace(0, 4 * np.pi, int(max_radius / size) * 10):
            radius = size * angle / (2 * np.pi)
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            if 0 <= x < width and 0 <= y < height and mask_array[y, x]:
                brightness = brightness_array[y, x]
                point_color = color + (int(255 * brightness),)
                draw.point((x, y), fill=point_color)

    elif pattern == "Pencil Shading":
        if pencil_style == "Light":
            # Light pencil shading implementation
            line_spacing = size * 2
            line_width = 1
            angle_range = (-np.pi/6, np.pi/6)
            for y in range(0, height, line_spacing):
                for x in range(0, width, line_spacing):
                    if mask_array[y, x]:
                        brightness = brightness_array[y, x]
                        opacity = int(50 + brightness * 205)
                        line_color = color + (opacity,)
                        angle = np.random.uniform(angle_range[0], angle_range[1])
                        length = int(size + brightness * size * 2)
                        dx = length * np.cos(angle)
                        dy = length * np.sin(angle)
                        x_end = x + dx
                        y_end = y + dy
                        draw.line((x - dx, y - dy, x_end, y_end), fill=line_color, width=line_width)

        elif pencil_style == "Medium":
            # Medium pencil shading implementation
            line_spacing = size
            line_width = 1
            angle_range = (-np.pi/4, np.pi/4)
            for y in range(0, height, line_spacing):
                for x in range(0, width, line_spacing):
                    if mask_array[y, x]:
                        brightness = brightness_array[y, x]
                        opacity = int(70 + brightness * 185)
                        line_color = color + (opacity,)
                        angle = np.random.uniform(angle_range[0], angle_range[1])
                        length = int(size + brightness * size * 2)
                        dx = length * np.cos(angle)
                        dy = length * np.sin(angle)
                        x_end = x + dx
                        y_end = y + dy
                        draw.line((x - dx, y - dy, x_end, y_end), fill=line_color, width=line_width)

        elif pencil_style == "Dark":
            # Dark pencil shading implementation
            line_spacing = max(1, size // 2)
            line_width = 2
            angle_range = (-np.pi/2, np.pi/2)
            for y in range(0, height, line_spacing):
                for x in range(0, width, line_spacing):
                    if mask_array[y, x]:
                        brightness = brightness_array[y, x]
                        opacity = int(90 + brightness * 165)
                        line_color = color + (opacity,)
                        angle = np.random.uniform(angle_range[0], angle_range[1])
                        length = int(size + brightness * size * 2)
                        dx = length * np.cos(angle)
                        dy = length * np.sin(angle)
                        x_end = x + dx
                        y_end = y + dy
                        draw.line((x - dx, y - dy, x_end, y_end), fill=line_color, width=line_width)

        elif pencil_style == "Hatched":
            # Hatched pencil shading implementation
            line_spacing = size
            line_width = 1
            fixed_angle = np.pi / 4  # 45 degrees
            for y in range(0, height, line_spacing):
                for x in range(0, width, line_spacing):
                    if mask_array[y, x]:
                        brightness = brightness_array[y, x]
                        opacity = int(70 + brightness * 185)
                        line_color = color + (opacity,)
                        angle = fixed_angle
                        length = int(size + brightness * size * 2)
                        dx = length * np.cos(angle)
                        dy = length * np.sin(angle)
                        x_end = x + dx
                        y_end = y + dy
                        draw.line((x - dx, y - dy, x_end, y_end), fill=line_color, width=line_width)

        elif pencil_style == "Cross-hatched":
            # Cross-hatched pencil shading implementation
            line_spacing = size
            line_width = 1
            angles = [np.pi / 4, -np.pi / 4]
            for angle in angles:
                for y in range(0, height, line_spacing):
                    for x in range(0, width, line_spacing):
                        if mask_array[y, x]:
                            brightness = brightness_array[y, x]
                            opacity = int(70 + brightness * 185)
                            line_color = color + (opacity,)
                            length = int(size + brightness * size * 2)
                            dx = length * np.cos(angle)
                            dy = length * np.sin(angle)
                            x_end = x + dx
                            y_end = y + dy
                            draw.line((x - dx, y - dy, x_end, y_end), fill=line_color, width=line_width)

        elif pencil_style == "Manga Style":
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

        else:
            # Default pencil shading implementation
            for y in range(0, height, size):
                for x in range(0, width, size):
                    if mask_array[y, x]:
                        brightness = brightness_array[y, x]
                        opacity = int(70 + brightness * 185)
                        line_color = color + (opacity,)
                        angle = np.random.uniform(-np.pi/4, np.pi/4)
                        length = int(size + brightness * size * 2)
                        dx = length * np.cos(angle)
                        dy = length * np.sin(angle)
                        x_end = x + dx
                        y_end = y + dy
                        draw.line((x - dx, y - dy, x_end, y_end), fill=line_color, width=1)

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
def process_image(params):
    global processed_image, original_image, screen_tone_color, screen_tone_color2

    # Extract parameters from params
    threshold_value = int(params.get('threshold', 128))
    contrast_value = float(params.get('contrast', 1.0))
    brightness_value = float(params.get('brightness', 1.0))
    gamma_value = float(params.get('gamma', 1.0))
    exposure_value = float(params.get('exposure', 1.0))
    method = params.get('method', 'Special Adaptive Thresholding')
    block_size = int(params.get('block_size', 11))
    c_value = int(params.get('c_value', 6))  # Default C Value set to 6
    invert = params.get('invert', 'false') == 'true'
    noise_reduction_method = params.get('noise_reduction', 'None')
    sharpen = params.get('sharpen', 'false') == 'true'
    edge_enhance = params.get('edge_enhance', 'false') == 'true'
    hist_eq = params.get('hist_eq', 'false') == 'true'
    clahe = params.get('clahe', 'false') == 'true'
    clip_limit = float(params.get('clip_limit', 2.0))
    tile_grid_size = int(params.get('tile_grid_size', 8))
    local_contrast = params.get('local_contrast', 'false') == 'true'
    kernel_size = int(params.get('kernel_size', 9))

    # Screen Tone Layer 1
    screen_tone_1 = params.get('screen_tone_1', 'false') == 'true'
    screen_tone_pattern_1 = params.get('screen_tone_pattern_1', 'None')
    screen_tone_size_1 = int(params.get('screen_tone_size_1', 2))  # Default Size set to 2
    screen_tone_density_1 = int(params.get('screen_tone_density_1', 50))  # Default Density set to 50
    screen_tone_area_pattern_1 = params.get('screen_tone_area_pattern_1', 'Global Darkness')
    pencil_shading_style_1 = params.get('pencil_shading_style_1', 'Light')
    screen_tone_color_1 = params.get('screen_tone_color_1', '#323232')
    screen_tone_color_1 = tuple(int(screen_tone_color_1.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    # Screen Tone Layer 2
    screen_tone_2 = params.get('screen_tone_2', 'false') == 'true'
    screen_tone_pattern_2 = params.get('screen_tone_pattern_2', 'None')
    screen_tone_size_2 = int(params.get('screen_tone_size_2', 2))  # Default Size set to 2
    screen_tone_density_2 = int(params.get('screen_tone_density_2', 50))  # Default Density set to 50
    screen_tone_area_pattern_2 = params.get('screen_tone_area_pattern_2', 'Shadow Regions')
    pencil_shading_style_2 = params.get('pencil_shading_style_2', 'Medium')
    screen_tone_color_2 = params.get('screen_tone_color_2', '#505050')
    screen_tone_color_2 = tuple(int(screen_tone_color_2.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    # Set threshold_type to THRESH_BINARY
    threshold_type = cv2.THRESH_BINARY

    # Process the image
    if original_image is None:
        return None

    img = original_image.copy()

    # Resize image if it's too large to prevent excessive processing time
    MAX_RESOLUTION = (2000, 2000)  # ตัวอย่างขนาดสูงสุด
    img.thumbnail(MAX_RESOLUTION, Image.ANTIALIAS)

    # Adjust Contrast and Brightness
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
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

    # Add pixel dimensions to images
    def add_dimensions(img, size):
        draw = ImageDraw.Draw(img)
        font_size = max(20, size[0] // 30)  # ปรับขนาดฟอนต์ตามขนาดภาพ
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        text = f"{size[0]} x {size[1]} pixels"
        text_width, text_height = draw.textsize(text, font=font)
        draw.rectangle([(0,0), (text_width + 10, text_height + 10)], fill=(255, 255, 255, 128))
        draw.text((5, 5), text, fill=(0, 0, 0), font=font)
        return img

    original_img_with_dims = add_dimensions(original_image.copy(), original_image.size)
    processed_img_with_dims = add_dimensions(processed_image.copy(), processed_image.size)

    # Combine original and processed images side by side
    def combine_images(img1, img2):
        combined_width = img1.width + img2.width + 20  # 20px เว้นระยะห่าง
        combined_height = max(img1.height, img2.height)
        combined_img = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (img1.width + 20, 0))
        return combined_img

    final_img = combine_images(original_img_with_dims, processed_img_with_dims)

    # Convert final image to base64
    final_img_str = image_to_base64(final_img)

    # Store the processed image
    processed_image = final_img

    # Return Base64 for display
    return final_img_str

@app.route('/', methods=['GET', 'POST'])
def index():
    global original_image, original_format, original_info

    if request.method == 'POST':
        # Receive image file from user
        file = request.files['image']
        if file:
            try:
                img = Image.open(file.stream)
                original_image = img.convert('RGB')
                original_format = img.format
                original_info = img.info
                original_img_str = image_to_base64(original_image)
                return render_template('index.html', original_image=original_img_str,
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
            except Exception as e:
                logger.error(f"Error processing uploaded image: {e}")
                return "Error processing the uploaded image.", 500
    else:
        return render_template('index.html', threshold_methods=threshold_methods,
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
def process():
    params = request.json
    img_str = process_image(params)
    if img_str:
        return jsonify({'status': 'done', 'image': img_str})
    else:
        return jsonify({'status': 'error'}), 500

@app.route('/save_image')
def save_image():
    global processed_image
    if processed_image:
        buffered = io.BytesIO()
        processed_image.save(buffered, format='PNG')
        buffered.seek(0)
        return send_file(buffered, mimetype='image/png', as_attachment=True,
                         download_name='processed_image.png')
    else:
        return 'No image to save.', 400

# เพิ่ม Error Handler เพื่อจับข้อผิดพลาดทั่วไป
@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return jsonify(error=e.description), e.code
    else:
        # Log ข้อผิดพลาดลงใน Logs
        logger.error(f"Unhandled Exception: {e}")
        return jsonify(error="Internal Server Error"), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
