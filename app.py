import os
import base64
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image, ImageDraw
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.secret_key = 'your_secret_key'

# สร้างโฟลเดอร์ถ้ายังไม่มี
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# รายการของ Threshold Methods
threshold_methods = [
    "Global Thresholding",
    "Adaptive Mean Thresholding",
    "Adaptive Gaussian Thresholding",
    "Special Adaptive Thresholding"
]

# รายการของ Noise Reduction Methods
noise_reduction_methods = [
    "None", "Median Filter", "Bilateral Filter", "Gaussian Blur", "Non-local Means Denoising"
]

# รายการของ Screen Tone Patterns
screen_tone_patterns = [
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

# รายการของ Pencil Shading Styles
pencil_shading_styles = [
    "Light",
    "Medium",
    "Dark",
    "Hatched",
    "Cross-hatched",
    "Manga Style"
]

# รายการของ Screen Tone Area Patterns
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

# ฟังก์ชันช่วยเหลือ
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'bmp'}

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** invGamma * 255
    table = np.clip(table, 0, 255).astype("uint8")
    return cv2.LUT(image, table)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    return tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

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
        mask = gray_image < 100  # ค่าเริ่มต้น

    return mask

def apply_screen_tone(image, size=5, pattern="None", mask=None, gray_image=None, color=(50, 50, 50), density=50, pencil_style="Light"):
    if pattern == "None" or mask is None:
        return image  # ไม่ทำการเพิ่มสกรีนโทน

    draw = ImageDraw.Draw(image, 'RGBA')  # ใช้โหมด RGBA เพื่อรองรับการโปร่งใส
    width, height = image.size

    # สร้างภาพมาสก์สำหรับการประมวลผล
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
    mask_image = mask_image.resize((width, height), Image.NEAREST)
    mask_array = np.array(mask_image)

    # เตรียมข้อมูลความสว่างของภาพ
    if gray_image is not None:
        gray_image_resized = cv2.resize(gray_image, (width, height), interpolation=cv2.INTER_NEAREST)
        brightness_array = 255 - gray_image_resized  # ยิ่งค่าสูง ยิ่งมืด
        brightness_array = brightness_array / 255.0  # นำไปใช้เป็นสัดส่วน
    else:
        brightness_array = np.ones((height, width))

    # เลือกแพทเทิร์น
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
        # เพิ่มการจัดการสไตล์ของ Pencil Shading ตามโค้ดต้นฉบับ
        pass  # นำโค้ดจากต้นฉบับมาใส่ในส่วนนี้

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # รับไฟล์ที่อัปโหลด
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(original_path)
            return redirect(url_for('process', filename=filename))
    return render_template('index.html')

@app.route('/process/<filename>', methods=['GET', 'POST'])
def process(filename):
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(original_path):
        return redirect(url_for('index'))

    # โหลดภาพต้นฉบับ
    original_image = Image.open(original_path).convert('RGB')

    if request.method == 'POST':
        # รับพารามิเตอร์จากฟอร์ม
        parameters = request.form.to_dict()
        # ประมวลผลภาพ
        processed_image = process_image(original_image, parameters)
        # บันทึกภาพที่ประมวลผล
        processed_filename = 'processed_' + filename
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        processed_image.save(processed_path)
    else:
        # เมื่อ GET ให้ประมวลผลด้วยพารามิเตอร์เริ่มต้น
        parameters = {}
        processed_image = process_image(original_image, parameters)
        processed_filename = 'processed_' + filename
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        processed_image.save(processed_path)

    # แปลงภาพเป็น base64 สำหรับการแสดงผล
    with open(original_path, "rb") as image_file:
        original_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    with open(processed_path, "rb") as image_file:
        processed_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    return render_template('process.html',
                           original_image=original_base64,
                           processed_image=processed_base64,
                           parameters=parameters,
                           threshold_methods=threshold_methods,
                           noise_reduction_methods=noise_reduction_methods,
                           screen_tone_patterns=screen_tone_patterns,
                           pencil_shading_styles=pencil_shading_styles,
                           screen_tone_area_patterns=screen_tone_area_patterns)

def process_image(image, parameters):
    # ดึงค่าพารามิเตอร์ด้วยค่าเริ่มต้น
    threshold_value = int(parameters.get('threshold', 128))
    contrast_value = float(parameters.get('contrast', 1.0))
    brightness_value = float(parameters.get('brightness', 1.0))
    gamma_value = float(parameters.get('gamma', 1.0))
    exposure_value = float(parameters.get('exposure', 1.0))
    method = parameters.get('method', 'Global Thresholding')
    block_size = int(parameters.get('block_size', 11))
    c_value = int(parameters.get('c_value', 2))
    invert = 'invert' in parameters
    noise_reduction_method = parameters.get('noise_reduction', 'None')
    sharpen = 'sharpen' in parameters
    edge_enhance = 'edge_enhance' in parameters
    hist_eq = 'hist_eq' in parameters
    clahe = 'clahe' in parameters
    clip_limit = float(parameters.get('clip_limit', 2.0))
    tile_grid_size = int(parameters.get('tile_grid_size', 8))
    local_contrast = 'local_contrast' in parameters
    kernel_size = int(parameters.get('kernel_size', 9))
    # พารามิเตอร์ของ Screen Tone Layer 1
    screen_tone_var = 'screen_tone_var' in parameters
    screen_tone_pattern = parameters.get('screen_tone_pattern', 'Dots')
    screen_tone_size = int(parameters.get('screen_tone_size', 5))
    screen_tone_density = int(parameters.get('screen_tone_density', 50))
    screen_tone_area_pattern = parameters.get('screen_tone_area_pattern', 'Global Darkness')
    screen_tone_color = parameters.get('screen_tone_color', '#323232')
    pencil_shading_style = parameters.get('pencil_shading_style', 'Light')
    # พารามิเตอร์ของ Screen Tone Layer 2
    screen_tone_var2 = 'screen_tone_var2' in parameters
    screen_tone_pattern2 = parameters.get('screen_tone_pattern2', 'Hatching')
    screen_tone_size2 = int(parameters.get('screen_tone_size2', 7))
    screen_tone_density2 = int(parameters.get('screen_tone_density2', 70))
    screen_tone_area_pattern2 = parameters.get('screen_tone_area_pattern2', 'Shadow Regions')
    screen_tone_color2 = parameters.get('screen_tone_color2', '#505050')
    pencil_shading_style2 = parameters.get('pencil_shading_style2', 'Medium')

    # แปลงภาพ PIL เป็นรูปแบบ OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # ปรับค่า Contrast และ Brightness
    img_cv = cv2.convertScaleAbs(img_cv, alpha=contrast_value, beta=(brightness_value - 1) * 255)
    # ปรับค่า Gamma Correction
    img_cv = adjust_gamma(img_cv, gamma=gamma_value)
    # แปลงภาพเป็นสีเทา
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # Exposure Compensation
    gray = cv2.convertScaleAbs(gray, alpha=exposure_value, beta=0)
    # Histogram Equalization
    if hist_eq:
        gray = cv2.equalizeHist(gray)
    # CLAHE
    if clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        gray = clahe_obj.apply(gray)
    # Local Contrast Enhancement
    if local_contrast:
        if kernel_size % 2 == 0:
            kernel_size += 1
        gaussian = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        gray = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
    # การลดสัญญาณรบกวน
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
    # Sharpen
    if sharpen:
        gaussian = cv2.GaussianBlur(denoised, (0, 0), sigmaX=3)
        denoised = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
    # Edge Enhance
    if edge_enhance:
        edges = cv2.Canny(denoised, threshold1=50, threshold2=150)
        denoised = cv2.bitwise_or(denoised, edges)
    # Thresholding
    threshold_type = cv2.THRESH_BINARY
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
    # Invert สี
    if invert:
        thresh = cv2.bitwise_not(thresh)
    # แปลงกลับเป็น RGB
    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    processed_pil = Image.fromarray(thresh_rgb)
    # เพิ่ม Screen Tone Layer 1
    if screen_tone_var:
        mask1 = generate_mask(gray, screen_tone_area_pattern)
        color1 = hex_to_rgb(screen_tone_color)
        processed_pil = apply_screen_tone(
            processed_pil,
            size=screen_tone_size,
            pattern=screen_tone_pattern,
            mask=mask1,
            gray_image=gray,
            color=color1,
            density=screen_tone_density,
            pencil_style=pencil_shading_style
        )
    # เพิ่ม Screen Tone Layer 2
    if screen_tone_var2:
        mask2 = generate_mask(gray, screen_tone_area_pattern2)
        color2 = hex_to_rgb(screen_tone_color2)
        processed_pil = apply_screen_tone(
            processed_pil,
            size=screen_tone_size2,
            pattern=screen_tone_pattern2,
            mask=mask2,
            gray_image=gray,
            color=color2,
            density=screen_tone_density2,
            pencil_style=pencil_shading_style2
        )
    return processed_pil

if __name__ == '__main__':
    app.run(debug=True)
