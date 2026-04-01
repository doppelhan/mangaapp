import cv2
import numpy as np

def adjust_gamma(image, gamma=1.0):
    """Adjusts gamma of an image."""
    invGamma = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** invGamma * 255
    table = np.clip(table, 0, 255).astype("uint8")
    return cv2.LUT(image, table)

def apply_base_adjustments(img, params):
    """Applies contrast, brightness, gamma, and initial gray conversion."""
    contrast_value = float(params.get('contrast', 1.0))
    brightness_value = float(params.get('brightness', 1.0))
    gamma_value = float(params.get('gamma', 1.0))
    exposure_value = float(params.get('exposure', 1.0))

    # Convert to OpenCV format (BGR)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Contrast and Brightness
    img_cv = cv2.convertScaleAbs(img_cv, alpha=contrast_value, beta=(brightness_value - 1) * 255)

    # Gamma Correction
    img_cv = adjust_gamma(img_cv, gamma=gamma_value)

    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Exposure Compensation
    gray = cv2.convertScaleAbs(gray, alpha=exposure_value, beta=0)

    return gray

def apply_enhancements(gray, params):
    """Applies histogram eq, CLAHE, and local contrast."""
    hist_eq = params.get('hist_eq', 'false') == 'true'
    clahe = params.get('clahe', 'false') == 'true'
    clip_limit = float(params.get('clip_limit', 2.0))
    tile_grid_size = int(params.get('tile_grid_size', 8))
    local_contrast = params.get('local_contrast', 'false') == 'true'
    kernel_size = int(params.get('kernel_size', 9))

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

    return gray

def apply_noise_reduction(gray, params):
    """Applies selected noise reduction method."""
    method = params.get('noise_reduction', 'None')
    if method == "Median Filter":
        return cv2.medianBlur(gray, 3)
    elif method == "Bilateral Filter":
        return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    elif method == "Gaussian Blur":
        return cv2.GaussianBlur(gray, (5, 5), 0)
    elif method == "Non-local Means Denoising":
        return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return gray

def apply_sharpen_and_edges(denoised, params):
    """Applies sharpening and edge enhancement."""
    sharpen = params.get('sharpen', 'false') == 'true'
    edge_enhance = params.get('edge_enhance', 'false') == 'true'

    # Sharpen
    if sharpen:
        gaussian = cv2.GaussianBlur(denoised, (0, 0), sigmaX=3)
        denoised = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)

    # Edge Enhance
    if edge_enhance:
        edges = cv2.Canny(denoised, threshold1=50, threshold2=150)
        denoised = cv2.bitwise_or(denoised, edges)

    return denoised

def apply_thresholding(denoised, gray, params):
    """Applies the selected thresholding method."""
    method = params.get('method', 'Global Thresholding')
    threshold_value = int(params.get('threshold', 128))
    block_size = int(params.get('block_size', 11))
    c_value = int(params.get('c_value', 2))
    invert = params.get('invert', 'false') == 'true'
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

    if invert:
        thresh = cv2.bitwise_not(thresh)

    return thresh
