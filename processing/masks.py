import cv2
import numpy as np

def generate_mask(gray_image, pattern):
    """Generates a boolean mask based on the selected area pattern."""
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
