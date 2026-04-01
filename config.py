import os

# Base directory for temporary files
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'static/uploads')
PROCESSED_FOLDER = os.environ.get('PROCESSED_FOLDER', 'static/processed')

# Make sure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Application configuration
SECRET_KEY = os.environ.get('SECRET_KEY', 'super-secret-key-change-me')
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size

# Default colors
SCREEN_TONE_COLOR_DEFAULT = (50, 50, 50)
SCREEN_TONE_COLOR_DEFAULT_2 = (80, 80, 80)

# Method configurations
THRESHOLD_METHODS = [
    "Global Thresholding",
    "Adaptive Mean Thresholding",
    "Adaptive Gaussian Thresholding",
    "Special Adaptive Thresholding"
]

NOISE_REDUCTION_METHODS = [
    "None", "Median Filter", "Bilateral Filter", "Gaussian Blur", "Non-local Means Denoising"
]

SCREEN_TONE_PATTERNS = [
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

SCREEN_TONE_AREA_PATTERNS = [
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

PENCIL_SHADING_STYLES = [
    "Light",
    "Medium",
    "Dark",
    "Hatched",
    "Cross-hatched",
    "Manga Style"
]
