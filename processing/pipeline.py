import cv2
import numpy as np
from PIL import Image

from .filters import (
    apply_base_adjustments,
    apply_enhancements,
    apply_noise_reduction,
    apply_sharpen_and_edges,
    apply_thresholding
)
from .masks import generate_mask
from .tones import apply_screen_tone

def process_image_pipeline(img, params):
    """Orchestrates the image processing pipeline."""

    # 1. Base Adjustments
    gray = apply_base_adjustments(img, params)

    # 2. Enhancements
    gray = apply_enhancements(gray, params)

    # 3. Noise Reduction
    denoised = apply_noise_reduction(gray, params)

    # 4. Sharpen and Edges
    denoised = apply_sharpen_and_edges(denoised, params)

    # 5. Thresholding
    thresh = apply_thresholding(denoised, gray, params)

    # 6. Convert back to PIL Image (RGB)
    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    processed_pil = Image.fromarray(thresh_rgb)

    # 7. Apply Screen Tone Layer 1
    screen_tone_1 = params.get('screen_tone_1', 'false') == 'true'
    screen_tone_pattern_1 = params.get('screen_tone_pattern_1', 'None')
    if screen_tone_1 and screen_tone_pattern_1 != "None":
        mask1 = generate_mask(gray, params.get('screen_tone_area_pattern_1', 'Global Darkness'))
        color_1 = params.get('screen_tone_color_1', '#323232')
        color_1 = tuple(int(color_1.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        processed_pil = apply_screen_tone(
            processed_pil,
            size=int(params.get('screen_tone_size_1', 5)),
            pattern=screen_tone_pattern_1,
            mask=mask1,
            gray_image=gray,
            color=color_1,
            density=int(params.get('screen_tone_density_1', 50)),
            pencil_style=params.get('pencil_shading_style_1', 'Light')
        )

    # 8. Apply Screen Tone Layer 2
    screen_tone_2 = params.get('screen_tone_2', 'false') == 'true'
    screen_tone_pattern_2 = params.get('screen_tone_pattern_2', 'None')
    if screen_tone_2 and screen_tone_pattern_2 != "None":
        mask2 = generate_mask(gray, params.get('screen_tone_area_pattern_2', 'Shadow Regions'))
        color_2 = params.get('screen_tone_color_2', '#505050')
        color_2 = tuple(int(color_2.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        processed_pil = apply_screen_tone(
            processed_pil,
            size=int(params.get('screen_tone_size_2', 7)),
            pattern=screen_tone_pattern_2,
            mask=mask2,
            gray_image=gray,
            color=color_2,
            density=int(params.get('screen_tone_density_2', 70)),
            pencil_style=params.get('pencil_shading_style_2', 'Medium')
        )

    return processed_pil
