import numpy as np
import cv2
from PIL import Image, ImageDraw
from .gpu_core import HAS_GPU, to_gpu, to_cpu, get_backend

def apply_screen_tone(image, size=5, pattern="None", mask=None, gray_image=None, color=(50, 50, 50), density=50, pencil_style="Light"):
    """Applies screen tone patterns using optimized NumPy operations where possible."""
    if pattern == "None" or mask is None:
        return image

    # We convert image to a numpy array (RGBA)
    if not isinstance(image, Image.Image):
        return image

    img_array = np.array(image.convert("RGBA"))
    height, width = img_array.shape[:2]

    mask_uint8 = (mask * 255).astype(np.uint8)
    if mask_uint8.shape[:2] != (height, width):
        mask_uint8 = cv2.resize(mask_uint8, (width, height), interpolation=cv2.INTER_NEAREST)

    if gray_image is not None:
        gray_image_resized = cv2.resize(gray_image, (width, height), interpolation=cv2.INTER_NEAREST)
        brightness_array = 255 - gray_image_resized
        brightness_array = brightness_array / 255.0
    else:
        brightness_array = np.ones((height, width))

    # Create an overlay for the pattern
    overlay = np.zeros((height, width, 4), dtype=np.uint8)

    # Pre-calculate spacing and conditions
    min_spacing = max(2, 15 - size)
    max_spacing = max(5, 25 - size)
    spacing = int(max_spacing - (density / 100) * (max_spacing - min_spacing))
    spacing = max(spacing, 2)
    step = size * 2 if size * 2 > 0 else 1

    # We attempt to vectorize the simple ones:
    y_indices, x_indices = np.where(mask_uint8 > 0)

    if pattern in ["Dots", "Halftone Circles"]:
        # Faster CPU approach: We still iterate but only over the active grid points
        # To avoid PIL overhead, we draw on OpenCV overlay
        max_radius = size
        min_radius = 1

        # Grid logic
        y_grid = np.arange(0, height, step)
        x_grid = np.arange(0, width, step)

        for y in y_grid:
            for x in x_grid:
                if mask_uint8[y, x]:
                    b = brightness_array[y, x]
                    r = int(min_radius + b * (max_radius - min_radius))
                    if r > 0:
                        alpha = int(255 * b)
                        dot_color = color + (alpha,)
                        # Draw circle on overlay
                        cv2.circle(overlay, (x, y), r, dot_color, -1)

    elif pattern in ["Vertical Lines", "Horizontal Lines", "Lines"]:
        if pattern in ["Lines", "Horizontal Lines"]:
            y_grid = np.arange(0, height, spacing)
            for y in y_grid:
                if np.any(mask_uint8[y, :]):
                    b = np.mean(brightness_array[y, :][mask_uint8[y, :] > 0])
                    alpha = int(255 * b)
                    cv2.line(overlay, (0, y), (width, y), color + (alpha,), 1)
        else:
            x_grid = np.arange(0, width, spacing)
            for x in x_grid:
                if np.any(mask_uint8[:, x]):
                    b = np.mean(brightness_array[:, x][mask_uint8[:, x] > 0])
                    alpha = int(255 * b)
                    cv2.line(overlay, (x, 0), (x, height), color + (alpha,), 1)

    elif pattern == "Checkerboard":
        for y in range(0, height, step):
            for x in range(0, width, step):
                # Ensure we don't go out of bounds
                end_y = min(y + step, height)
                end_x = min(x + step, width)

                roi_mask = mask_uint8[y:end_y, x:end_x]
                if np.any(roi_mask):
                    roi_bright = brightness_array[y:end_y, x:end_x]
                    b = np.mean(roi_bright[roi_mask > 0])
                    alpha = int(255 * b)

                    if (x // step + y // step) % 2 == 0:
                        cv2.rectangle(overlay, (x, y), (min(x+size, width), min(y+size, height)), color + (alpha,), -1)
                    else:
                        cv2.rectangle(overlay, (min(x+size, width), y), (min(x+step, width), min(y+size, height)), color + (alpha,), -1)

    elif pattern == "Diagonal Stripes":
        sp = size * 2 if size * 2 > 0 else 1
        b_mean = np.mean(brightness_array[mask_uint8 > 0]) if np.any(mask_uint8 > 0) else 0.5
        alpha = int(255 * b_mean)
        for i in range(-height, width, sp):
            # We draw across the whole image but mask it later
            cv2.line(overlay, (i, 0), (i + height, height), color + (alpha,), 1)

    # For complex drawing like pencil shading, scratching, we fall back to PIL as it's intricate
    # and doing it perfectly in OpenCV requires translating a lot of logic
    else:
        # Fallback to original logic for complex patterns (Hatching, Cross-Hatching, Pencil, Noise, etc)
        # We use PIL on the original image for these
        draw = ImageDraw.Draw(image, 'RGBA')
        if pattern == "Hatching":
             for y in range(0, height, spacing):
                 for x in range(0, width, spacing):
                     if mask_uint8[y, x]:
                         b = brightness_array[y, x]
                         line_color = color + (int(255 * b),)
                         length = int(size + b * size)
                         draw.line((x, y, x + length, y), fill=line_color, width=1)
                         draw.line((x, y, x + length, y + length), fill=line_color, width=1)
        elif pattern == "Cross-Hatching":
             for y in range(0, height, spacing):
                 for x in range(0, width, spacing):
                     if mask_uint8[y, x]:
                         b = brightness_array[y, x]
                         line_color = color + (int(255 * b),)
                         length = int(size + b * size)
                         draw.line((x, y, x + length, y + length), fill=line_color, width=1)
                         draw.line((x + length, y, x, y + length), fill=line_color, width=1)
        elif pattern == "Spiral":
             center_x, center_y = width // 2, height // 2
             max_radius = min(center_x, center_y)
             for angle in np.linspace(0, 4 * np.pi, int(max_radius / max(1, size)) * 10):
                 r = size * angle / (2 * np.pi)
                 x = int(center_x + r * np.cos(angle))
                 y = int(center_y + r * np.sin(angle))
                 if 0 <= x < width and 0 <= y < height and mask_uint8[y, x]:
                     b = brightness_array[y, x]
                     draw.point((x, y), fill=color + (int(255 * b),))
        elif pattern == "Pencil Shading":
            if pencil_style == "Light":
                line_spacing = size * 2
                angle_range = (-np.pi/6, np.pi/6)
                for y in range(0, height, line_spacing):
                    for x in range(0, width, line_spacing):
                        if mask_uint8[y, x]:
                            b = brightness_array[y, x]
                            line_color = color + (int(50 + b * 205),)
                            angle = np.random.uniform(*angle_range)
                            length = int(size + b * size * 2)
                            dx, dy = length * np.cos(angle), length * np.sin(angle)
                            draw.line((x - dx, y - dy, x + dx, y + dy), fill=line_color, width=1)
            elif pencil_style == "Medium":
                line_spacing = size
                angle_range = (-np.pi/4, np.pi/4)
                for y in range(0, height, line_spacing):
                    for x in range(0, width, line_spacing):
                        if mask_uint8[y, x]:
                            b = brightness_array[y, x]
                            line_color = color + (int(70 + b * 185),)
                            angle = np.random.uniform(*angle_range)
                            length = int(size + b * size * 2)
                            dx, dy = length * np.cos(angle), length * np.sin(angle)
                            draw.line((x - dx, y - dy, x + dx, y + dy), fill=line_color, width=1)
            elif pencil_style == "Dark":
                line_spacing = max(1, size // 2)
                angle_range = (-np.pi/2, np.pi/2)
                for y in range(0, height, line_spacing):
                    for x in range(0, width, line_spacing):
                        if mask_uint8[y, x]:
                            b = brightness_array[y, x]
                            line_color = color + (int(90 + b * 165),)
                            angle = np.random.uniform(*angle_range)
                            length = int(size + b * size * 2)
                            dx, dy = length * np.cos(angle), length * np.sin(angle)
                            draw.line((x - dx, y - dy, x + dx, y + dy), fill=line_color, width=2)
            elif pencil_style == "Hatched":
                line_spacing = size
                for y in range(0, height, line_spacing):
                    for x in range(0, width, line_spacing):
                        if mask_uint8[y, x]:
                            b = brightness_array[y, x]
                            line_color = color + (int(70 + b * 185),)
                            length = int(size + b * size * 2)
                            dx, dy = length * np.cos(np.pi/4), length * np.sin(np.pi/4)
                            draw.line((x - dx, y - dy, x + dx, y + dy), fill=line_color, width=1)
            elif pencil_style == "Cross-hatched" or pencil_style == "Manga Style":
                line_spacing = size
                angles = [np.pi/4, -np.pi/4] if pencil_style == "Cross-hatched" else [np.pi/6, -np.pi/6]
                base_opacity = 70 if pencil_style == "Cross-hatched" else 80
                for angle in angles:
                    for y in range(0, height, line_spacing):
                        for x in range(0, width, line_spacing):
                            if mask_uint8[y, x]:
                                b = brightness_array[y, x]
                                line_color = color + (int(base_opacity + b * (255-base_opacity)),)
                                length = int(size + b * size * 2) if pencil_style == "Cross-hatched" else int(size*2 + b * size * 2)
                                dx, dy = length * np.cos(angle), length * np.sin(angle)
                                draw.line((x - dx, y - dy, x + dx, y + dy), fill=line_color, width=1)
        elif pattern == "Scratch Lines":
            num_lines = int((width + height) * density / 1000)
            for _ in range(num_lines):
                x_start = np.random.randint(0, width)
                y_start = np.random.randint(0, height)
                if mask_uint8[y_start, x_start]:
                    b = brightness_array[y_start, x_start]
                    line_color = color + (int(255 * b * 0.5),)
                    length = np.random.randint(size * 5, size * 20)
                    angle = np.random.uniform(0, 2 * np.pi)
                    x_end = int(x_start + length * np.cos(angle))
                    y_end = int(y_start + length * np.sin(angle))
                    draw.line((x_start, y_start, x_end, y_end), fill=line_color, width=1)
        elif pattern == "Noise":
             num_dots = int(width * height * (density / 1000))
             # Vectorized approach for noise
             xs = np.random.randint(0, width, num_dots)
             ys = np.random.randint(0, height, num_dots)
             valid = mask_uint8[ys, xs] > 0
             xs, ys = xs[valid], ys[valid]
             for x, y in zip(xs, ys):
                 b = brightness_array[y, x]
                 draw.point((x, y), fill=color + (int(255 * b * 0.3),))
        elif pattern == "Fish Scale":
             r = size
             step_f = int(step * 0.75)
             for y in range(0, height, step_f):
                 offset = 0 if (y // step) % 2 == 0 else r
                 for x in range(-r, width + r, step):
                     if 0 <= y < height and 0 <= x + offset < width and mask_uint8[y, x + offset]:
                         b = brightness_array[y, x + offset]
                         bbox = (x + offset - r, y - r, x + offset + r, y + r)
                         draw.pieslice(bbox, start=0, end=180, fill=color + (int(255 * b),))
        elif pattern == "Scary Classic":
            num_shadows = int(width * height * density / 100000)
            for _ in range(num_shadows):
                x_c = np.random.randint(0, width)
                y_c = np.random.randint(0, height)
                if mask_uint8[y_c, x_c]:
                    b = brightness_array[y_c, x_c]
                    r = np.random.randint(size * 5, size * 15)
                    draw.ellipse((x_c - r, y_c - r, x_c + r, y_c + r), fill=color + (int(255*b*0.1),))
            num_lines = int((width + height) * density / 500)
            for _ in range(num_lines):
                x_s = np.random.randint(0, width)
                y_s = np.random.randint(0, height)
                if mask_uint8[y_s, x_s]:
                    b = brightness_array[y_s, x_s]
                    length = np.random.randint(size * 10, size * 30)
                    angle = np.random.uniform(0, 2 * np.pi)
                    x_e = int(x_s + length * np.cos(angle))
                    y_e = int(y_s + length * np.sin(angle))
                    draw.line((x_s, y_s, x_e, y_e), fill=color + (int(255*b*0.2),), width=1)

        return image

    # If we used the OpenCV overlay approach, we merge it with PIL
    if pattern in ["Dots", "Halftone Circles", "Vertical Lines", "Horizontal Lines", "Lines", "Checkerboard", "Diagonal Stripes"]:
        # Apply mask strictly (mostly needed for Diagonal Stripes which draws everywhere)
        mask_3d = np.repeat(mask_uint8[:, :, np.newaxis], 4, axis=2)
        overlay = np.where(mask_3d > 0, overlay, 0)

        # Convert to PIL
        overlay_pil = Image.fromarray(overlay, mode="RGBA")

        # Alpha composite
        image = image.convert("RGBA")
        image = Image.alpha_composite(image, overlay_pil)

    return image
