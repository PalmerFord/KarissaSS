import os
import random
import numpy as np
from PIL import Image, ImageDraw
import math
import colorsys
from noise import pnoise2

NOISE_OFFSET_X = random.uniform(0, 1000)
NOISE_OFFSET_Y = random.uniform(0, 1000)

def generate_random_color(monochrome_value, constraints=None):
    value = monochrome_value / 255.0

    hue_range = (0, 1)
    sat_range = (0, 1)
    val_range = (0, 1)

    if constraints:
        hue_range = constraints.get("hue_range", hue_range)
        sat_range = constraints.get("saturation_range", sat_range)
        val_range = constraints.get("value_range", val_range)

    hue = random.uniform(*hue_range)
    saturation = random.uniform(*sat_range)
    brightness = max(min(value, val_range[1]), val_range[0])

    rgb = colorsys.hsv_to_rgb(hue, saturation, brightness)
    return tuple(int(c * 255) for c in rgb)

def reduce_grayscale_levels(img, palette_size):
    factor = 255 / (palette_size - 1)
    pixels = np.array(img)
    reduced_pixels = (pixels // factor) * factor
    return Image.fromarray(reduced_pixels.astype(np.uint8))

def recolor_palette_effect(img, palette_size=100, random_color_constraints=None):
    grayscale_img = img.convert("L")
    reduced_img = reduce_grayscale_levels(grayscale_img, palette_size)
    reduced_pixels = np.array(reduced_img)

    unique_levels = np.unique(reduced_pixels)
    color_map = {
        level: generate_random_color(level, random_color_constraints) for level in unique_levels
    }

    recolored_pixels = np.zeros((reduced_pixels.shape[0], reduced_pixels.shape[1], 3), dtype=np.uint8)
    for level, color in color_map.items():
        recolored_pixels[reduced_pixels == level] = color

    return Image.fromarray(recolored_pixels, "RGB")

def add_grain(color, grain_intensity=0.1):
    noise = np.random.normal(0, grain_intensity * 255, 3)
    color_with_grain = np.clip(np.array(color[:3]) + noise, 0, 255)
    return tuple(map(int, color_with_grain)) + (color[3],) if len(color) > 3 else tuple(map(int, color_with_grain))

def rotate_points(points, center, angle_degrees):
    angle_rad = math.radians(angle_degrees)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)

    rotated_points = []
    for px, py in points:
        dx = px - center[0]
        dy = py - center[1]
        rx = dx * cos_angle - dy * sin_angle
        ry = dx * sin_angle + dy * cos_angle
        rotated_points.append((rx + center[0], ry + center[1]))
    return rotated_points

def calculate_gradient(x, y, scale, epsilon=1e-3):
    dx = (pnoise2((x + epsilon + NOISE_OFFSET_X) / scale, 
                  (y + NOISE_OFFSET_Y) / scale) 
          - pnoise2((x - epsilon + NOISE_OFFSET_X) / scale, 
                    (y + NOISE_OFFSET_Y) / scale)) / (2 * epsilon)
    dy = (pnoise2((x + NOISE_OFFSET_X) / scale, 
                  (y + epsilon + NOISE_OFFSET_Y) / scale) 
          - pnoise2((x + NOISE_OFFSET_X) / scale, 
                    (y - epsilon + NOISE_OFFSET_Y) / scale)) / (2 * epsilon)
    return dx, dy

def generate_contour_mask(width, height, contour_scale=50.0, contour_octaves=6, 
                          contour_persistence=0.5, contour_lacunarity=2.0, 
                          contour_threshold=0.1, contour_line_width=1):
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for y in range(height):
        for x in range(width):
            value = pnoise2((x + NOISE_OFFSET_X) / contour_scale, 
                            (y + NOISE_OFFSET_Y) / contour_scale,
                            octaves=contour_octaves,
                            persistence=contour_persistence,
                            lacunarity=contour_lacunarity)
            value = (value + 1) / 2
            for i in range(-contour_line_width + 1, contour_line_width):
                if abs((value + i/100) % contour_threshold) < 0.005:
                    draw.point((x, y), fill=128)
                    break
    return mask


def convert_to_geometric(img, grid_size=10, skew_angle=15, size_factor=1.0, resolution_factor=2.0,
                        grain_intensity=0.1, contour_scale=50.0, rotation_noise_multiplier=1.5,
                        monochrome=False, monochrome_before_noise=True, enable_recolor_palette=False,
                        palette_size=100, random_color_constraints=None):
    size_factor = min(1.0, max(0.1, size_factor))
    scaled_grid = int(grid_size * resolution_factor)

    if monochrome and monochrome_before_noise:
        img = img.convert("L").convert("RGB")

    img = img.convert("RGB")
    
    base_width = img.width
    base_height = img.height
    scaled_width = int(base_width * resolution_factor)
    scaled_height = int(base_height * resolution_factor)
    img_resized = img.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
    skew_rad = math.radians(skew_angle)

    dx = math.cos(math.pi/4 + skew_rad) * scaled_grid * size_factor
    dy = math.sin(math.pi/4 + skew_rad) * scaled_grid * size_factor
    
    padding = int(max(abs(dx), abs(dy)) * 2)
    canvas_width = scaled_width + 2 * padding
    canvas_height = scaled_height + 2 * padding
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas, 'RGBA')
    pixels = np.array(img_resized)
    step_x = scaled_grid // 2
    step_y = scaled_grid // 2

    for row in range(-2, (scaled_height // step_y) + 4):
        row_offset = row * step_y
        is_odd_row = row % 2
        for col in range(-2, (scaled_width // step_x) + 4):
            center_x = col * step_x + padding
            center_y = row_offset + padding

            if is_odd_row:
                center_x += step_x

            rhombus_points = [
                (center_x, center_y - dy),
                (center_x + dx, center_y),
                (center_x, center_y + dy),
                (center_x - dx, center_y),
            ]

            sample_x = (center_x - padding) / resolution_factor
            sample_y = (center_y - padding) / resolution_factor
            grad_x, grad_y = calculate_gradient(sample_x, sample_y, contour_scale)

            if grad_x == 0 and grad_y == 0:
                rotation_angle = 0
            else:
                rotation_angle = (math.degrees(math.atan2(grad_y, grad_x)) + 90) * rotation_noise_multiplier

            rotated_points = rotate_points(rhombus_points, (center_x, center_y), rotation_angle)

            try:
                base_color = tuple(pixels[min(int(center_y - padding), pixels.shape[0]-1), 
                                           min(int(center_x - padding), pixels.shape[1]-1)])
                color_with_alpha = base_color + (255,)
                grainy_color = add_grain(color_with_alpha, grain_intensity)

                if monochrome and monochrome_before_noise == False:
                    grayscale_value = int(0.2989 * grainy_color[0] + 0.587 * grainy_color[1] + 0.114 * grainy_color)
                    grainy_color = (grayscale_value, grayscale_value, grayscale_value, grainy_color[3])

                draw.polygon(rotated_points, fill=grainy_color)
            except Exception as e:
                print(f"Error drawing polygon at ({center_x}, {center_y}): {e}")

    if enable_recolor_palette:
        recolored_canvas = recolor_palette_effect(canvas, palette_size, random_color_constraints)
        canvas = Image.alpha_composite(canvas.convert("RGBA"), recolored_canvas.convert("RGBA"))

    final_image = canvas.crop((padding, padding, canvas_width - padding, canvas_height - padding))
    return final_image

def process_image(input_folder, output_folder, **kwargs):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, f"processed_{file_name}")
            try:
                with Image.open(input_path) as img:
                    processed_img = convert_to_geometric(img, **kwargs)
                    processed_img.save(output_path, format="PNG", quality=95)
                print(f"Processed: {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    input_folder = "input_images"
    output_folder = "output_images"
    random_color_constraints = {
        "hue_range": (0.2, 0.7),
        "saturation_range": (0, 0.95),
        "value_range": (0.1, 0.9),
    }
    params = {
        'grid_size': 20,
        'skew_angle': 12,
        'size_factor': 1.3,
        'resolution_factor': 0.30,
        'grain_intensity': 0.03,
        'contour_scale': 150,
        'monochrome': False,
        'monochrome_before_noise': True,
        'enable_recolor_palette': False,
        'palette_size': 256,
        'random_color_constraints': random_color_constraints
    }

    process_image(input_folder, output_folder, **params)