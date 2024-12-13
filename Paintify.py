import os
import numpy as np
from PIL import Image, ImageDraw
import math
from noise import pnoise2

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
    dx = (pnoise2((x + epsilon) / scale, y / scale) - pnoise2((x - epsilon) / scale, y / scale)) / (2 * epsilon)
    dy = (pnoise2(x / scale, (y + epsilon) / scale) - pnoise2(x / scale, (y - epsilon) / scale)) / (2 * epsilon)
    return dx, dy

def generate_contour_mask(width, height, contour_scale=50.0, contour_octaves=6, 
                         contour_persistence=0.5, contour_lacunarity=2.0, 
                         contour_threshold=0.1, contour_line_width=1):
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for y in range(height):
        for x in range(width):
            value = pnoise2(x / contour_scale, 
                          y / contour_scale,
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
                        grain_intensity=0.1, contour_scale=50.0, contour_octaves=6,
                        contour_persistence=0.5, contour_lacunarity=2.0, contour_threshold=0.1,
                        contour_line_width=1, contour_opacity=0.3, rotation_noise_multiplier=1.5):
    size_factor = min(1.0, max(0.1, size_factor))
    scaled_grid = int(grid_size * resolution_factor)
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
                draw.polygon(rotated_points, fill=grainy_color)
            except IndexError:
                continue
    canvas = canvas.crop((padding, padding, canvas_width - padding, canvas_height - padding))
    return canvas

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
    params = {
        'grid_size': 30,
        'skew_angle': 12,
        'size_factor': 1.3,
        'resolution_factor': 0.20,
        'grain_intensity': 0.05,
        'contour_scale': 300,
        'contour_octaves': 14,
        'contour_persistence': 0.5,
        'contour_lacunarity': 2.0,
        'contour_threshold': 0.5,
        'contour_line_width': 4,
        'contour_opacity': 0.3
    }
    process_image(input_folder, output_folder, **params)