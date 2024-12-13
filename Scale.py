import os
from PIL import Image

def process_image(input_folder, output_folder, resolution):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith('.png'):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            try:
                with Image.open(input_path) as img:
                    img = img.resize(resolution, Image.Resampling.NEAREST)
                    img.save(output_path, format="PNG")
                
                os.remove(input_path)
                print(f"Processed and moved: {file_name}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    input_folder = "input_images"
    output_folder = "output_images"
    target_resolution = (64, 64)

    process_image(input_folder, output_folder, target_resolution)
