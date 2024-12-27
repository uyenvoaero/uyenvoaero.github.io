import json
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image  # Để xử lý ảnh gốc và lấy kích thước

def create_mask_from_labelme(json_path, output_png_path, label_value=100):
    # Load JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Get image dimensions
    img_shape = data['imageHeight'], data['imageWidth']

    # Initialize mask with zeros
    mask = np.zeros(img_shape, dtype=np.uint8)

    # Draw polygons onto the mask
    for shape in data['shapes']:
        points = np.array(shape['points'], dtype=np.int32)
        rr, cc = points[:, 1], points[:, 0]
        mask[rr.min():rr.max(), cc.min():cc.max()] = label_value

    # Save the mask using Matplotlib
    plt.imsave(output_png_path, mask, cmap='gray')
    print(f"Mask saved at {output_png_path}")

def create_empty_mask(png_path, output_png_path):
    # Load the original image to get its dimensions
    img = Image.open(png_path)
    img_shape = img.size[1], img.size[0]  # Height, Width (Pillow returns Width, Height)

    # Create an empty mask (all zeros)
    mask = np.zeros(img_shape, dtype=np.uint8)

    # Save the mask as a PNG
    plt.imsave(output_png_path, mask, cmap='gray')
    print(f"Empty mask saved at {output_png_path}")

def process_directory(input_dir, output_dir, label_value=100):
    # Walk through all files in the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.png'):  # Process PNG files
                png_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]

                # Check if corresponding JSON file exists
                json_path = os.path.join(root, f"{base_name}.json")
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                # Output mask path
                output_png_path = os.path.join(output_subdir, f"{base_name}_mask.png")

                if os.path.exists(json_path):
                    # Create mask using JSON if it exists
                    create_mask_from_labelme(json_path, output_png_path, label_value)
                else:
                    # Create an empty mask if no JSON is found
                    create_empty_mask(png_path, output_png_path)

# Example usage
input_directory = "path/to/your/input_folder"
output_directory = "path/to/your/output_folder"
process_directory(input_directory, output_directory)
