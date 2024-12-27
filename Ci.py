import json
import numpy as np
import os
import matplotlib.pyplot as plt

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

def process_directory(input_dir, output_dir, label_value=100):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Walk through all files in the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]

                # Match the JSON file with corresponding PNG
                png_name = f"{base_name}.png"
                png_path = os.path.join(root, png_name)

                if os.path.exists(png_path):  # Ensure the PNG exists
                    output_png_path = os.path.join(output_dir, f"{base_name}_mask.png")
                    create_mask_from_labelme(json_path, output_png_path, label_value)
                else:
                    print(f"Warning: PNG file not found for {json_path}")

# Example usage
input_directory = "path/to/your/input_folder"
output_directory = "path/to/your/output_folder"
process_directory(input_directory, output_directory)
