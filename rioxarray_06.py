import rioxarray as rxr
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Function to check the image type (Panchromatic or Multispectral)
def get_image_type(data):
    num_bands = data.shape[0]  # Number of bands (band, height, width)
    if num_bands == 1:
        return "Panchromatic"
    else:
        return "Multispectral"

# Function to pad the image so its size is a multiple of the tile_size
def pad_image_to_tile_size(data, tile_size):
    height, width = data.shape[1], data.shape[2]
    tile_height, tile_width = tile_size

    # Calculate the new dimensions of the image
    new_height = ((height // tile_height) + 1) * tile_height
    new_width = ((width // tile_width) + 1) * tile_width

    # Create a new padded image
    padded_data = np.pad(data.values, ((0, 0), (0, new_height - height), (0, new_width - width)), mode='constant', constant_values=0)
    
    # Return a DataArray with the new dimensions
    return data.copy(data=padded_data)

# Function to crop the image into smaller tiles and save them as PNG files
def crop_and_save_tiles(file_path, output_dir, tile_size=(500, 500)):
    # Open the image with rioxarray
    data = rxr.open_rasterio(file_path, masked=True)

    # Identify the image type
    image_type = get_image_type(data)
    print(f"Image type: {image_type} ({data.shape[0]} band{'s' if data.shape[0] > 1 else ''})")

    # Check dimension names
    print("Dimension names:", data.dims)

    # Check band names
    print("Band names:", data.coords['band'].values)

    # Create an output directory based on the image file name
    base_name = Path(file_path).stem
    image_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(image_output_dir, exist_ok=True)

    # Pad the image so its dimensions are a multiple of the tile_size
    data = pad_image_to_tile_size(data, tile_size)

    # Image dimensions after padding
    height, width = data.shape[1], data.shape[2]
    print(f"Padded image size: {height}x{width}")

    # Tile dimensions
    tile_height, tile_width = tile_size

    # Crop the image into tiles and save them
    for i in range(0, height, tile_height):
        for j in range(0, width, tile_width):
            # Crop the image
            tile = data.isel(
                x=slice(j, j + tile_width),
                y=slice(i, i + tile_height)
            )

            # Generate the tile file name
            tile_name = f"{base_name}_tile_{i}_{j}.png"
            tile_path = os.path.join(image_output_dir, tile_name)

            # Process Panchromatic images
            if image_type == "Panchromatic":
                tile_numpy = tile.squeeze().values  # Convert to numpy array
                plt.imsave(tile_path, tile_numpy, cmap='gray')  # Save grayscale image
            elif image_type == "Multispectral":
                # Check for the 'band' dimension to select bands
                if 'band' in data.dims:
                    print("Using 'band' dimension")
                    # Ensure there are at least 3 bands
                    if data.shape[0] >= 3:
                        # Select band names instead of indices
                        bands_to_select = data.coords['band'].values[:3]  # Select the first 3 bands
                        print(f"Selecting bands: {bands_to_select}")
                        tile_numpy = tile.sel(band=bands_to_select).values  # Select the first 3 bands
                        tile_numpy = np.moveaxis(tile_numpy, 0, -1)  # Rearrange axes to (height, width, 3)

                        # Ensure the image is uint8 if within [0, 255]
                        if tile_numpy.max() <= 255:
                            tile_numpy = tile_numpy.astype(np.uint8)  # Convert to uint8
                        
                        # Clip values outside the [0, 255] range
                        tile_numpy = np.clip(tile_numpy, 0, 255)
                        
                        plt.imsave(tile_path, tile_numpy)  # Save RGB image
                    else:
                        print("Warning: Not enough bands for RGB.")
                else:
                    print("No 'band' dimension found.")
            print(f"Saved: {tile_path}")

# Function to process all .tif images in a directory
def process_directory(input_dir, output_dir, tile_size=(500, 500)):
    # Iterate through all files in the directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".tif"):  # Check file format
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                crop_and_save_tiles(file_path, output_dir, tile_size)

# Input and output directory paths
input_dir = 'C:/Users/vothi/python/test_tif/images'  # Replace with your input directory path
output_dir = 'C:/Users/vothi/python/test_tif/images'  # Replace with your output directory path

# Process images in the directory
process_directory(input_dir, output_dir)
