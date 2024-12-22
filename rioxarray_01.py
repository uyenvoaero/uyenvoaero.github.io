import rioxarray as rxr
import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path

# Function to check image type (Panchromatic or Multispectral)
def get_image_type(data):
    num_bands = data.shape[0]
    if num_bands == 1:
        return "Panchromatic"
    else:
        return "Multispectral"

# Function to pad the image so its size is a multiple of tile_size
def pad_tile_to_size(tile, tile_size):
    tile_height, tile_width = tile.shape[1], tile.shape[2]
    tile_size_height, tile_size_width = tile_size

    # Calculate the required padding size
    pad_height = (tile_size_height - (tile_height % tile_size_height)) % tile_size_height
    pad_width = (tile_size_width - (tile_width % tile_size_width)) % tile_size_width

    # Pad each tile individually
    padded_tile = np.pad(tile.values, ((0, 0), (0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

    # Update coordinates for the padded tile
    new_coords = {
        'band': tile.coords['band'],
        'y': np.arange(padded_tile.shape[1]),
        'x': np.arange(padded_tile.shape[2])
    }

    padded_tile_array = xr.DataArray(padded_tile, dims=['band', 'y', 'x'], coords=new_coords)
    return padded_tile_array

# Function to crop the image into smaller tiles and save as PNG
def crop_and_save_tiles(file_path, output_dir, tile_size=(500, 500)):
    data = rxr.open_rasterio(file_path, masked=True)

    # Determine the image type
    image_type = get_image_type(data)
    print(f"Image type: {image_type} ({data.shape[0]} band{'s' if data.shape[0] > 1 else ''})")

    # Create output directory based on the image file name
    base_name = Path(file_path).stem
    image_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(image_output_dir, exist_ok=True)

    # Check the image size
    height, width = data.shape[1], data.shape[2]
    print(f"Original image size: {height}x{width}")

    # Crop the image into smaller tiles and save them
    tile_height, tile_width = tile_size
    for i in range(0, height, tile_height):
        for j in range(0, width, tile_width):
            tile = data.isel(x=slice(j, j + tile_width), y=slice(i, i + tile_height))

            # Pad the tile if necessary
            tile = pad_tile_to_size(tile, tile_size)

            # Name the tile file
            tile_name = f"{base_name}_tile_{i}_{j}.png"
            tile_path = os.path.join(image_output_dir, tile_name)

            # Process Panchromatic image
            if image_type == "Panchromatic":
                tile_numpy = tile.squeeze().values  # Convert to numpy array
                plt.imsave(tile_path, tile_numpy, cmap='gray')  # Save as grayscale image
            elif image_type == "Multispectral":
                if tile.shape[0] >= 3:
                    tile_numpy = tile.sel(band=[1, 2, 3]).values  # Take the first 3 bands
                    tile_numpy = np.moveaxis(tile_numpy, 0, -1)  # Move the band axis to the end
                    
                    # Normalize RGB values to the range 0-1
                    tile_numpy = tile_numpy / np.max(tile_numpy)  # Divide by the maximum value to normalize

                    # Save the normalized RGB image
                    plt.imsave(tile_path, tile_numpy)  # Save as RGB image
                else:
                    print("Warning: Not enough bands for RGB.")
            print(f"Saved: {tile_path}")

# Function to iterate through the directory and process all .tif images
def process_directory(input_dir, output_dir, tile_size=(500, 500)):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".tif"):
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                crop_and_save_tiles(file_path, output_dir, tile_size)

# Directory path containing the .tif files
input_dir = 'C:/Users/vothi/python/test_tif/画像'
output_dir = 'C:/Users/vothi/python/test_tif/画像'

# Process the images in the directory
process_directory(input_dir, output_dir)
