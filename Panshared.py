import rasterio
import numpy as np
from rasterio.enums import Resampling

def pansharpen(pan_band, ms_bands):
    pan_band = pan_band.astype(np.float32)
    ms_bands = ms_bands.astype(np.float32)
    ms_sum = np.sum(ms_bands, axis=0)
    pansharpened = [(ms / ms_sum) * pan_band for ms in ms_bands]
    return np.stack(pansharpened, axis=0)

# Load PAN and MS images
with rasterio.open("panchromatic.tif") as pan_src:
    pan_band = pan_src.read(1)

with rasterio.open("multispectral.tif") as ms_src:
    ms_bands = ms_src.read(
        out_shape=(ms_src.count, pan_band.shape[0], pan_band.shape[1]),
        resampling=Resampling.bilinear
    )

# Perform pansharpening
pansharpened = pansharpen(pan_band, ms_bands)

# Save the result
output_profile = ms_src.profile
output_profile.update(height=pan_band.shape[0], width=pan_band.shape[1], count=pansharpened.shape[0])
with rasterio.open("pansharpened_output.tif", "w", **output_profile) as dst:
    dst.write(pansharpened)
