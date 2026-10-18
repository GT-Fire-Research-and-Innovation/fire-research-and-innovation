"""Reading GeoTIFFs.

Level 1 (prep). Used by prep/windows.py.
"""

import rasterio


def read_tiff(file_path):
    """Return (array (bands, rows, cols), rasterio profile) of one GeoTIFF. [released]"""
    with rasterio.open(file_path, 'r') as reader:
        profile = reader.profile
        tif_as_array = reader.read()
    return tif_as_array, profile
