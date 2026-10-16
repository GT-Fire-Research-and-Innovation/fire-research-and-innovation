"""Read source channels into one target-grid patch at a time.

Continuous fields use bilinear interpolation by default. Categorical land
cover and fire labels use nearest-neighbor. Fine-grid values from coarse
sources are interpolated/repeated measurements, not new observations.
"""

from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject


def read_raster_patch(
    path: str | Path,
    dst_crs,
    dst_transform: Affine,
    width: int,
    height: int,
    bands: Iterable[int] | None = None,
    categorical_bands: Iterable[int] = (),
    default_resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Reproject selected 1-based raster bands to `(bands, height, width)`."""
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {path}")
        selected = list(bands) if bands is not None else list(range(1, src.count + 1))
        if not selected:
            return np.zeros((0, height, width), dtype=np.float32)
        invalid = [band for band in selected if band < 1 or band > src.count]
        if invalid:
            raise ValueError(f"{path} has {src.count} bands; invalid selection {invalid}")
        categorical = set(categorical_bands)
        result = np.zeros((len(selected), height, width), dtype=np.float32)
        for out_index, band_index in enumerate(selected):
            reproject(
                source=rasterio.band(src, band_index),
                destination=result[out_index],
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=0.0,
                resampling=Resampling.nearest if band_index in categorical else default_resampling,
                init_dest_nodata=True,
            )
    return np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def read_prediction_features(
    day_path: str | Path,
    night_path: str | Path | None,
    firepred_path: str | Path,
    dst_crs,
    dst_transform: Affine,
    patch_size: int,
) -> np.ndarray:
    """Return the 27 prediction input channels for one spatial patch.

    Channel order matches the existing TS-SatFire prediction preprocessor:
    six VIIRS_Day bands, two VIIRS_Night bands (or zeros), then 19 FirePred
    bands. FirePred band 14 is categorical land cover; bands 5, 11, and 17 are
    angular values in the existing loader. All four use nearest-neighbor to
    avoid interpolating categories or averaging across the 0/360-degree wrap.
    """
    with rasterio.open(day_path) as day:
        if day.count < 8:
            raise ValueError(f"{day_path} has {day.count} bands; expected 8 including AF/BA labels")
    day_features = read_raster_patch(
        day_path, dst_crs, dst_transform, patch_size, patch_size, bands=range(1, 7)
    )

    night_bands = None
    if night_path is not None:
        with rasterio.open(night_path) as src:
            if src.count == 5:
                night_bands = (4, 5)  # equivalent to the old array slice [3:5]
            elif src.count >= 2:
                night_bands = (1, 2)
            else:
                # The reference preprocessing treats an unusable night pass
                # like a missing one: both night channels are zero-filled.
                night_bands = None
    if night_bands is None:
        night = np.zeros((2, patch_size, patch_size), dtype=np.float32)
    else:
        night = read_raster_patch(
            night_path, dst_crs, dst_transform, patch_size, patch_size, bands=night_bands
        )

    with rasterio.open(firepred_path) as src:
        if src.count != 19:
            raise ValueError(f"{firepred_path} has {src.count} bands; expected 19 FirePred bands")
    firepred = read_raster_patch(
        firepred_path,
        dst_crs,
        dst_transform,
        patch_size,
        patch_size,
        bands=range(1, 20),
        categorical_bands=(5, 11, 14, 17),
    )
    channels = np.concatenate((day_features, night, firepred), axis=0)
    if channels.shape != (27, patch_size, patch_size):
        raise ValueError(f"Expected 27 input channels; got {channels.shape}")
    return channels


def read_prediction_labels(
    day_path: str | Path,
    dst_crs,
    dst_transform: Affine,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read active-fire and burned-area masks using nearest-neighbor sampling."""
    with rasterio.open(day_path) as src:
        if src.count < 8:
            raise ValueError(f"{day_path} has {src.count} bands; expected label bands 7 and 8")
    labels = read_raster_patch(
        day_path,
        dst_crs,
        dst_transform,
        patch_size,
        patch_size,
        bands=(7, 8),
        default_resampling=Resampling.nearest,
    )
    active_fire = np.isfinite(labels[0]) & (labels[0] > 0)
    burned_area = np.isfinite(labels[1]) & (labels[1] > 0)
    return active_fire, burned_area


def add_temporal_maxima(features: np.ndarray, running_max: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Preserve the legacy preprocessing's running maxima for channels 3:5,6:8."""
    running_max = np.maximum(running_max, features)
    result = features.copy()
    result[3:5] = running_max[3:5]
    result[6:8] = running_max[6:8]
    return result, running_max
