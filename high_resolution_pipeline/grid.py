"""Create exact shared target grids and patch transforms from GeoTIFF metadata."""

from dataclasses import dataclass
import math
from pathlib import Path

import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, array_bounds, from_origin
from rasterio.windows import Window
from rasterio.warp import transform as transform_coords, transform_bounds


@dataclass(frozen=True)
class GridSpec:
    """North-up projected grid covering one fire's current 256-pixel crop."""

    crs: CRS
    transform: Affine
    width: int
    height: int
    resolution_m: float
    source_crs: str
    source_transform: tuple[float, ...]
    source_pixel_size: tuple[float, float]
    source_crop_bounds: tuple[float, float, float, float]

    def to_dict(self) -> dict:
        return {
            "crs": self.crs.to_string(),
            "transform": list(tuple(self.transform)[:6]),
            "width": self.width,
            "height": self.height,
            "resolution_m": self.resolution_m,
            "source_crs": self.source_crs,
            "source_transform": list(self.source_transform[:6]),
            "source_pixel_size": list(self.source_pixel_size),
            "source_crop_bounds": list(self.source_crop_bounds),
        }


def _local_utm_crs(bounds_lonlat: tuple[float, float, float, float]) -> CRS:
    """Select a local UTM CRS centered on the source crop."""
    west, south, east, north = bounds_lonlat
    lon, lat = (west + east) / 2, (south + north) / 2
    zone = max(1, min(60, int(math.floor((lon + 180) / 6)) + 1))
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def build_target_grid(
    reference_path: str | Path,
    resolution_m: float = 10.0,
    crop_offset: int = 128,
    crop_size: int = 256,
) -> GridSpec:
    """Use the actual VIIRS GeoTIFF transform to define a meter-based grid.

    By default, the crop matches the rows/columns 128:384 used by the legacy
    preprocessor. Callers can select a different pixel offset and crop size.
    Its four corner bounds are transformed into a local UTM CRS; target
    width/height are then computed from the requested ground spacing.
    """
    if resolution_m < 10:
        raise ValueError("resolution_m below 10 m is finer than the supported TESSERA grid")
    if crop_offset < 0 or crop_size <= 0:
        raise ValueError("crop_offset must be nonnegative and crop_size must be positive")
    with rasterio.open(reference_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {reference_path}")
        if src.width < crop_offset + crop_size or src.height < crop_offset + crop_size:
            raise ValueError(
                f"Raster {reference_path} is {src.width}x{src.height}; "
                f"crop offset {crop_offset} size {crop_size} does not fit"
            )
        crop = Window(crop_offset, crop_offset, crop_size, crop_size)
        crop_transform = src.window_transform(crop)
        left, bottom, right, top = array_bounds(crop_size, crop_size, crop_transform)
        crop_bounds = (left, bottom, right, top)
        lonlat_bounds = transform_bounds(src.crs, "EPSG:4326", *crop_bounds, densify_pts=41)
        target_crs = _local_utm_crs(lonlat_bounds)
        west, south, east, north = transform_bounds(
            src.crs, target_crs, *crop_bounds, densify_pts=41
        )
        width = int(math.ceil((east - west) / resolution_m))
        height = int(math.ceil((north - south) / resolution_m))
        transform = from_origin(west, north, resolution_m, resolution_m)
        origin_x, origin_y = crop_transform * (0, 0)
        x1, y1 = crop_transform * (1, 0)
        x2, y2 = crop_transform * (0, 1)
        projected_x, projected_y = transform_coords(
            src.crs, target_crs,
            [origin_x, x1, x2],
            [origin_y, y1, y2],
        )
        pixel_x = math.hypot(projected_x[1] - projected_x[0], projected_y[1] - projected_y[0])
        pixel_y = math.hypot(projected_x[2] - projected_x[0], projected_y[2] - projected_y[0])
        return GridSpec(
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
            resolution_m=float(resolution_m),
            source_crs=src.crs.to_string(),
            source_transform=tuple(crop_transform),
            source_pixel_size=(pixel_x, pixel_y),
            source_crop_bounds=crop_bounds,
        )


def patch_transform(grid: GridSpec, row: int, col: int) -> Affine:
    """Return the affine transform for a patch whose origin is `(row, col)`."""
    return grid.transform * Affine.translation(col, row)


def estimated_pixels(grid: GridSpec) -> int:
    """Number of target-grid cells in the full fire crop."""
    return grid.width * grid.height
