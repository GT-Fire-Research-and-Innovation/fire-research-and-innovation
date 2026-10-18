"""Prior-year TESSERA embeddings averaged onto one fire's 256 x 256 VIIRS grid.

Level 1 (prep). Used by hub/publish_tessera.py. [ours]

prep/windows.py crops rows/cols 128:384 of every VIIRS_Day GeoTIFF, so one
256 x 256 grid covers all of a fire's windows. build_fire_tessera() reads that
grid's georeference, streams the TESSERA embeddings covering it, and averages
every 10 m TESSERA pixel into the crop pixel containing its centre. Each
tessera/p<id>.npz holds:

    embedding      (128, 256, 256) float16, 0 where coverage is 0
    coverage       (256, 256) float16, fraction of 10 m pixels with data
    fire_year, tessera_year, crs, transform, bounds, bounds_lonlat,
    source_store, model_version, geotessera_version, code_version, day_files

Embeddings come from the year before the fire: an annual embedding for the
fire year is built partly from post-fire imagery and would leak the burn scar
into the inputs. Fires whose prior year is not published (2017 fires) get none.
"""

import gc
import os
import subprocess
from importlib.metadata import version as package_version

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import Affine, array_bounds
from rasterio.warp import transform_bounds

from firepred.config.channels import TESSERA_BANDS
from firepred.config.dataset import CROP_OFFSET, CROP_SIZE, FIRST_TESSERA_YEAR, TESSERA_FORMAT_VERSION


def has_prior_year_tessera(fire_year):
    """True if the year before `fire_year` has published TESSERA embeddings."""
    return fire_year is not None and int(fire_year) - 1 >= FIRST_TESSERA_YEAR


def code_version():
    """Git commit of this repository, or "" when not run from a checkout."""
    try:
        return subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return ""


def crop_grid(profile, offset=CROP_OFFSET, size=CROP_SIZE):
    """Return (crs, transform) of the crop process_location takes from a GeoTIFF."""
    if profile["height"] < offset + size or profile["width"] < offset + size:
        raise ValueError(
            f"GeoTIFF is {profile['height']}x{profile['width']}; the "
            f"{size}x{size} crop at offset {offset} does not fit"
        )
    # Shift the GeoTIFF's geotransform by the crop offset: the result maps crop
    # pixel (row, col) to map coordinates, the same square prep/windows.py uses.
    return profile["crs"], profile["transform"] * Affine.translation(offset, offset)


def utm_zones(west, east):
    """UTM zone numbers spanned by a longitude range (no antimeridian wrap)."""
    # UTM zones are 6 degrees of longitude wide, numbered 1..60 from 180 W.
    # TESSERA tiles are stored per UTM zone, so a crop may need more than one zone.
    first = int(np.floor((west + 180) / 6)) + 1
    last = int(np.floor((np.nextafter(east, west) + 180) / 6)) + 1
    return list(range(max(1, first), min(60, last) + 1))


def grid_tessera(gt, dst_crs, dst_transform, year, size=CROP_SIZE, strip_rows=128):
    """Area-average one year of TESSERA onto a size x size raster grid.

    `gt` is a geotessera.GeoTesseraZarr. The region is streamed in row strips
    so a full-resolution crop (about 9600 x 9600 x 128 values) never sits in
    memory at once. Each 10 m pixel is assigned to the grid cell containing
    its centre; pixels without data (water, gaps) are left out of the mean.

    Returns (embedding, coverage, source_pixels, bounds_lonlat):
        embedding      (128, size, size) float32, NaN where coverage is 0
        coverage       (size, size) float32, valid / total 10 m pixels
        source_pixels  (size, size) int64, total 10 m pixels per cell
    """
    # Running totals per crop cell (flattened 256 x 256): the sum of each band over
    # the 10 m pixels that fall inside it, how many of those had data, and how many
    # there were in total. Means and coverage are computed at the very end.
    n_cells = size * size
    sums = np.zeros((TESSERA_BANDS, n_cells), dtype=np.float64)
    valid_counts = np.zeros(n_cells, dtype=np.int64)
    total_counts = np.zeros(n_cells, dtype=np.int64)

    # The crop's outline in longitude/latitude, to ask TESSERA for the right area.
    bounds_lonlat = transform_bounds(
        dst_crs,
        "EPSG:4326",
        *array_bounds(size, size, dst_transform),
        densify_pts=21,
    )
    west, south, east, north = bounds_lonlat
    inverse = ~dst_transform
    zones = utm_zones(west, east)

    # Read TESSERA one UTM zone at a time, in horizontal strips, so the full
    # 10 m crop (about 9600 x 9600 x 128 numbers) never has to fit in memory.
    for zone in zones:
        # Each zone serves only its own longitudes, so a crop on a zone seam
        # is read once per side without counting any pixel twice.
        zone_west, zone_east = zone * 6 - 186, zone * 6 - 180
        bbox = (max(west, zone_west), south, min(east, zone_east), north)
        to_dst = to_lonlat = None

        for block, transform, zone_crs in gt.iter_region(bbox, year, strip_rows=strip_rows):
            if to_dst is None:
                to_dst = Transformer.from_crs(zone_crs, dst_crs, always_xy=True)
                to_lonlat = Transformer.from_crs(zone_crs, "EPSG:4326", always_xy=True)

            # Map every 10 m pixel centre of this strip into the crop's grid and find
            # which 375 m crop cell it falls in; pixels outside the crop are dropped.
            rows, cols = block.shape[:2]
            col_centres, row_centres = np.meshgrid(
                np.arange(cols) + 0.5,
                np.arange(rows) + 0.5,
            )
            xs = transform.c + transform.a * col_centres + transform.b * row_centres
            ys = transform.f + transform.d * col_centres + transform.e * row_centres

            px, py = to_dst.transform(xs, ys)
            dst_col = np.floor(inverse.a * px + inverse.b * py + inverse.c)
            dst_row = np.floor(inverse.d * px + inverse.e * py + inverse.f)
            inside = (
                (dst_col >= 0) & (dst_col < size)
                & (dst_row >= 0) & (dst_row < size)
            )
            if len(zones) > 1:
                lon, _ = to_lonlat.transform(xs, ys)
                inside &= (lon >= zone_west) & (lon < zone_east)
            if not inside.any():
                continue

            # Add this strip's pixels to their cells: bincount sums values per cell index
            # in one vectorized pass. Pixels with any missing band count toward the total
            # but not toward the mean.
            cell = (dst_row[inside] * size + dst_col[inside]).astype(np.int64)
            values = block[inside]
            has_data = np.isfinite(values).all(axis=1)
            total_counts += np.bincount(cell, minlength=n_cells)

            cell = cell[has_data]
            values = values[has_data]
            valid_counts += np.bincount(cell, minlength=n_cells)
            for band in range(TESSERA_BANDS):
                sums[band] += np.bincount(
                    cell,
                    weights=values[:, band],
                    minlength=n_cells,
                )

            del block, values, xs, ys, px, py
            gc.collect()

    # Turn the totals into per-cell means and a coverage fraction. Cells with no
    # valid 10 m pixel stay NaN here (stored as 0 with coverage 0).
    embedding = np.full((TESSERA_BANDS, n_cells), np.nan, dtype=np.float32)
    has_cell_data = valid_counts > 0
    embedding[:, has_cell_data] = sums[:, has_cell_data] / valid_counts[has_cell_data]
    coverage = np.divide(
        valid_counts,
        total_counts,
        out=np.zeros(n_cells),
        where=total_counts > 0,
    )

    return (
        embedding.reshape(TESSERA_BANDS, size, size),
        coverage.reshape(size, size).astype(np.float32),
        total_counts.reshape(size, size),
        bounds_lonlat,
    )


def save_tessera_npz(save_path, embedding, coverage, metadata):
    """Write one fire's embedding, coverage, and metadata, then verify it."""
    if embedding.shape != (TESSERA_BANDS, CROP_SIZE, CROP_SIZE):
        raise ValueError(f"Unexpected embedding shape {embedding.shape}")
    if coverage.shape != (CROP_SIZE, CROP_SIZE):
        raise ValueError(f"Unexpected coverage shape {coverage.shape}")

    # float16 halves the file size; TESSERA values are small enough not to overflow.
    stored = np.nan_to_num(embedding, nan=0.0).astype(np.float16)
    if not np.isfinite(stored).all():
        raise ValueError(f"{save_path}: embedding overflows float16")

    np.savez_compressed(
        save_path,
        format_version=np.asarray(TESSERA_FORMAT_VERSION, dtype=np.int16),
        embedding=stored,
        coverage=coverage.astype(np.float16),
        **{key: np.asarray(value) for key, value in metadata.items()},
    )

    # Read the file back once to make sure it is complete.
    with np.load(save_path, allow_pickle=False) as archive:
        if archive["embedding"].shape != (TESSERA_BANDS, CROP_SIZE, CROP_SIZE):
            raise RuntimeError(f"Failed to verify {save_path}")


def build_fire_tessera(gt, fire_id, day_paths, fire_year, save_path, strip_rows=128):
    """Build tessera/p<fire_id>.npz from local VIIRS_Day GeoTIFFs.

    `day_paths` maps source repo names to local paths. Every given file must
    share one georeference, otherwise the fixed crop would cover different
    ground on different days and no single embedding could match it.
    """
    # All days of a fire must share one georeference, otherwise the fixed crop would
    # cover different ground on different days and one embedding could not match them.
    profiles = {}
    for remote_name, local_path in day_paths.items():
        with rasterio.open(local_path) as reader:
            profiles[remote_name] = reader.profile

    reference_name, reference = next(iter(profiles.items()))
    for remote_name, profile in profiles.items():
        if profile["crs"] != reference["crs"] or not profile["transform"].almost_equals(
            reference["transform"]
        ):
            raise ValueError(
                f"{fire_id}: georeference of {remote_name} differs from "
                f"{reference_name}"
            )

    # Use the year BEFORE the fire: an embedding of the fire year itself is built
    # partly from post-fire images and would leak the burn scar into the inputs.
    dst_crs, dst_transform = crop_grid(reference)
    tessera_year = int(fire_year) - 1
    embedding, coverage, _, bounds_lonlat = grid_tessera(
        gt,
        dst_crs,
        dst_transform,
        tessera_year,
        strip_rows=strip_rows,
    )

    # Provenance stored with the embedding: where and when it came from, and which code built it.
    metadata = {
        "fire_id": str(fire_id),
        "fire_year": np.int16(fire_year),
        "tessera_year": np.int16(tessera_year),
        "crs": dst_crs.to_wkt(),
        "transform": np.asarray(tuple(dst_transform)[:6], dtype=np.float64),
        "bounds": np.asarray(array_bounds(CROP_SIZE, CROP_SIZE, dst_transform), dtype=np.float64),
        "bounds_lonlat": np.asarray(bounds_lonlat, dtype=np.float64),
        "source_store": str(gt.url),
        "model_version": str(getattr(gt, "model_version", "")),
        "geotessera_version": package_version("geotessera"),
        "code_version": code_version(),
        "day_files": np.asarray(sorted(day_paths), dtype=str),
    }
    save_tessera_npz(save_path, embedding, coverage, metadata)

    print(
        f"{fire_id}: TESSERA {tessera_year}, mean coverage "
        f"{coverage.mean():.3f}, cells without data "
        f"{int(np.count_nonzero(coverage == 0))}/{coverage.size}"
    )
    return coverage
