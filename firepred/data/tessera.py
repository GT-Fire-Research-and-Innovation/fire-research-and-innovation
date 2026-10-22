"""Building TESSERA for the dataset: per-fire grids, which fires get one, training views.

Level 1 (data, dataset building). Used by data/tessera_master.py. [ours]
rasterio and pyproj are imported inside the functions that need them, so the
masters import without the geospatial stack. The file format itself is in
data/archive.py.

    grid        build_fire_tessera: a fire's VIIRS grid -> averaged TESSERA -> tessera/p<id>.npz
    rules       has_prior_year_tessera, tessera_excluded_npz, missing_tessera
    views       build_tessera_view: a symlinked training root without the excluded fires

data/windows.py crops rows/cols 128:384 of every VIIRS_Day GeoTIFF, so one
256 x 256 grid covers all of a fire's windows. build_fire_tessera() reads that
grid's georeference, streams the TESSERA embeddings covering it, and averages
every 10 m TESSERA pixel into the crop pixel containing its centre. With
cells_per_pixel=k each VIIRS pixel is split into k x k cells first (k = 5 gives
75 m cells and a 1280 x 1280 map); the default 1 is one vector per VIIRS pixel.

Embeddings come from the year before the fire: an annual embedding for the
fire year is built partly from post-fire imagery and would leak the burn scar
into the inputs. Fires whose prior year is not published (2017 fires) get none,
and are left out of TESSERA runs and their matched baselines.
"""

import gc
import importlib.metadata
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from firepred.config.dataset import (
    CROP_OFFSET,
    CROP_SIZE,
    FIRST_TESSERA_YEAR,
    SPLITS,
    STRIP_ROWS,
    TESSERA_BANDS,
    TESSERA_CELLS_PER_PIXEL,
    TESSERA_EXCLUDED_LIST,
    WINDOW_CONFIG_FILE,
)
from firepred.data.archive import save_tessera_npz
from firepred.data.files import code_version


@dataclass(frozen=True)
class FireTessera:
    """Describe one TESSERA file that was just built.

    The master prints its summary and uploads the file. The coverage map shows
    how much of the fire's square the embeddings covered.

    Attributes:
        path (Path): the written tessera/p<id>.npz.
        tessera_year (int): the embedding year (the year before the fire).
        coverage (np.ndarray): (256 k, 256 k) fraction of 10 m pixels with data.
    """

    path: Path
    tessera_year: int
    coverage: np.ndarray

    def summary(self):
        """Summarize the build in one line."""
        empty = int(np.count_nonzero(self.coverage == 0))
        return (f"TESSERA {self.tessera_year}, mean coverage {self.coverage.mean():.3f}, "
                f"cells without data {empty}/{self.coverage.size}")


def crop_grid(profile, offset=CROP_OFFSET, size=CROP_SIZE, cells_per_pixel=TESSERA_CELLS_PER_PIXEL):
    """Find the georeference of the fire's 256 x 256 crop of a GeoTIFF, optionally subdivided.

    data/windows.py crops rows/columns 128:384 of every image. Shifting the
    GeoTIFF's geotransform by that offset maps crop pixels to map coordinates,
    so TESSERA can be averaged onto exactly the same square. With k cells per
    pixel the transform is also shrunk by k, so the same square holds
    256 k x 256 k cells and VIIRS pixel (r, c) covers cells k r .. k r + k - 1.

    Args:
        profile (dict): a rasterio profile (crs, transform, height, width).
        offset (int): first row/column of the crop.
        size (int): crop size in pixels.
        cells_per_pixel (int): TESSERA cells along each side of one VIIRS pixel.

    Returns:
        tuple: (crs, affine transform) of the crop grid.

    Raises:
        ValueError: if the crop does not fit inside the image, or cells_per_pixel < 1.
    """
    from rasterio.transform import Affine

    if int(cells_per_pixel) != cells_per_pixel or cells_per_pixel < 1:
        raise ValueError(f"cells_per_pixel must be a positive whole number, got {cells_per_pixel}")

    if profile["height"] < offset + size or profile["width"] < offset + size:
        raise ValueError(
            f"GeoTIFF is {profile['height']}x{profile['width']}; the "
            f"{size}x{size} crop at offset {offset} does not fit"
        )
    # Shift the GeoTIFF's geotransform by the crop offset: the result maps crop
    # pixel (row, col) to map coordinates, the same square data/windows.py uses.
    transform = profile["transform"] * Affine.translation(offset, offset)
    if cells_per_pixel > 1:
        transform = transform * Affine.scale(1 / cells_per_pixel)
    return profile["crs"], transform


def utm_zones(west, east):
    """List the UTM zones (1..60, west to east) a longitude range crosses; the antimeridian is not handled."""
    # UTM zones are 6 degrees of longitude wide, numbered 1..60 from 180 W.
    # TESSERA tiles are stored per UTM zone, so a crop may need more than one zone.
    first = int(np.floor((west + 180) / 6)) + 1
    last = int(np.floor((np.nextafter(east, west) + 180) / 6)) + 1
    return list(range(max(1, first), min(60, last) + 1))


def grid_tessera(gt, dst_crs, dst_transform, year, size=CROP_SIZE, strip_rows=STRIP_ROWS):
    """Average one year of TESSERA onto the fire's size x size grid.

    Every 10 m TESSERA pixel is assigned to the 375 m crop cell that contains
    its centre, and each cell gets the mean of its pixels. The region is read
    in row strips, so the full-resolution crop (about 9600 x 9600 x 128 values)
    never sits in memory at once. Pixels without data (water, gaps) are left
    out of the mean.

    Args:
        gt (geotessera.GeoTesseraZarr): the TESSERA store client.
        dst_crs (rasterio.crs.CRS): the crop's coordinate system.
        dst_transform (affine.Affine): the crop's geotransform.
        year (int): the embedding year.
        size (int): grid size in cells.
        strip_rows (int): rows read per request.

    Returns:
        tuple: (embedding (128, size, size) float32 with NaN where empty,
        coverage (size, size) float32, source_pixels (size, size) int64,
        bounds_lonlat (west, south, east, north)).
    """
    from pyproj import Transformer
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds

    # Running totals per crop cell (flattened 256 x 256): the sum of each band over
    # the 10 m pixels that fall inside it, how many of those had data, and how many
    # there were in total. Means and coverage are computed at the very end.
    n_cells = size * size
    sums = np.zeros((TESSERA_BANDS, n_cells), dtype=np.float64)
    valid_counts = np.zeros(n_cells, dtype=np.int64)
    total_counts = np.zeros(n_cells, dtype=np.int64)

    # The crop's outline in longitude/latitude, to ask TESSERA for the right area.
    bounds_lonlat = transform_bounds(dst_crs, "EPSG:4326", *array_bounds(size, size, dst_transform), densify_pts=21)
    west, south, east, north = bounds_lonlat
    inverse = ~dst_transform
    zones = utm_zones(west, east)

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
            col_centres, row_centres = np.meshgrid(np.arange(cols) + 0.5, np.arange(rows) + 0.5)
            xs = transform.c + transform.a * col_centres + transform.b * row_centres
            ys = transform.f + transform.d * col_centres + transform.e * row_centres

            px, py = to_dst.transform(xs, ys)
            dst_col = np.floor(inverse.a * px + inverse.b * py + inverse.c)
            dst_row = np.floor(inverse.d * px + inverse.e * py + inverse.f)
            inside = (dst_col >= 0) & (dst_col < size) & (dst_row >= 0) & (dst_row < size)
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
                sums[band] += np.bincount(cell, weights=values[:, band], minlength=n_cells)

            del block, values, xs, ys, px, py
            gc.collect()

    # Turn the totals into per-cell means and a coverage fraction. Cells with no
    # valid 10 m pixel stay NaN here (stored as 0 with coverage 0).
    embedding = np.full((TESSERA_BANDS, n_cells), np.nan, dtype=np.float32)
    has_cell_data = valid_counts > 0
    embedding[:, has_cell_data] = sums[:, has_cell_data] / valid_counts[has_cell_data]
    coverage = np.divide(valid_counts, total_counts, out=np.zeros(n_cells), where=total_counts > 0)

    return (
        embedding.reshape(TESSERA_BANDS, size, size),
        coverage.reshape(size, size).astype(np.float32),
        total_counts.reshape(size, size),
        bounds_lonlat,
    )


def build_fire_tessera(gt, fire_id, day_paths, fire_year, save_path, strip_rows=STRIP_ROWS,
                       cells_per_pixel=TESSERA_CELLS_PER_PIXEL):
    """Build and save one fire's TESSERA file from two of its VIIRS_Day GeoTIFFs.

    All given days must share one georeference, otherwise the fixed crop would
    cover different ground on different days. The embeddings come from the
    year before the fire, so the burn scar cannot leak into the inputs. The
    file also records where and how it was made.

    Args:
        gt (geotessera.GeoTesseraZarr): the TESSERA store client.
        fire_id (str): the fire id.
        day_paths (dict[str, str]): {source repo path: local path} of the day GeoTIFFs.
        fire_year (int): the year the fire started.
        save_path (str | Path): where to write tessera/p<id>.npz.
        strip_rows (int): rows read per request.
        cells_per_pixel (int): TESSERA cells along each side of one VIIRS pixel
            (1: one averaged vector per pixel; 5: 5 x 5 cells of 75 m).

    Returns:
        FireTessera: the written path, the year used and the coverage map.

    Raises:
        ValueError: if the GeoTIFFs do not share one georeference.
    """
    import rasterio
    from rasterio.transform import array_bounds

    profiles = {}
    for remote_name, local_path in day_paths.items():
        with rasterio.open(local_path) as reader:
            profiles[remote_name] = reader.profile

    reference_name, reference = next(iter(profiles.items()))
    for remote_name, profile in profiles.items():
        if profile["crs"] != reference["crs"] or not profile["transform"].almost_equals(reference["transform"]):
            raise ValueError(f"{fire_id}: georeference of {remote_name} differs from {reference_name}")

    # Use the year BEFORE the fire: an embedding of the fire year itself is built
    # partly from post-fire images and would leak the burn scar into the inputs.
    dst_crs, dst_transform = crop_grid(reference, cells_per_pixel=cells_per_pixel)
    tessera_year = int(fire_year) - 1
    grid_size = CROP_SIZE * cells_per_pixel
    embedding, coverage, _, bounds_lonlat = grid_tessera(gt, dst_crs, dst_transform, tessera_year, size=grid_size,
                                                         strip_rows=strip_rows)

    # Provenance stored with the embedding: where and when it came from, and which code built it.
    metadata = {
        "fire_id": str(fire_id),
        "fire_year": np.int16(fire_year),
        "tessera_year": np.int16(tessera_year),
        "crs": dst_crs.to_wkt(),
        "transform": np.asarray(tuple(dst_transform)[:6], dtype=np.float64),
        "bounds": np.asarray(array_bounds(grid_size, grid_size, dst_transform), dtype=np.float64),
        "bounds_lonlat": np.asarray(bounds_lonlat, dtype=np.float64),
        "source_store": str(gt.url),
        "model_version": str(getattr(gt, "model_version", "")),
        "geotessera_version": importlib.metadata.version("geotessera"),
        "code_version": code_version(),
        "day_files": np.asarray(sorted(day_paths), dtype=str),
    }
    if cells_per_pixel > 1:
        # Files without this member have one cell per VIIRS pixel (all files built before the option).
        metadata["cells_per_pixel"] = np.int16(cells_per_pixel)
    path = save_tessera_npz(save_path, embedding, coverage, metadata)
    return FireTessera(path, tessera_year, coverage)


# ----- Which fires get TESSERA ----- #


def has_prior_year_tessera(fire_year, first_year=FIRST_TESSERA_YEAR):
    """Say whether TESSERA exists for the year before a fire (published from 2017, so 2017 fires have none)."""
    return fire_year is not None and int(fire_year) - 1 >= first_year


def tessera_excluded_npz(roi):
    """List the "split/p<id>.npz" archives TESSERA runs and matched baselines leave out (no prior-year TESSERA)."""
    return sorted(
        f"{split}/p{fire_id}.npz"
        for split, fire_ids in roi.ids_by_split().items()
        for fire_id in fire_ids
        if not has_prior_year_tessera(roi.start_years.get(str(fire_id)))
    )


def missing_tessera(roi, present_names, excluded):
    """List kept (not excluded) fires that still have no TESSERA file, as "split/p<id>.npz"."""
    return sorted(
        f"{split}/p{fire_id}.npz"
        for split, fire_ids in roi.ids_by_split().items()
        for fire_id in fire_ids
        if f"{split}/p{fire_id}.npz" not in excluded and f"p{fire_id}.npz" not in present_names
    )


# ----- Training roots without the excluded fires ----- #


@dataclass(frozen=True)
class TesseraView:
    """Describe a training folder that leaves out fires without TESSERA.

    The view is made of symbolic links to the real archives, so the extracted
    data is never changed. The same extraction can serve TESSERA and
    no-TESSERA runs.

    Attributes:
        root (Path): the view folder (use it as the training data root).
        dropped (tuple[str, ...]): "split/p<id>.npz" archives left out.
    """

    root: Path
    dropped: tuple

    def summary(self):
        """Summarize the view in one line."""
        return f"TESSERA view {self.root}: dropped {len(self.dropped)} archives without prior-year TESSERA"


def read_excluded_list(tessera_dir):
    """Read excluded_npz.txt, shipped in tessera.zip: the window archives every TESSERA run leaves out."""
    path = Path(tessera_dir) / TESSERA_EXCLUDED_LIST
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_excluded_list(tessera_dir, excluded):
    """Write excluded_npz.txt (one archive name per line) for tessera.zip."""
    path = Path(tessera_dir) / TESSERA_EXCLUDED_LIST
    path.write_text("\n".join(excluded) + "\n")
    return path


def drop_excluded_archives(data_root, excluded):
    """Delete the listed window archives from a data root.

    This is the one place window data is deleted, and only when asked (pulling
    tessera.zip with drop_excluded). Names that do not exist are ignored.

    Args:
        data_root (str | Path): the data root.
        excluded (list[str]): "split/p<id>.npz" names to delete.

    Returns:
        list[str]: the names actually deleted.
    """
    removed = []
    for name in excluded:
        path = Path(data_root) / name
        if path.exists():
            path.unlink()
            removed.append(name)
    return removed


def clear_view(view_dir):
    """Remove an old view folder before building a new one, but never real data.

    A view holds only symbolic links. If the folder holds any real file, it is
    not a view and is left untouched with an error.

    Args:
        view_dir (str | Path): the view folder.

    Returns:
        Path: view_dir.

    Raises:
        RuntimeError: if the folder holds real files.
    """
    view_dir = Path(view_dir)
    if not view_dir.exists():
        return view_dir
    real_files = [p for p in view_dir.rglob("*") if p.is_file() and not p.is_symlink()
                  and not any(parent.is_symlink() for parent in p.parents)]
    if real_files:
        raise RuntimeError(f"{view_dir} holds real files (e.g. {real_files[0]}); refusing to clear it")
    shutil.rmtree(view_dir)
    return view_dir


def build_tessera_view(data_root, tessera_dir, view_dir):
    """Build a training folder of links to the kept archives plus tessera/ and window_config.json.

    Fires listed in excluded_npz.txt are left out, and the extracted data
    itself is never modified. Every kept archive must have its TESSERA file, so
    a missing file fails here, before training.

    Args:
        data_root (str | Path): the extracted data root.
        tessera_dir (str | Path): the folder of TESSERA files and excluded_npz.txt.
        view_dir (str | Path): where to build the view.

    Returns:
        TesseraView: the view folder and the archives left out.

    Raises:
        RuntimeError: if a kept archive has no TESSERA file, or view_dir holds real files.
    """
    data_root, tessera_dir, view_dir = Path(data_root), Path(tessera_dir), Path(view_dir)
    excluded = set(read_excluded_list(tessera_dir))
    clear_view(view_dir)
    dropped = []
    # Symlink every kept archive into view_dir/<split>/; excluded ones are only counted.
    for split in SPLITS:
        (view_dir / split).mkdir(parents=True)
        for archive in sorted((data_root / split).glob("*.npz")):
            if f"{split}/{archive.name}" in excluded:
                dropped.append(f"{split}/{archive.name}")
            else:
                (view_dir / split / archive.name).symlink_to(archive.resolve())
    (view_dir / "tessera").symlink_to(tessera_dir.resolve(), target_is_directory=True)
    # Training reads the data root's window_config.json first; the view links it too.
    if (data_root / WINDOW_CONFIG_FILE).is_file():
        (view_dir / WINDOW_CONFIG_FILE).symlink_to((data_root / WINDOW_CONFIG_FILE).resolve())
    # Every kept archive needs its TESSERA file; fail before training if any is missing.
    missing = [f"{split}/{archive.name}" for split in SPLITS
               for archive in (view_dir / split).glob("*.npz")
               if not (tessera_dir / archive.name).exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} kept archives have no TESSERA file: {missing[:10]}")
    return TesseraView(view_dir, tuple(dropped))
