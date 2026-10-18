"""The TESSERA file format: one tessera/p<id>.npz per fire, written, read and turned into model input.

Level 1 (data). Written by data/tessera.py while building the dataset; read
by data/fire_dataset.py and training/ while training. [ours] This is the
TESSERA counterpart of data/archive.py (the window archive format). Each file
holds:

    embedding      (128, 256, 256) float16, 0 where coverage is 0
    coverage       (256, 256) float16, fraction of 10 m pixels with data
    format_version, plus provenance metadata (fire_id, fire_year, tessera_year,
    crs, transform, bounds, bounds_lonlat, source_store, model_version,
    geotessera_version, code_version, day_files)

numpy only, so it loads on any training machine.
"""

from pathlib import Path

import numpy as np

from firepred.config.dataset import CROP_SIZE, TESSERA_BANDS, TESSERA_FORMAT_VERSION
from firepred.data.archive import fire_name


# ----- Grid size ----- #


def cells_per_pixel_of(shape, expected=None):
    """Find how many TESSERA cells per VIIRS pixel side a stored map has.

    A map holds k x k cells for every one of the 256 x 256 VIIRS pixels, so its
    side is 256 k. k = 1 is one averaged vector per VIIRS pixel. A map whose
    side is not a whole multiple of 256 is refused.

    Args:
        shape (tuple[int, ...]): the embedding shape (128, S, S) or coverage shape (S, S).
        expected (int | None): the k the caller needs, or None for any.

    Returns:
        int: k.

    Raises:
        ValueError: if the shape is not square, not a multiple of 256, or not the expected k.
    """
    side = shape[-1]
    if len(shape) < 2 or shape[-2] != side or side % CROP_SIZE or side == 0:
        raise ValueError(f"TESSERA map shape {tuple(shape)} is not a square multiple of {CROP_SIZE}")
    if len(shape) == 3 and shape[0] != TESSERA_BANDS:
        raise ValueError(f"TESSERA map shape {tuple(shape)} does not have {TESSERA_BANDS} bands")
    cells = side // CROP_SIZE
    if expected is not None and cells != expected:
        raise ValueError(f"TESSERA map has {cells} cells per VIIRS pixel; this run needs {expected}")
    return cells


# ----- Writing ----- #


def save_tessera_npz(save_path, embedding, coverage, metadata):
    """Write one fire's TESSERA file and check that it reads back.

    The embedding is stored as float16 (missing cells as 0) to halve the file
    size, together with the coverage map and provenance metadata. The file is
    read back once to make sure it is complete.

    Args:
        save_path (str | Path): where to write tessera/p<id>.npz.
        embedding (np.ndarray): (128, 256 k, 256 k) float32, NaN where there is no data
            (k TESSERA cells per VIIRS pixel side; 1 by default).
        coverage (np.ndarray): (256 k, 256 k), the fraction of 10 m pixels with data.
        metadata (dict): extra members (fire id, years, grid, versions).

    Returns:
        Path: save_path.

    Raises:
        ValueError: on a wrong shape or a value too large for float16.
        RuntimeError: if the written file does not read back correctly.
    """
    cells = cells_per_pixel_of(embedding.shape)
    if coverage.shape != embedding.shape[1:]:
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
        if archive["embedding"].shape != (TESSERA_BANDS, CROP_SIZE * cells, CROP_SIZE * cells):
            raise RuntimeError(f"Failed to verify {save_path}")
    return Path(save_path)


# ----- Reading ----- #


def read_tessera(path, cells_per_pixel=None):
    """Read one fire's TESSERA embedding and coverage.

    The arrays are marked read-only, because one fire's map is shared by all of
    its windows (and by DataLoader worker processes). Nothing may change it.

    Args:
        path (str | Path): a tessera/p<id>.npz file.
        cells_per_pixel (int | None): the k the caller needs, or None for any.

    Returns:
        tuple[np.ndarray, np.ndarray]: (embedding (128, 256 k, 256 k) float16,
        coverage (256 k, 256 k) float16).

    Raises:
        ValueError: if the embedding has the wrong shape or another k.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        embedding = archive["embedding"]
        coverage = archive["coverage"]
    try:
        cells_per_pixel_of(embedding.shape, cells_per_pixel)
    except ValueError as error:
        raise ValueError(f"{path}: embedding has shape {embedding.shape}; {error}") from None
    embedding.flags.writeable = False
    coverage.flags.writeable = False
    return embedding, coverage


def tessera_paths(tessera_dir, fire_sources):
    """Find the TESSERA file of every fire in a list, and refuse if any is missing.

    A missing file fails here, before training, not at some later window.
    Nothing is read yet.

    Args:
        tessera_dir (str | Path): the folder of tessera/p<id>.npz files.
        fire_sources (list[Path]): the fires' archives or folders.

    Returns:
        dict[str, Path]: {"p<id>": its TESSERA file}, in fire order.

    Raises:
        RuntimeError: if any fire has no TESSERA file.
    """
    tessera_dir = Path(tessera_dir)
    names = [fire_name(path) for path in fire_sources]
    missing = [f"{name}.npz" for name in names if not (tessera_dir / f"{name}.npz").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} fires have no TESSERA file in {tessera_dir}: {missing[:10]}")
    return {name: tessera_dir / f"{name}.npz" for name in names}


def load_tessera_maps(tessera_dir, fire_sources, cells_per_pixel=None):
    """Read the TESSERA file of every fire in a list, once each.

    Pass the result both to compute_tessera_stats and to FireDataset, so neither
    reads the files again (about 16 MiB per fire at one cell per VIIRS pixel).
    Every fire must have a file.

    Args:
        tessera_dir (str | Path): the folder of tessera/p<id>.npz files.
        fire_sources (list[Path]): the fires' archives or folders.
        cells_per_pixel (int | None): the k every map must have, or None for any.

    Returns:
        dict[str, tuple[np.ndarray, np.ndarray]]: {"p<id>": (embedding, coverage)}.

    Raises:
        RuntimeError: if any fire has no TESSERA file.
    """
    return {name: read_tessera(path, cells_per_pixel)
            for name, path in tessera_paths(tessera_dir, fire_sources).items()}


def iter_tessera_maps(tessera_dir, fire_sources, cells_per_pixel=None):
    """Read the fires' TESSERA files one at a time, for statistics over large maps.

    With k x k cells per VIIRS pixel a map is k² times larger (about 400 MiB per
    fire at k = 5), so they cannot all sit in memory. Missing files are found
    at once; each map is read only when the iterator reaches it.

    Args:
        tessera_dir (str | Path): the folder of tessera/p<id>.npz files.
        fire_sources (list[Path]): the fires' archives or folders.
        cells_per_pixel (int | None): the k every map must have, or None for any.

    Returns:
        Iterator[tuple[np.ndarray, np.ndarray]]: (embedding, coverage) of one fire at a time.

    Raises:
        RuntimeError: if any fire has no TESSERA file.
    """
    paths = tessera_paths(tessera_dir, fire_sources)
    return (read_tessera(path, cells_per_pixel) for path in paths.values())


def compute_tessera_stats(tessera_maps, min_coverage=0.5):
    """Compute the per-band mean and spread of TESSERA over well-covered pixels.

    These statistics standardize the embeddings before the model sees them.
    Compute them from the training fires only and reuse them for validation
    and test, so nothing is learned from held-out fires. Pixels with less
    coverage than min_coverage are left out.

    Args:
        tessera_maps (Iterable[tuple[np.ndarray, np.ndarray]]): (embedding, coverage) pairs.
        min_coverage (float): the smallest coverage a pixel needs to count.

    Returns:
        dict: {"mean": (128,) float32, "std": (128,) float32, "min_coverage": float}.

    Raises:
        RuntimeError: if no pixel has enough coverage.
    """
    # Accumulate sum and sum of squares over all well-covered pixels of the
    # training fires (float64 to avoid rounding drift), then mean and std per band.
    total = np.zeros(TESSERA_BANDS, dtype=np.float64)
    total_sq = np.zeros(TESSERA_BANDS, dtype=np.float64)
    count = 0
    for embedding, coverage in tessera_maps:
        valid = coverage.astype(np.float32) >= min_coverage
        # Select first, then widen: only the kept pixels are copied to float64.
        values = embedding[:, valid].astype(np.float64)
        total += values.sum(axis=1)
        total_sq += (values ** 2).sum(axis=1)
        count += values.shape[1]
    if count == 0:
        raise RuntimeError("No TESSERA pixels with sufficient coverage")
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 1e-12))
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "min_coverage": float(min_coverage),
    }


def static_map(embedding, valid, stats):
    """Build one fire's TESSERA model input: the normalized embedding plus a validity mask.

    Each band is standardized with the training statistics. Pixels with too
    little coverage are set to 0 (the normalized mean) and marked 0 in an extra
    mask channel, so the model can tell real values from filled ones.

    Args:
        embedding (np.ndarray): (128, H, W) stored embedding.
        valid (np.ndarray): (H, W) bool, True where coverage is high enough.
        stats (dict): compute_tessera_stats() of the training fires.

    Returns:
        np.ndarray: (129, H, W) float32.
    """
    embedding = embedding.astype(np.float32)
    mean = stats["mean"][:, None, None]
    std = stats["std"][:, None, None]
    embedding = np.where(valid, (embedding - mean) / std, 0.0)
    return np.concatenate((embedding, valid[None].astype(np.float32)), axis=0).astype(np.float32)
