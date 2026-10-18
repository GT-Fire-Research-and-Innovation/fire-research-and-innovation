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


# ----- Writing ----- #


def save_tessera_npz(save_path, embedding, coverage, metadata):
    """Write one fire's TESSERA file and check that it reads back.

    The embedding is stored as float16 (missing cells as 0) to halve the file
    size, together with the coverage map and provenance metadata. The file is
    read back once to make sure it is complete.

    Args:
        save_path (str | Path): where to write tessera/p<id>.npz.
        embedding (np.ndarray): (128, 256, 256) float32, NaN where there is no data.
        coverage (np.ndarray): (256, 256), the fraction of 10 m pixels with data.
        metadata (dict): extra members (fire id, years, grid, versions).

    Returns:
        Path: save_path.

    Raises:
        ValueError: on a wrong shape or a value too large for float16.
        RuntimeError: if the written file does not read back correctly.
    """
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
    return Path(save_path)


# ----- Reading ----- #


def read_tessera(path):
    """Read one fire's TESSERA embedding and coverage.

    The arrays are marked read-only, because one fire's map is shared by all of
    its windows (and by DataLoader worker processes). Nothing may change it.

    Args:
        path (str | Path): a tessera/p<id>.npz file.

    Returns:
        tuple[np.ndarray, np.ndarray]: (embedding (128, 256, 256) float16,
        coverage (256, 256) float16).

    Raises:
        ValueError: if the embedding has the wrong shape.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        embedding = archive["embedding"]
        coverage = archive["coverage"]
    if embedding.shape != (TESSERA_BANDS, CROP_SIZE, CROP_SIZE):
        raise ValueError(
            f"{path}: embedding has shape {embedding.shape}; "
            f"expected ({TESSERA_BANDS}, {CROP_SIZE}, {CROP_SIZE})"
        )
    embedding.flags.writeable = False
    coverage.flags.writeable = False
    return embedding, coverage


def load_tessera_maps(tessera_dir, fire_sources):
    """Read the TESSERA file of every fire in a list, once each.

    Pass the result both to compute_tessera_stats and to FireDataset, so neither
    reads the files again (about 16 MiB per fire). Every fire must have a file.

    Args:
        tessera_dir (str | Path): the folder of tessera/p<id>.npz files.
        fire_sources (list[Path]): the fires' archives or folders.

    Returns:
        dict[str, tuple[np.ndarray, np.ndarray]]: {"p<id>": (embedding, coverage)}.

    Raises:
        RuntimeError: if any fire has no TESSERA file.
    """
    tessera_dir = Path(tessera_dir)
    names = [fire_name(path) for path in fire_sources]
    missing = [f"{name}.npz" for name in names if not (tessera_dir / f"{name}.npz").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} fires have no TESSERA file in {tessera_dir}: {missing[:10]}")
    return {name: read_tessera(tessera_dir / f"{name}.npz") for name in names}


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
        embedding = embedding.astype(np.float64)
        valid = coverage.astype(np.float32) >= min_coverage
        values = embedding[:, valid]
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
