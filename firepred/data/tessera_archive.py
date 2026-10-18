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
    """Write one fire's embedding, coverage, and metadata, verify it, and return the path."""
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
    """Read one TESSERA file and return its stored (embedding, coverage).

    The arrays are marked read-only: one fire's map is shared by every window
    of that fire (and by forked DataLoader workers), so nothing may modify it.
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
    """{fire name: (embedding, coverage)} for every fire archive in `fire_sources`.

    Each file is read once here (about 16 MiB per fire as float16); pass the
    result to compute_tessera_stats and to FireDataset(tessera_maps=...) so
    neither reads it again. Raises if any fire has no TESSERA file.
    """
    tessera_dir = Path(tessera_dir)
    names = [fire_name(path) for path in fire_sources]
    missing = [f"{name}.npz" for name in names if not (tessera_dir / f"{name}.npz").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} fires have no TESSERA file in {tessera_dir}: {missing[:10]}")
    return {name: read_tessera(tessera_dir / f"{name}.npz") for name in names}


def compute_tessera_stats(tessera_maps, min_coverage=0.5):
    """Per-band mean/std over well-covered pixels of the given (embedding, coverage) maps.

    Compute this from the training fires only and reuse it for validation and
    test, so no statistics leak from held-out fires. Returns
    {"mean": (128,), "std": (128,), "min_coverage": float}.
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
    """(129, H, W) float32 model input for one fire: the normalized embedding plus a validity mask.

    Pixels whose TESSERA coverage is below the training threshold (`valid`
    False) are zeroed (the normalized mean) and marked 0 in the mask channel.
    `stats` is compute_tessera_stats() of the training fires.
    """
    embedding = embedding.astype(np.float32)
    mean = stats["mean"][:, None, None]
    std = stats["std"][:, None, None]
    embedding = np.where(valid, (embedding - mean) / std, 0.0)
    return np.concatenate((embedding, valid[None].astype(np.float32)), axis=0).astype(np.float32)
