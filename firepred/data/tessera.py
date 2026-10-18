"""Reading per-fire TESSERA embeddings and their normalization statistics.

Level 1 (data). Used by data/dataset.py and training/run.py. The files are
built by prep/tessera_grid.py: tessera/p<id>.npz with a (128, 256, 256)
float16 `embedding` and a (256, 256) `coverage` (fraction of 10 m pixels
with data). [ours]
"""

from pathlib import Path

import numpy as np

from firepred.config.channels import TESSERA_BANDS
from firepred.config.dataset import CROP_SIZE

# Stored (embedding, coverage) arrays by resolved TESSERA path, so each file
# is unzipped once per process (about 16 MiB per fire as float16) and shared
# by compute_tessera_stats and every FireDataset.
_TESSERA_CACHE = {}


def read_tessera(path):
    """Return one TESSERA file's stored (embedding, coverage), read once."""
    path = Path(path).resolve()
    if path not in _TESSERA_CACHE:
        with np.load(path, allow_pickle=False) as archive:
            embedding = archive["embedding"]
            coverage = archive["coverage"]
        if embedding.shape != (TESSERA_BANDS, CROP_SIZE, CROP_SIZE):
            raise ValueError(
                f"{path}: embedding has shape {embedding.shape}; "
                f"expected ({TESSERA_BANDS}, {CROP_SIZE}, {CROP_SIZE})"
            )
        # Shared by every dataset and DataLoader worker: never modify them.
        embedding.flags.writeable = False
        coverage.flags.writeable = False
        _TESSERA_CACHE[path] = (embedding, coverage)
    return _TESSERA_CACHE[path]


def compute_tessera_stats(tessera_paths, min_coverage=0.5):
    """Per-band mean/std over well-covered pixels of the given TESSERA files.

    Compute this from the training fires only and reuse it for validation and
    test, so no statistics leak from held-out fires.
    """
    # Accumulate sum and sum of squares over all well-covered pixels of the
    # training fires (float64 to avoid rounding drift), then mean and std per band.
    total = np.zeros(TESSERA_BANDS, dtype=np.float64)
    total_sq = np.zeros(TESSERA_BANDS, dtype=np.float64)
    count = 0
    for path in tessera_paths:
        embedding, coverage = read_tessera(path)
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
