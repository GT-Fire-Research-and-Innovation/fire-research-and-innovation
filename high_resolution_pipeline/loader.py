"""PyTorch reader for the high-resolution patch archive format.

Each dataset item is one temporal window from one spatial patch. The reader
decompresses only that window and its patch's static TESSERA map. TESSERA stays
one (129, H, W) tensor in host memory; a training loop can repeat it across
time on the accelerator, as the standard prediction trainer now does.

Input imagery is returned as 27 raw channels. Callers must apply the desired
VIIRS normalization, angle conversion, and land-cover encoding before model
input. Labels are returned as two class channels and `valid_mask` identifies
padding or other pixels to exclude from loss calculations.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from high_resolution_pipeline.package import patch_archive_window_count


def patch_files_for_split(split_dir: str | Path) -> list[Path]:
    """Find patch archives beneath `<split>/p<fire_id>/patches/`."""
    split_dir = Path(split_dir)
    return sorted(split_dir.glob("p*/patches/*.npz"))


def compute_tessera_stats(patch_files, min_coverage: float = 0.5) -> dict:
    """Compute TESSERA band statistics from training patch files only.

    A band-at-a-time pass bounds temporary memory. If patches overlap, their
    repeated cells are counted repeatedly, matching how those cells appear in
    the training sample stream.
    """
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be between 0 and 1")
    sums = np.zeros(128, dtype=np.float64)
    sums_squared = np.zeros(128, dtype=np.float64)
    count = 0
    patch_files = list(patch_files)
    if not patch_files:
        raise ValueError("No patch archives supplied for TESSERA statistics")

    for path in patch_files:
        with np.load(path, allow_pickle=False) as archive:
            embedding = archive["embedding"]
            coverage = archive["coverage"].astype(np.float32)
            valid_mask = archive["valid_mask"].astype(bool)
            if embedding.ndim != 3 or embedding.shape[0] != 128:
                raise ValueError(f"{path}: expected TESSERA shape (128,H,W), got {embedding.shape}")
            valid = (coverage >= min_coverage) & valid_mask
            pixel_count = int(valid.sum())
            if not pixel_count:
                continue
            for band in range(128):
                values = embedding[band][valid].astype(np.float64, copy=False)
                sums[band] += values.sum(dtype=np.float64)
                sums_squared[band] += np.square(values).sum(dtype=np.float64)
            count += pixel_count

    if count == 0:
        raise RuntimeError("No TESSERA pixels meet the requested coverage threshold")
    mean = sums / count
    std = np.sqrt(np.maximum(sums_squared / count - mean**2, 1e-12))
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "min_coverage": float(min_coverage),
    }


class HighResolutionPatchDataset(Dataset):
    """Expose each patch/window pair as a PyTorch dataset item.

    `split_dir` is the directory containing fire folders, such as
    `data_high_resolution/<settings>/train`. Supply `tessera_stats` computed
    from training patches to include normalized static TESSERA channels.
    """

    def __init__(self, split_dir, time_series_length: int, tessera_stats=None):
        if time_series_length <= 0:
            raise ValueError("time_series_length must be positive")
        self.split_dir = Path(split_dir)
        self.time_series_length = int(time_series_length)
        self.tessera_stats = tessera_stats
        self.patch_files = patch_files_for_split(self.split_dir)
        if not self.patch_files:
            raise RuntimeError(f"No patch archives found under {self.split_dir}")

        if tessera_stats is not None:
            mean = np.asarray(tessera_stats.get("mean"))
            std = np.asarray(tessera_stats.get("std"))
            threshold = tessera_stats.get("min_coverage")
            if mean.shape != (128,) or std.shape != (128,):
                raise ValueError("TESSERA stats must contain 128-element mean and std arrays")
            if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
                raise ValueError("TESSERA means/stds must be finite and stds must be positive")
            if threshold is None or not 0 <= float(threshold) <= 1:
                raise ValueError("TESSERA stats must include min_coverage between 0 and 1")

        self.index = []
        for path in self.patch_files:
            count = patch_archive_window_count(path)
            for window_index in range(count):
                self.index.append((path, window_index))
        if not self.index:
            raise RuntimeError(f"Patch archives under {self.split_dir} contain no windows")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        path, window_index = self.index[index]
        suffix = f"{window_index:05d}"
        with np.load(path, allow_pickle=False) as archive:
            data = archive[f"data_{suffix}"].astype(np.float32)
            label = archive[f"label_{suffix}"].astype(bool)
            valid_mask = archive["valid_mask"].astype(bool)
            if self.tessera_stats is not None:
                embedding = archive["embedding"].astype(np.float32)
                coverage = archive["coverage"].astype(np.float32)

        expected = (27, self.time_series_length, *label.shape)
        if data.shape != expected:
            raise ValueError(f"{path}: data has shape {data.shape}; expected {expected}")
        if label.ndim != 2 or valid_mask.shape != label.shape:
            raise ValueError(f"{path}: label and valid_mask must be matching 2-D arrays")

        label &= valid_mask
        target = torch.from_numpy(
            np.stack((~label, label), axis=0).astype(np.int64, copy=False)
        )
        sample = {
            "data": torch.from_numpy(data),
            "labels": target,
            "valid_mask": torch.from_numpy(valid_mask),
        }

        if self.tessera_stats is not None:
            if embedding.shape != (128, *label.shape) or coverage.shape != label.shape:
                raise ValueError(f"{path}: TESSERA, coverage, and labels have inconsistent shapes")
            threshold = float(self.tessera_stats["min_coverage"])
            valid_tessera = (coverage >= threshold) & valid_mask
            mean = np.asarray(self.tessera_stats["mean"], dtype=np.float32)[:, None, None]
            std = np.asarray(self.tessera_stats["std"], dtype=np.float32)[:, None, None]
            embedding = np.where(valid_tessera[None], (embedding - mean) / std, 0.0)
            static = np.concatenate((embedding, valid_tessera[None].astype(np.float32)), axis=0)
            sample["static"] = torch.from_numpy(static.astype(np.float32, copy=False))

        return sample
