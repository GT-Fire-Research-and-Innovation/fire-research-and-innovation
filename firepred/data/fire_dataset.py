"""From window archives to model input: TESSERA maps, FireDataset, loaders, batches.

Level 1 (data). Used by training/ and cli/check.py. Three parts, in the order
data flows through them:

    TESSERA      read_tessera, load_tessera_maps, compute_tessera_stats:
                 per-fire embedding files and their normalization statistics
    FireDataset  every window of every fire in a split, as one Dataset;
                 dataset[i] reads one window from its fire archive
                 (data/archive.py) and runs data/transforms.py on it:
                     load       (27, T, 256, 256) float32 input, (256, 256) target
                     normalize  standardized channels
                     augment    (training only) flips/rotation
                     preprocess -> {"data": (43, T, 256, 256), "labels": (2, 256, 256)}
                 In TESSERA mode it also returns "static": (129, 256, 256),
                 the fire's normalized embedding plus a validity mask.
    loaders      make_dataloader, validate_dataset_output, and batch_inputs,
                 which repeats the static map across the T days on the GPU.

TESSERA files are built by data/tessera.py: tessera/p<id>.npz with a
(128, 256, 256) float16 `embedding` and a (256, 256) `coverage` (fraction of
10 m pixels with data). [ours]
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from firepred.config.dataset import (
    BASE_MODEL_CHANNELS,
    CROP_SIZE,
    DEGREE_CHANNELS,
    LANDCOVER_CHANNEL,
    N_STORED_CHANNELS,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    TESSERA_BANDS,
    TESSERA_CHANNELS,
)
from firepred.data.archive import find_fire_sources, fire_name, index_fires, open_fire
from firepred.data.tessera_archive import load_tessera_maps, static_map
from firepred.data.transforms import Normalize, augment, landcover_one_hot_matrix, preprocess


# ----- FireDataset ----- #


class FireDataset(Dataset):
    """Expose every temporal window across every fire as one Dataset item.

    npz_dir:              a split folder of p<id>.npz archives or p<id>/ folders
    ts_length:            input days T per window
    use_augmentations:    random flips/rotation (training split only)
    n_channel:            stored channels per window (27)
    target_is_single_day: labels are (H, W) next-day maps (prediction) rather
                          than (T, H, W) sequences
    tessera_dir:          folder of tessera/p<id>.npz files, or None
    tessera_stats:        training-split statistics from compute_tessera_stats()
    tessera_maps:         this split's maps from load_tessera_maps(), when the
                          caller already read them; otherwise read from tessera_dir
    """

    def __init__(
        self,
        npz_dir,
        ts_length,
        use_augmentations=False,
        n_channel=N_STORED_CHANNELS,
        target_is_single_day=False,
        tessera_dir=None,
        tessera_stats=None,
        tessera_maps=None,
    ):
        """Find the split's fires, load TESSERA maps if asked, and index every window."""
        # [ours] data and labels live together in version-2 archives, or in
        # unzipped p<id>/ folders of the same members.
        # 1. Find the split's fires (archives or unzipped folders).
        self.npz_dir = Path(npz_dir)
        self.npz_files = find_fire_sources(self.npz_dir)

        if not self.npz_files:
            raise RuntimeError(
                f"No .npz archives or unzipped fire folders found in {self.npz_dir}"
            )

        # [ours] optional TESSERA mode. Each fire archive p<id>.npz pairs with
        # <tessera_dir>/p<id>.npz; None keeps the 27-channel behavior.
        # 2. Optional TESSERA: load every fire's embedding map once, up front.
        self.tessera_dir = Path(tessera_dir) if tessera_dir is not None else None
        self.tessera_stats = tessera_stats
        # Each fire's TESSERA map is read here, once, rather than in every
        # __getitem__. DataLoader workers forked later (Linux, including
        # Kaggle) share these arrays instead of copying them.
        self.tessera_maps = {}
        if self.tessera_dir is not None:
            if tessera_stats is None:
                raise ValueError("tessera_stats is required with tessera_dir")
            if tessera_maps is None:
                tessera_maps = load_tessera_maps(self.tessera_dir, self.npz_files)
            for path in self.npz_files:
                name = fire_name(path)
                embedding, coverage = tessera_maps[name]
                valid = (
                    coverage.astype(np.float32) >= tessera_stats["min_coverage"]
                )
                self.tessera_maps[name] = (embedding, valid)

        # 3. Settings used when loading each window.
        self.n_channel = n_channel
        self.ts_length = ts_length
        self.target_is_single_day = target_is_single_day
        self.use_augmentations = use_augmentations

        # [released] angles in degrees, transformed with sine after augmentation.
        self.indices_of_degree_features = list(DEGREE_CHANNELS)

        # 4. [ours] The global index: one (fire, window number) entry per window
        # across all fires (data/archive.index_fires), so PyTorch can shuffle
        # windows across fires.
        self.index = index_fires(self.npz_files)
        if not self.index:
            raise RuntimeError(
                f"NPZ files in {self.npz_dir} contain no temporal windows"
            )

        # [released] the original 27-channel training statistics; angle and
        # land-cover channels are not normalized.
        # 5. Fixed preprocessing tools: per-channel standardization and the land-cover
        # one-hot lookup table.
        self.normalizer = Normalize(
            mean=NORMALIZATION_MEAN,
            std=NORMALIZATION_STD,
            dont_normalize_idc=self.indices_of_degree_features + [LANDCOVER_CHANNEL],
        )

        self.one_hot_matrix = landcover_one_hot_matrix()

    def windows_by_fire(self):
        """{fire id (no "p"): [dataset indices of its windows]}, in index order."""
        groups = {}
        for dataset_index, (archive_path, _) in enumerate(self.index):
            groups.setdefault(fire_name(archive_path).removeprefix("p"), []).append(dataset_index)
        return groups

    def __len__(self):
        """The number of windows across all fires."""
        return len(self.index)

    def __getitem__(self, idx):
        """Window `idx` as {"data", "labels"[, "static"]}, ready for the model."""
        # DataLoader supplies a global window index; load_data maps it to one
        # member pair inside one fire archive.
        # One training example, in pipeline order: read the window from disk,
        # standardize, (training only) flip/rotate, then encode angles and land cover.
        x, y = self.load_data(idx)

        # [released] normalize before augmentation.
        x = self.normalizer(x)

        # [ours] the fire's TESSERA map is static, so it is returned once as
        # (129, H, W). Repeating it across the T days here cost T times its
        # size in host RAM; the trainer repeats it on the GPU.
        static = self.load_tessera(idx) if self.tessera_dir is not None else None

        # [released] augment training examples but leave validation unchanged.
        if self.use_augmentations:
            x, y, static = augment(x, y, static, self.indices_of_degree_features)

        # [released] degrees -> sine, land cover -> one-hot.
        x = preprocess(x, self.one_hot_matrix, self.indices_of_degree_features)

        sample = {
            "data": x,
            "labels": y,
        }
        if static is not None:
            sample["static"] = static
        return sample

    def load_data(self, idx):
        """Load and decode exactly one temporal window.

        Opening the archive reads its ZIP index; accessing data_###/label_###
        decompresses only those two members into RAM.
        """
        # Find which fire and which window this index refers to.
        npz_path, window_idx = self.index[idx]
        suffix = f"{window_idx:03d}"
        data_key = f"data_{suffix}"
        label_key = f"label_{suffix}"

        with open_fire(npz_path) as archive:
            data_array = archive[data_key]
            label_array = archive[label_key]

        # No squeeze(): it could remove a real temporal dimension when T == 1.
        x = torch.from_numpy(data_array).to(dtype=torch.float32)
        label = torch.from_numpy(label_array)

        # Validate the stored input instead of letting a later model layer
        # fail with a less informative shape error.
        expected_x_shape = (
            self.n_channel,
            self.ts_length,
            CROP_SIZE,
            CROP_SIZE,
        )
        if tuple(x.shape) != expected_x_shape:
            raise ValueError(
                f"{npz_path}:{data_key} has shape {tuple(x.shape)}; "
                f"expected {expected_x_shape}"
            )

        # [released] target_is_single_day: one map, or a temporal sequence.
        # Prediction targets are one 256 x 256 map (the newly burned pixels of the next day).
        if self.target_is_single_day:
            # Accept a harmless leading singleton left by older exports.
            if label.ndim == 3 and label.shape[0] == 1:
                label = label[0]

            if label.ndim != 2:
                raise ValueError(
                    f"target_is_single_day=True requires (H, W), but "
                    f"{npz_path}:{label_key} has shape {tuple(label.shape)}"
                )
        else:
            if label.ndim != 3:
                raise ValueError(
                    f"target_is_single_day=False requires (T, H, W), but "
                    f"{npz_path}:{label_key} has shape {tuple(label.shape)}"
                )

            if label.shape[0] != self.ts_length:
                raise ValueError(
                    f"{npz_path}:{label_key} contains {label.shape[0]} "
                    f"target frames; expected {self.ts_length}"
                )

        if tuple(label.shape[-2:]) != (CROP_SIZE, CROP_SIZE):
            raise ValueError(
                f"{npz_path}:{label_key} has spatial shape "
                f"{tuple(label.shape[-2:])}; expected ({CROP_SIZE}, {CROP_SIZE})"
            )

        # [released] the binary target as two channels: 0 background, 1 fire.
        # stack infers either (2,H,W) or (2,T,H,W).
        # Two-channel target for the two output channels: background (0) and fire (1).
        y = torch.stack(
            (
                label == 0,
                label > 0,
            ),
            dim=0,
        ).long()

        return x, y

    def load_tessera(self, idx):
        """Return (129, H, W): this window's fire map, normalized, plus its validity mask (tessera_archive.static_map)."""
        npz_path, _ = self.index[idx]
        # The stored map was read in __init__; only normalization runs per window.
        embedding, valid = self.tessera_maps[fire_name(npz_path)]
        return torch.from_numpy(static_map(embedding, valid, self.tessera_stats))


# ----- Loaders and batches ----- #


def make_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory, generator=None):
    """Create a conservative loader that can be tuned after RAM is measured.

    `generator` (a torch.Generator) gives the loader its own random stream for
    shuffling and worker seeds; None draws from the global torch generator.
    """
    # Batches of windows; training shuffles, validation and test keep a fixed order.
    loader_options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if generator is not None:
        loader_options["generator"] = generator

    # Worker prefetching can hold multiple decompressed windows in RAM.
    # Keep one prefetched batch per worker rather than PyTorch's default two.
    if num_workers > 0:
        loader_options["prefetch_factor"] = 1
        loader_options["persistent_workers"] = True

    return DataLoader(**loader_options)


def validate_dataset_output(dataset, expected_model_channels, dataset_name):
    """Fail before model construction if the Dataset/model shapes disagree."""
    if len(dataset) == 0:
        raise RuntimeError(f"{dataset_name} dataset contains no windows")

    # Load one window and check its channel count matches what the model expects,
    # so a data/model mismatch fails here, before any training time is spent.
    sample = dataset[0]
    data_shape = tuple(sample["data"].shape)
    label_shape = tuple(sample["labels"].shape)
    static_channels = sample["static"].shape[0] if "static" in sample else 0

    if len(data_shape) != 4:
        raise ValueError(
            f"{dataset_name} sample must be (C,T,H,W), found {data_shape}"
        )

    actual_channels = data_shape[0] + static_channels
    if actual_channels != expected_model_channels:
        raise ValueError(
            f"{dataset_name} FireDataset produces {actual_channels} channels "
            f"after preprocessing, but the model is configured for "
            f"{expected_model_channels}. The 27-channel archive plus 17-class "
            f"land-cover expansion gives {BASE_MODEL_CHANNELS} channels; "
            f"--tessera adds {TESSERA_CHANNELS} more."
        )

    print(
        f"{dataset_name} windows={len(dataset)}, "
        f"data shape={data_shape}, label shape={label_shape}"
        + (f", static TESSERA channels={static_channels}" if static_channels else "")
    )


def batch_inputs(batch, device):
    """Model input on `device`: the 43 stored channels, then (TESSERA mode)
    the static (129, H, W) map repeated across the T days.

    The repeat happens here on the GPU, not in the loader, so host RAM holds
    the TESSERA map once per sample instead of once per day. The channel
    order matches appending TESSERA after the base channels.
    """
    # Move the batch to the GPU; in TESSERA mode, copy the per-fire map onto every
    # day there and append it after the 43 regular channels.
    data = batch["data"].to(device, non_blocking=True)
    if "static" in batch:
        static = batch["static"].to(device, non_blocking=True)
        static = static.unsqueeze(2).expand(-1, -1, data.shape[2], -1, -1)
        data = torch.cat((data, static), dim=1)
    return data
