"""From window archives to model input: TESSERA maps, FireDataset, loaders, batches.

Level 1 (data). Used by training/ and cli/check.py. Three parts, in the order
data flows through them:

    TESSERA      read_tessera, load_tessera_maps, compute_tessera_stats:
                 per-fire embedding files and their normalization statistics
    FireDataset  every window of every fire in a split, as one Dataset. It is
                 a thin adapter PyTorch needs: dataset[i] calls read_window
                 (one window from its fire archive, data/archive.py) and
                 prepare_window (data/transforms.py), both plain functions:
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
from firepred.data.transforms import augment, landcover_one_hot_matrix, normalize, preprocess


# ----- One window: read, then prepare ----- #

# [released] angles and land cover are encoded by preprocess, not standardized.
NORMALIZE_SKIP = (*DEGREE_CHANNELS, LANDCOVER_CHANNEL)


def read_window(source, window_idx, n_channel, ts_length, target_is_single_day):
    """Read exactly one window and its target from a fire archive, and check their shapes.

    Only this window's two members are decompressed. The target becomes two
    channels: background (0) and newly burned (1). A wrong shape fails here
    rather than deep inside the model.

    Args:
        source (Path): a p<id>.npz archive or unzipped p<id>/ folder.
        window_idx (int): the window's number inside the fire.
        n_channel (int): stored channels per window (27).
        ts_length (int): input days per window.
        target_is_single_day (bool): the target is one (H, W) next-day map rather
            than a (T, H, W) sequence.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (x (27, T, H, W) float32, y (2, H, W) long).

    Raises:
        ValueError: if the stored input or target has an unexpected shape.
    """
    suffix = f"{window_idx:03d}"
    data_key = f"data_{suffix}"
    label_key = f"label_{suffix}"

    with open_fire(source) as archive:
        data_array = archive[data_key]
        label_array = archive[label_key]

    # No squeeze(): it could remove a real temporal dimension when T == 1.
    x = torch.from_numpy(data_array).to(dtype=torch.float32)
    label = torch.from_numpy(label_array)

    expected_x_shape = (n_channel, ts_length, CROP_SIZE, CROP_SIZE)
    if tuple(x.shape) != expected_x_shape:
        raise ValueError(f"{source}:{data_key} has shape {tuple(x.shape)}; expected {expected_x_shape}")

    # [released] target_is_single_day: one map, or a temporal sequence.
    # Prediction targets are one 256 x 256 map (the newly burned pixels of the next day).
    if target_is_single_day:
        # Accept a harmless leading singleton left by older exports.
        if label.ndim == 3 and label.shape[0] == 1:
            label = label[0]
        if label.ndim != 2:
            raise ValueError(
                f"target_is_single_day=True requires (H, W), but "
                f"{source}:{label_key} has shape {tuple(label.shape)}"
            )
    else:
        if label.ndim != 3:
            raise ValueError(
                f"target_is_single_day=False requires (T, H, W), but "
                f"{source}:{label_key} has shape {tuple(label.shape)}"
            )
        if label.shape[0] != ts_length:
            raise ValueError(f"{source}:{label_key} contains {label.shape[0]} target frames; expected {ts_length}")

    if tuple(label.shape[-2:]) != (CROP_SIZE, CROP_SIZE):
        raise ValueError(
            f"{source}:{label_key} has spatial shape {tuple(label.shape[-2:])}; expected ({CROP_SIZE}, {CROP_SIZE})"
        )

    # [released] the binary target as two channels: 0 background, 1 fire.
    # stack infers either (2,H,W) or (2,T,H,W).
    y = torch.stack((label == 0, label > 0), dim=0).long()
    return x, y


def prepare_window(x, y, one_hot_matrix, static=None, augment_window=False, rng=np.random):
    """Turn a read window into model input: standardize, (training only) augment, encode.

    The steps run in the released order: normalize before augmentation, then
    sin() of the angles and one-hot land cover. Every step returns new tensors,
    so x and y are unchanged.

    Args:
        x (torch.Tensor): the stored window, (27, T, H, W).
        y (torch.Tensor): the two-channel target, (2, H, W).
        one_hot_matrix (torch.Tensor): landcover_one_hot_matrix().
        static (torch.Tensor | None): the fire's TESSERA map (129, H, W), or None.
        augment_window (bool): random flips and rotation (training only).
        rng (np.random.Generator | module): where augmentation draws come from.

    Returns:
        dict[str, torch.Tensor]: {"data": (43, T, H, W), "labels": (2, H, W)}, plus
        "static": (129, H, W) when a TESSERA map is given.
    """
    x = normalize(x, NORMALIZATION_MEAN, NORMALIZATION_STD, skip=NORMALIZE_SKIP)
    # [released] augment training examples but leave validation unchanged.
    if augment_window:
        x, y, static = augment(x, y, static, list(DEGREE_CHANNELS), rng=rng)
    # [released] degrees -> sine, land cover -> one-hot.
    x = preprocess(x, one_hot_matrix, list(DEGREE_CHANNELS))
    sample = {"data": x, "labels": y}
    if static is not None:
        sample["static"] = static
    return sample


def valid_tessera_maps(tessera_maps, sources, min_coverage):
    """Pair every fire's TESSERA embedding with its validity mask.

    A TESSERA cell is valid when enough of its 10 m pixels had data. The mask
    is computed once per fire here, not once per window.

    Args:
        tessera_maps (dict): {fire name: (embedding, coverage)} (load_tessera_maps).
        sources (list[Path]): the split's fire archives.
        min_coverage (float): the smallest valid coverage (from the training statistics).

    Returns:
        dict[str, tuple[np.ndarray, np.ndarray]]: {fire name: (embedding, valid mask)}.
    """
    maps = {}
    for path in sources:
        name = fire_name(path)
        embedding, coverage = tessera_maps[name]
        maps[name] = (embedding, coverage.astype(np.float32) >= min_coverage)
    return maps


def group_by_fire(index):
    """Group a window index by fire.

    Test scores are computed per fire and then averaged, so the evaluator needs
    to know which items belong to which fire. Fire ids are given without the
    leading "p".

    Args:
        index (list[tuple[Path, int]]): (fire archive, window number) per item.

    Returns:
        dict[str, list[int]]: {fire id: item positions of its windows}, in index order.
    """
    groups = {}
    for dataset_index, (archive_path, _) in enumerate(index):
        groups.setdefault(fire_name(archive_path).removeprefix("p"), []).append(dataset_index)
    return groups


# ----- FireDataset ----- #


class FireDataset(Dataset):
    """Serve every window of every fire in one split as one PyTorch Dataset.

    PyTorch's loaders need a Dataset object, so this is a thin adapter: it
    checks and indexes the split once, then item i is read_window followed by
    prepare_window. The index runs across all fires, so shuffling mixes them.
    In TESSERA mode each item also carries the fire's static TESSERA map.
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
        """Find a split's fires, read their TESSERA maps if asked, and index every window.

        All checks happen here, before training: the split must have fires, every
        fire needs a TESSERA file in TESSERA mode, and every archive must be in the
        current format. Windows themselves are read only when asked for.

        Args:
            npz_dir (str | Path): a split folder of p<id>.npz archives or p<id>/ folders.
            ts_length (int): input days per window.
            use_augmentations (bool): random flips and rotation (training split only).
            n_channel (int): stored channels per window (27).
            target_is_single_day (bool): targets are one (H, W) next-day map
                (prediction) rather than a (T, H, W) sequence.
            tessera_dir (str | Path | None): folder of tessera/p<id>.npz files, or None.
            tessera_stats (dict | None): training statistics (compute_tessera_stats).
            tessera_maps (dict | None): this split's maps (load_tessera_maps), when the
                caller already read them; otherwise they are read from tessera_dir.

        Raises:
            RuntimeError: if the split has no fires or windows, or a TESSERA file is missing.
            ValueError: if tessera_dir is given without tessera_stats.
        """
        # [ours] data and labels live together in version-2 archives, or in
        # unzipped p<id>/ folders of the same members.
        self.npz_dir = Path(npz_dir)
        self.npz_files = find_fire_sources(self.npz_dir)
        if not self.npz_files:
            raise RuntimeError(f"No .npz archives or unzipped fire folders found in {self.npz_dir}")

        # [ours] optional TESSERA mode: each p<id>.npz pairs with <tessera_dir>/p<id>.npz.
        # Every fire's map is read here, once, rather than in every __getitem__;
        # DataLoader workers forked later (Linux, including Kaggle) share the arrays.
        self.tessera_dir = Path(tessera_dir) if tessera_dir is not None else None
        self.tessera_stats = tessera_stats
        self.tessera_maps = {}
        if self.tessera_dir is not None:
            if tessera_stats is None:
                raise ValueError("tessera_stats is required with tessera_dir")
            if tessera_maps is None:
                tessera_maps = load_tessera_maps(self.tessera_dir, self.npz_files)
            self.tessera_maps = valid_tessera_maps(tessera_maps, self.npz_files, tessera_stats["min_coverage"])

        self.n_channel = n_channel
        self.ts_length = ts_length
        self.target_is_single_day = target_is_single_day
        self.use_augmentations = use_augmentations

        # [ours] the global index: one (fire, window number) entry per window
        # across all fires (data/archive.index_fires), so PyTorch can shuffle
        # windows across fires.
        self.index = index_fires(self.npz_files)
        if not self.index:
            raise RuntimeError(f"NPZ files in {self.npz_dir} contain no temporal windows")
        self.one_hot_matrix = landcover_one_hot_matrix()

    def windows_by_fire(self):
        """Group the dataset's windows by fire.

        The evaluator scores each fire on its own. See group_by_fire.

        Returns:
            dict[str, list[int]]: {fire id: dataset indices of its windows}, in index order.
        """
        return group_by_fire(self.index)

    def __len__(self):
        """Count the windows across all fires.

        This is the dataset length PyTorch uses. It is the number of windows, not
        the number of fires.

        Returns:
            int: the number of windows.
        """
        return len(self.index)

    def __getitem__(self, idx):
        """Return window idx ready for the model.

        The window is read from its fire archive, then prepared (standardized,
        augmented in training, encoded). In TESSERA mode the fire's map is
        normalized and added; it is returned once as (129, H, W), and the trainer
        repeats it across the days on the GPU.

        Args:
            idx (int): the window's position in the index.

        Returns:
            dict[str, torch.Tensor]: {"data": (43, T, H, W), "labels": (2, H, W)}, plus
            "static": (129, H, W) in TESSERA mode.
        """
        source, window_idx = self.index[idx]
        x, y = read_window(source, window_idx, self.n_channel, self.ts_length, self.target_is_single_day)
        static = None
        if self.tessera_dir is not None:
            embedding, valid = self.tessera_maps[fire_name(source)]
            static = torch.from_numpy(static_map(embedding, valid, self.tessera_stats))
        return prepare_window(x, y, self.one_hot_matrix, static, self.use_augmentations)


# ----- Loaders and batches ----- #


def make_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory, generator=None):
    """Create the DataLoader that batches a dataset's windows.

    Training shuffles; validation and test keep a fixed order. With workers,
    each keeps only one batch ready ahead (not PyTorch's two) to save RAM, and
    workers persist across epochs. A generator gives the loader its own random
    stream, so loading never draws from the global random numbers.

    Args:
        dataset (torch.utils.data.Dataset): the windows to batch.
        batch_size (int): windows per batch.
        shuffle (bool): reshuffle every epoch.
        num_workers (int): loading processes (0 = in the main process).
        pin_memory (bool): pin batches in host memory for faster GPU copies.
        generator (torch.Generator | None): the loader's own random generator.

    Returns:
        torch.utils.data.DataLoader: the loader.
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
    """Check that a dataset produces what the model expects, before any training.

    One window is loaded and its channel count (including TESSERA) compared
    with the model's input width. A mismatch therefore fails at once instead
    of deep inside a training step. The shapes are printed.

    Args:
        dataset (FireDataset): the dataset to check.
        expected_model_channels (int): the model's input channels (43 or 172).
        dataset_name (str): a name for messages, e.g. "validation".

    Returns:
        tuple: the sample's data shape.

    Raises:
        RuntimeError: if the dataset is empty.
        ValueError: if the shape or channel count is wrong.
    """
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
    return data_shape


def batch_inputs(batch, device):
    """Assemble a batch into the model's input tensor on the device.

    The batch's 43 channels are moved to the device. In TESSERA mode the static
    (129, H, W) map is copied onto every day there and appended, giving 172
    channels; doing the copy on the GPU keeps host RAM at one map per sample.

    Args:
        batch (dict[str, torch.Tensor]): a DataLoader batch with "data" (and "static").
        device (torch.device | str): where the model runs.

    Returns:
        torch.Tensor: shape (B, C, T, H, W) on device.
    """
    # Move the batch to the GPU; in TESSERA mode, copy the per-fire map onto every
    # day there and append it after the 43 regular channels.
    data = batch["data"].to(device, non_blocking=True)
    if "static" in batch:
        static = batch["static"].to(device, non_blocking=True)
        static = static.unsqueeze(2).expand(-1, -1, data.shape[2], -1, -1)
        data = torch.cat((data, static), dim=1)
    return data
