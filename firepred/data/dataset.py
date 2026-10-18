"""FireDataset: every window of every fire in a split, as one PyTorch Dataset.

Level 1 (data). Used by training/run.py and cli/check.py.

dataset[i] reads one window from its fire archive (data/archive.py) and runs
data/transforms.py on it:

    load      (27, T, 256, 256) float32 input, (256, 256) target
    normalize standardized channels
    augment   (training only) flips/rotation
    preprocess -> {"data": (43, T, 256, 256), "labels": (2, 256, 256)}

In TESSERA mode it also returns "static": (129, 256, 256), the fire's
normalized embedding plus a validity mask; the trainer repeats it across the
T days on the GPU (data/loader.py: batch_inputs).
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from firepred.config.channels import (
    DEGREE_CHANNELS,
    LANDCOVER_CHANNEL,
    N_STORED_CHANNELS,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
)
from firepred.config.dataset import ARCHIVE_FORMAT_VERSION, CROP_SIZE
from firepred.data.archive import find_fire_sources, fire_name, open_fire
from firepred.data.tessera import read_tessera
from firepred.data.transforms import Normalize, augment, landcover_one_hot_matrix, preprocess


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
    ):
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
            missing = [
                f"{fire_name(path)}.npz"
                for path in self.npz_files
                if not (self.tessera_dir / f"{fire_name(path)}.npz").exists()
            ]
            if missing:
                raise RuntimeError(
                    f"{len(missing)} archives in {self.npz_dir} have no TESSERA "
                    f"file in {self.tessera_dir}: {missing[:10]}"
                )
            for path in self.npz_files:
                name = fire_name(path)
                embedding, coverage = read_tessera(self.tessera_dir / f"{name}.npz")
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

        # [ours] a global index lets PyTorch shuffle windows across all fires.
        # Each entry is `(archive_path, window_index_inside_archive)`.
        # 4. Build the global index: one (fire, window number) entry per window across
        # all fires. Its length is the dataset length, and shuffling it mixes fires.
        self.index = []
        for npz_path in self.npz_files:
            with open_fire(npz_path) as archive:
                # Reject old archives rather than silently loading the whole
                # `data` member and defeating lazy decompression.
                if "num_windows" not in archive.files:
                    raise RuntimeError(
                        f"{npz_path} uses the old NPZ format. Expected "
                        "num_windows plus data_###/label_### members."
                    )

                number_of_windows = int(
                    np.asarray(archive["num_windows"]).item()
                )

                if "format_version" in archive.files:
                    format_version = int(
                        np.asarray(archive["format_version"]).item()
                    )
                    if format_version != ARCHIVE_FORMAT_VERSION:
                        raise RuntimeError(
                            f"{npz_path} has unsupported format version "
                            f"{format_version}; expected version {ARCHIVE_FORMAT_VERSION}"
                        )

                # Checking member names reads only the ZIP directory; it does
                # not decompress the large window arrays.
                for window_idx in range(number_of_windows):
                    suffix = f"{window_idx:03d}"
                    data_key = f"data_{suffix}"
                    label_key = f"label_{suffix}"

                    if data_key not in archive.files:
                        raise RuntimeError(
                            f"{npz_path} is missing member {data_key}"
                        )
                    if label_key not in archive.files:
                        raise RuntimeError(
                            f"{npz_path} is missing member {label_key}"
                        )

                    self.index.append((npz_path, window_idx))

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

    def __len__(self):
        # The number of windows across all fires.
        return len(self.index)

    def __getitem__(self, idx):
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

        # NPZ-loaded arrays normally own writable memory already. Copy only
        # if necessary because normalization modifies x in place.
        if not data_array.flags.writeable:
            data_array = data_array.copy()

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
        """Return (129, H, W): normalized embedding plus validity mask.

        Pixels whose TESSERA coverage is below the training threshold are
        zeroed (the normalized mean) and marked 0 in the mask channel.
        """
        npz_path, _ = self.index[idx]
        # The stored map was read in __init__; only normalization runs per window.
        embedding, valid = self.tessera_maps[fire_name(npz_path)]
        embedding = embedding.astype(np.float32)

        # Standardize the embedding with training-split statistics; pixels with too
        # little TESSERA coverage are set to 0 and flagged in the extra mask channel.
        mean = self.tessera_stats["mean"][:, None, None]
        std = self.tessera_stats["std"][:, None, None]
        embedding = np.where(valid, (embedding - mean) / std, 0.0)

        static = np.concatenate(
            (embedding, valid[None].astype(np.float32)),
            axis=0,
        ).astype(np.float32)
        return torch.from_numpy(static)
