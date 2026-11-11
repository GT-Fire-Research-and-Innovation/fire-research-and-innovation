"""From window archives to model input: transforms, FireDataset, loaders, batches.

Level 1 (data). Used by training/ and cli/check.py. FireDataset is the thin adapter
PyTorch needs: dataset[i] runs plain functions, each returning new arrays:

    read_window    (27, T, 256, 256) float32 input and (256, 256) target from its archive
    normalize      standardize every channel except angles and land cover (the training
                   split's statistics from window_config.json, or the paper's)
    augment        (training only) random flips and 90-degree rotation
    preprocess     observed wind direction -> degrees / 180 - 1 (the whole compass on
                   -1..1), sin(angle) for aspect and forecast wind direction, one-hot
                   land cover: -> {"data": (43, T, 256, 256), "labels": (2, 256, 256)}

In TESSERA mode an item also carries "static": (129, 256, 256), the fire's normalized
embedding plus a validity mask, and batch_inputs repeats it across the T days on the
GPU. With k x k TESSERA cells per VIIRS pixel (k > 1) "static" is (129, 256 k, 256 k)
float16 read from its file for every window (a k = 5 map is about 400 MiB, too large to
keep per fire); the model's TESSERA stem compresses it to the VIIRS grid.

With zone files (zones_dir, training split only) an item also carries "zones":
(2, 256, 256) int32, the window's zones and target owners (data/fire_zones.py),
flipped and rotated with the target. The loss splits the window into one sample per
fire from them (training/objectives.fire_samples); the inputs never change.
"""

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

from firepred.config.dataset import (
    BASE_MODEL_CHANNELS,
    CROP_SIZE,
    DEGREE_CHANNELS,
    LANDCOVER_CHANNEL,
    N_LANDCOVER_CLASSES,
    N_STORED_CHANNELS,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    TESSERA_CELLS_PER_PIXEL,
    TESSERA_CHANNELS,
    WIND_DIRECTION_CHANNEL,
)
from firepred.data.archive import (
    find_fire_sources,
    fire_name,
    index_fires,
    load_tessera_maps,
    open_windows,
    read_tessera,
    read_zone_header,
    read_zone_window,
    static_map,
    tessera_paths,
    zone_paths,
)


# ===== Transforms: normalize, augment, preprocess ===== #


def normalize(sample, mean, std, skip=()):
    """Return a standardized copy of one window with fixed dataset statistics.

    Each channel has its own mean and spread, computed by the released code over
    the training data. Standardizing ((x - mean) / std) puts every channel on a
    similar scale, which helps training. Skipped channels (angles and land
    cover) are copied unchanged, because preprocess encodes them differently.

    Args:
        sample (torch.Tensor | np.ndarray): one window, shape (C, T, H, W).
        mean (Sequence[float]): per-channel means (27 values).
        std (Sequence[float]): per-channel standard deviations (27 values).
        skip (Iterable[int]): channels left as they are.

    Returns:
        torch.Tensor | np.ndarray: the standardized copy, same type and shape.

    Raises:
        ValueError: if mean and std differ in length, or the window is not 4-D
            or has the wrong channel count.
    """
    # [ours] fail early when the statistics or the window are inconsistent.
    if len(mean) != len(std):
        raise ValueError(f"Normalization mean/std lengths differ: {len(mean)} versus {len(std)}")
    if sample.ndim != 4:
        raise ValueError(f"Expected input shaped (C, T, H, W), but found {tuple(sample.shape)}")
    if sample.shape[0] != len(mean):
        raise ValueError(f"Expected {len(mean)} channels for normalization, but found {sample.shape[0]}")

    # [released] normalize one channel at a time and skip angle/land-cover
    # channels, which preprocess() encodes instead. The per-channel arithmetic
    # is kept as released so results stay bit-identical.
    skip = set(skip)
    normalized = sample.copy() if isinstance(sample, np.ndarray) else sample.clone()
    for channel_idx in range(len(mean)):
        if channel_idx not in skip:
            normalized[channel_idx] = (sample[channel_idx] - mean[channel_idx]) / std[channel_idx]
    return normalized


def landcover_one_hot_matrix(n_classes=N_LANDCOVER_CLASSES):
    """Build the land-cover lookup table: row k one-hot for class k [released], row 0 (missing) all zeros [ours]."""
    return torch.cat((torch.zeros(1, n_classes), torch.eye(n_classes)), dim=0)


def augment(x, y, static=None, degree_channels=DEGREE_CHANNELS, rng=np.random):
    """Randomly flip and rotate a training window, its target and its TESSERA map together.

    Three random draws decide a horizontal flip, a vertical flip and a rotation
    by 0, 90, 180 or 270 degrees [released]. The same change is applied to the
    input, the target and the TESSERA map, so they stay aligned; angle channels
    are rotated too, so wind still points the right way. New tensors are
    returned and the inputs are left unchanged.

    Args:
        x (torch.Tensor): the input window, shape (C, T, H, W).
        y (torch.Tensor): the target, shape (2, H, W).
        static (torch.Tensor | None): the TESSERA map (129, H, W), or None.
        degree_channels (list[int]): the channels holding angles in degrees.
        rng (np.random.Generator | module): where the three draws come from; numpy's
            global generator by default, as released.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: (x, y, static) after the change.
    """
    hflip = bool(rng.random() > 0.5)
    vflip = bool(rng.random() > 0.5)
    rotate = int(np.floor(rng.random() * 4))

    if hflip:
        x = TF.hflip(x)
        y = TF.hflip(y)
        if static is not None:
            static = TF.hflip(static)

        x[degree_channels] = (360 - x[degree_channels]) % 360  # reflect angles across the vertical axis

    if vflip:
        x = TF.vflip(x)
        y = TF.vflip(y)
        if static is not None:
            static = TF.vflip(static)

        x[degree_channels] = (180 - x[degree_channels]) % 360  # reflect angles across the horizontal axis

    if rotate:
        angle = rotate * 90
        # torchvision rotates the last two dimensions of (2, H, W) and (2, T, H, W) alike.
        x = TF.rotate(x, angle)
        y = TF.rotate(y, angle)
        if static is not None:
            static = TF.rotate(static, angle)

        x[degree_channels] = (x[degree_channels] - angle) % 360  # turn angles with the raster

    return x, y, static


def preprocess(x, one_hot_matrix, degree_channels=DEGREE_CHANNELS, landcover_channel=LANDCOVER_CHANNEL):
    """Encode a window for the model: the angles as numbers in -1..1, and one-hot land cover.

    The observed wind direction becomes degrees / 180 - 1 [ours]: north -1,
    east -0.5, south 0, west 0.5, rising toward 1 back at north, so every
    direction has its own value (config.dataset.WIND_DIRECTION_CHANNEL). The
    other angles become sin(angle) [released], which gives north and south the
    same 0. The land-cover channel (class 1..17, 0 = missing) becomes 17 one-hot
    planes, so the stored 27 channels become 43. A new tensor is returned and x
    is unchanged.

    Args:
        x (torch.Tensor): a window, shape (27, T, H, W).
        one_hot_matrix (torch.Tensor): landcover_one_hot_matrix().
        degree_channels (list[int]): channels holding angles in degrees.
        landcover_channel (int): the land-cover channel.

    Returns:
        torch.Tensor: shape (43, T, H, W).

    Raises:
        ValueError: if land cover holds a value outside 0..17.
    """
    new_shape = (x.shape[1], x.shape[2], x.shape[3], one_hot_matrix.shape[1])
    landcover_classes = x[landcover_channel].long()

    # [paper] 0 is a missing value (preprocessing replaces NaN with 0), read
    # as "an absence of all land covers", so it gets an all-zero encoding;
    # the released code indexed it as -1, silently making it class 17.
    invalid_landcover = (landcover_classes < 0) | (landcover_classes > one_hot_matrix.shape[1])
    if torch.any(invalid_landcover):
        invalid_values = torch.unique(landcover_classes[invalid_landcover]).tolist()
        raise ValueError(f"Land-cover channel contains values outside 0..17: {invalid_values[:20]}")

    # Row 0 of the padded matrix is the all-zero encoding for class 0.
    landcover_encoding = one_hot_matrix[landcover_classes.flatten()].reshape(new_shape).permute(3, 0, 1, 2)

    # Assemble the output channel by channel, so `x` itself is never written:
    # angles are encoded after augmentation has rotated them, land cover
    # becomes its one-hot planes, every other channel is kept.
    degree_channels = set(degree_channels)
    pieces = []
    for channel in range(x.shape[0]):
        if channel == landcover_channel:
            pieces.append(landcover_encoding)
        elif channel == WIND_DIRECTION_CHANNEL and channel in degree_channels:
            # [ours] the whole compass on -1..1 (0 and 360 are the same north).
            pieces.append(torch.remainder(x[channel:channel + 1], 360.0) / 180.0 - 1.0)
        elif channel in degree_channels:
            pieces.append(torch.sin(torch.deg2rad(x[channel:channel + 1])))
        else:
            pieces.append(x[channel:channel + 1])
    return torch.cat(pieces, dim=0)


# ===== FireDataset, loaders and batches ===== #


# ----- One window: read, then prepare ----- #

# [released] angles and land cover are encoded by preprocess, not standardized.
NORMALIZE_SKIP = (*DEGREE_CHANNELS, LANDCOVER_CHANNEL)


def read_window(source, window_idx, stored_channels, ts_length, target_is_single_day, ts_interval=None):
    """Read exactly one window and its target from a fire archive, and check their shapes.

    Only this window's two members are decompressed. The target becomes two
    channels: background (0) and newly burned (1). A wrong shape fails here
    rather than deep inside the model.

    Args:
        source (Path): a p<id>.npz archive or unzipped p<id>/ folder.
        window_idx (int): the window's number inside the fire.
        stored_channels (int): stored channels per window (27).
        ts_length (int): input days per window.
        target_is_single_day (bool): the target is one (H, W) next-day map rather
            than a (T, H, W) sequence.
        ts_interval (int | None): days between window starts; a daily archive is cut
            at ts_length and this (None: its own interval).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (x (27, T, H, W) float32, y (2, H, W) long).

    Raises:
        ValueError: if the stored input or target has an unexpected shape.
    """
    data_key, label_key = f"data_{window_idx:03d}", f"label_{window_idx:03d}"
    with open_windows(source, ts_length, ts_interval) as archive:
        data_array = archive[data_key]
        label_array = archive[label_key]

    # No squeeze(): it could remove a real temporal dimension when T == 1.
    x = torch.from_numpy(data_array).to(dtype=torch.float32)
    label = torch.from_numpy(label_array)

    expected_x_shape = (stored_channels, ts_length, CROP_SIZE, CROP_SIZE)
    if tuple(x.shape) != expected_x_shape:
        raise ValueError(f"{source}:{data_key} has shape {tuple(x.shape)}; expected {expected_x_shape}")

    # [released] target_is_single_day: one map, or a temporal sequence.
    # Prediction targets are one 256 x 256 map (the newly burned pixels of the next day).
    if target_is_single_day:
        # Accept a harmless leading singleton left by older exports.
        if label.ndim == 3 and label.shape[0] == 1:
            label = label[0]
        if label.ndim != 2:
            raise ValueError(f"target_is_single_day=True requires (H, W), but "
                             f"{source}:{label_key} has shape {tuple(label.shape)}")
    else:
        if label.ndim != 3:
            raise ValueError(f"target_is_single_day=False requires (T, H, W), but "
                             f"{source}:{label_key} has shape {tuple(label.shape)}")
        if label.shape[0] != ts_length:
            raise ValueError(f"{source}:{label_key} contains {label.shape[0]} target frames; expected {ts_length}")

    if tuple(label.shape[-2:]) != (CROP_SIZE, CROP_SIZE):
        raise ValueError(f"{source}:{label_key} has spatial shape {tuple(label.shape[-2:])}; "
                         f"expected ({CROP_SIZE}, {CROP_SIZE})")

    # [released] the binary target as two channels: 0 background, 1 fire.
    # stack infers either (2,H,W) or (2,T,H,W).
    y = torch.stack((label == 0, label > 0), dim=0).long()
    return x, y


def prepare_window(x, y, one_hot_matrix, static=None, train=False, rng=np.random,
                   mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD, zones=None):
    """Turn a read window into model input: standardize, (training only) augment, encode.

    The steps run in the released order: normalize before augmentation, then
    the angles as numbers in -1..1 and one-hot land cover. Every step returns
    new tensors, so x and y are unchanged. Zone maps ride along as extra target
    channels, so every flip and rotation moves them exactly as it moves the target.

    Args:
        x (torch.Tensor): the stored window, (27, T, H, W).
        y (torch.Tensor): the two-channel target, (2, H, W).
        one_hot_matrix (torch.Tensor): landcover_one_hot_matrix().
        static (torch.Tensor | None): the fire's TESSERA map (129, H, W), or None.
        train (bool): augment with random flips and rotation (training split only).
        rng (np.random.Generator | module): where augmentation draws come from.
        mean (Sequence[float]): per-channel means (default: the paper's).
        std (Sequence[float]): per-channel standard deviations (default: the paper's).
        zones (torch.Tensor | None): the window's (zones, target owners), (2, H, W), or None.

    Returns:
        dict[str, torch.Tensor]: {"data": (43, T, H, W), "labels": (2, H, W)}, plus
        "static": (129, H, W) when a TESSERA map is given and "zones": (2, H, W) int32
        when zone maps are given.
    """
    if zones is not None:
        y = torch.cat((y, zones.long()))  # channels 2 and 3: zones and target owners
    x = normalize(x, mean, std, skip=NORMALIZE_SKIP)
    # [released] augment training examples but leave validation unchanged.
    if train:
        x, y, static = augment(x, y, static, list(DEGREE_CHANNELS), rng=rng)
    # Degrees -> -1..1 (observed wind [ours]; the others sine [released]), land cover -> one-hot.
    x = preprocess(x, one_hot_matrix, list(DEGREE_CHANNELS))
    sample = {"data": x, "labels": y[:2]}
    if static is not None:
        sample["static"] = static
    if zones is not None:
        sample["zones"] = y[2:].int()
    return sample


def valid_tessera_maps(tessera_maps, sources, min_coverage):
    """Pair every fire's TESSERA embedding with its validity mask, computed once per fire, not per window."""
    maps = {}
    for path in sources:
        name = fire_name(path)
        embedding, coverage = tessera_maps[name]
        maps[name] = (embedding, coverage.astype(np.float32) >= min_coverage)
    return maps


def group_by_fire(index):
    """Group a window index by fire, {fire id without "p": item positions}, for per-fire test scores."""
    groups = {}
    for dataset_index, (archive_path, _) in enumerate(index):
        groups.setdefault(fire_name(archive_path).removeprefix("p"), []).append(dataset_index)
    return groups


@dataclass(frozen=True)
class ZoneLayout:
    """Know where each fire's zone file is and which zone window belongs to each of its windows.

    A zone file has a window for every day. A daily archive read at -it k
    (format 3) has a window every k days, so its window w is zone window w x k;
    other archives step 1.

    Attributes:
        files (dict[str, Path]): {"p<id>": zone file}.
        strides (dict[str, int]): {"p<id>": zone windows per archive window}.
        settings (SplitSettings): the thresholds every zone file was built with.
    """

    files: dict
    strides: dict
    settings: object


def zone_layout(zones_dir, sources, ts_length, ts_interval=None):
    """Pair every fire with its zone file and check that each file fits its archive.

    Every file must exist, predict the same day as the windows (its first target
    day is the window length), cover every window and share one set of
    settings, so zones of two distances never mix. Only headers are read here.

    Args:
        zones_dir (str | Path): the folder of zones/p<id>.npz files.
        sources (list[Path]): the split's fire archives.
        ts_length (int): input days per window.
        ts_interval (int | None): days between window starts of daily archives (None: their own).

    Returns:
        ZoneLayout: the files, the window steps and the shared settings.

    Raises:
        RuntimeError: if a fire has no zone file.
        ValueError: if a file does not fit its archive, or two files were built differently.
    """
    files, strides, settings = zone_paths(zones_dir, sources), {}, None
    for source in sources:
        name = fire_name(source)
        header = read_zone_header(files[name])
        with open_windows(source, ts_length, ts_interval) as archive:
            windows = int(np.asarray(archive["num_windows"]).item())
            stride = getattr(archive, "interval", 1)  # daily archives only; others are read as built
        if header.first_label_day != ts_length:
            raise ValueError(f"{files[name]} was built for {header.first_label_day}-day windows, not -ts {ts_length}")
        if (stride == 1 and header.num_windows != windows) or header.num_windows < (windows - 1) * stride + 1:
            raise ValueError(f"{files[name]} has {header.num_windows} windows; {source} needs one for each of its "
                             f"{windows} windows every {stride} days: rebuild the zone files from these archives")
        if settings is not None and header.settings != settings:
            changed = ", ".join(f"{field.name} {getattr(header.settings, field.name)} (others {getattr(settings, field.name)})"
                                for field in fields(settings)
                                if getattr(header.settings, field.name) != getattr(settings, field.name))
            raise ValueError(f"{files[name]} was built with other settings than the other zone files: {changed}. "
                             "Use one zones folder per setting")
        settings = header.settings
        strides[name] = stride
    return ZoneLayout(files, strides, settings)


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
        split_dir,
        ts_length,
        augment=False,
        stored_channels=N_STORED_CHANNELS,
        target_is_single_day=False,
        tessera_dir=None,
        tessera_stats=None,
        tessera_maps=None,
        cells_per_pixel=TESSERA_CELLS_PER_PIXEL,
        lazy_tessera=None,
        normalization=None,
        zones_dir=None,
        ts_interval=None,
    ):
        """Find a split's fires, read their TESSERA maps if asked, and index every window.

        All checks happen here, before training: the split must have fires, every
        fire needs a TESSERA file in TESSERA mode (and a fitting zone file with
        zones_dir), and every archive must be in the current format. Windows
        themselves are read only when asked for.

        Args:
            split_dir (str | Path): a split folder of p<id>.npz archives or p<id>/ folders.
            ts_length (int): input days per window.
            augment (bool): random flips and rotation (training split only).
            stored_channels (int): stored channels per window (27).
            target_is_single_day (bool): targets are one (H, W) next-day map
                (prediction) rather than a (T, H, W) sequence.
            tessera_dir (str | Path | None): folder of tessera/p<id>.npz files, or None.
            tessera_stats (dict | None): training statistics (compute_tessera_stats).
            tessera_maps (dict | None): this split's maps (load_tessera_maps), when the
                caller already read them; otherwise they are read from tessera_dir.
            cells_per_pixel (int): TESSERA cells per VIIRS pixel side every map must have.
            lazy_tessera (bool | None): read each window's map from its file instead
                of keeping every fire's map in memory. Default: only when
                cells_per_pixel > 1.
            normalization (tuple[Sequence[float], Sequence[float]] | None): the
                (mean, std) every window is standardized with; None uses the
                paper's NORMALIZATION_MEAN/STD.
            zones_dir (str | Path | None): folder of zones/p<id>.npz files; each item
                then carries its window's zone maps (training split only).
            ts_interval (int | None): days between window starts. Daily archives store
                days, so they are cut into windows of ts_length every ts_interval days
                (None: the interval they were converted with). Format-2 archives hold
                fixed windows and ignore it.

        Raises:
            RuntimeError: if the split has no fires or windows, or a TESSERA or zone file is missing.
            ValueError: if tessera_dir is given without tessera_stats, or a zone file does not fit.
        """
        # [ours] data and labels live together in version-2 archives, or in
        # unzipped p<id>/ folders of the same members.
        self.split_dir = Path(split_dir)
        self.sources = find_fire_sources(self.split_dir)
        if not self.sources:
            raise RuntimeError(f"No .npz archives or unzipped fire folders found in {self.split_dir}")

        # [ours] optional TESSERA mode: each p<id>.npz pairs with <tessera_dir>/p<id>.npz.
        # Every fire's map is read here, once, rather than in every __getitem__;
        # DataLoader workers forked later (Linux, including Kaggle) share the arrays.
        # With large maps (k > 1 cells per pixel) only the file paths are kept and
        # each window reads its fire's map (lazy_tessera).
        self.tessera_dir = Path(tessera_dir) if tessera_dir is not None else None
        self.tessera_stats = tessera_stats
        self.cells_per_pixel = cells_per_pixel
        self.lazy_tessera = cells_per_pixel > 1 if lazy_tessera is None else lazy_tessera
        self.tessera_maps = {}
        self.tessera_files = {}
        if self.tessera_dir is not None:
            if tessera_stats is None:
                raise ValueError("tessera_stats is required with tessera_dir")
            if self.lazy_tessera:
                self.tessera_files = tessera_paths(self.tessera_dir, self.sources)
            else:
                if tessera_maps is None:
                    tessera_maps = load_tessera_maps(self.tessera_dir, self.sources, cells_per_pixel)
                self.tessera_maps = valid_tessera_maps(tessera_maps, self.sources, tessera_stats["min_coverage"])

        self.stored_channels = stored_channels
        self.mean, self.std = normalization if normalization is not None else (NORMALIZATION_MEAN, NORMALIZATION_STD)
        if len(self.mean) != stored_channels or len(self.std) != stored_channels:
            raise ValueError(f"normalization needs {stored_channels} means and stds, got "
                             f"{len(self.mean)} and {len(self.std)}")
        self.ts_length = ts_length
        self.ts_interval = ts_interval
        self.target_is_single_day = target_is_single_day
        self.augment = augment

        # [ours] the global index: one (fire, window number) entry per window
        # across all fires (data/archive.index_fires), so PyTorch can shuffle
        # windows across fires.
        self.index = index_fires(self.sources, ts_length, ts_interval)
        if not self.index:
            raise RuntimeError(f"NPZ files in {self.split_dir} contain no temporal windows")
        self.one_hot_matrix = landcover_one_hot_matrix()
        # [ours] fire split zones: only headers now; each item reads its window's two maps.
        self.zones = zone_layout(zones_dir, self.sources, ts_length, ts_interval) if zones_dir is not None else None

    def windows_by_fire(self):
        """Group the dataset's windows by fire."""
        return group_by_fire(self.index)

    def __len__(self):
        """Count the windows across all fires."""
        return len(self.index)

    def __getitem__(self, idx):
        """Return window idx ready for the model.

        The window is read from its fire archive, then prepared (standardized,
        augmented in training, encoded). In TESSERA mode the fire's map is
        normalized and added; it is returned once as (129, H, W) (or, with k cells
        per pixel, (129, k H, k W) float16 read from its file), and the trainer
        repeats it across the days on the GPU.

        Args:
            idx (int): the window's position in the index.

        Returns:
            dict[str, torch.Tensor]: {"data": (43, T, H, W), "labels": (2, H, W)}, plus
            "static": (129, H, W), or (129, k H, k W) float16 when lazy, in TESSERA mode,
            and "zones": (2, H, W) int32 with zone files.
        """
        source, window_idx = self.index[idx]
        x, y = read_window(source, window_idx, self.stored_channels, self.ts_length, self.target_is_single_day,
                           self.ts_interval)
        static = None
        if self.tessera_dir is not None:
            static = torch.from_numpy(self.static_tessera(fire_name(source)))
        zones = self.window_zones(source, window_idx, y) if self.zones is not None else None
        sample = prepare_window(x, y, self.one_hot_matrix, static, self.augment, mean=self.mean, std=self.std,
                                zones=zones)
        if self.lazy_tessera and "static" in sample:
            # float16 halves the host memory of a large map (the mask, 0 or 1, stays exact);
            # it is cast only after augmentation, which needs float32 on the CPU.
            sample["static"] = sample["static"].half()
        return sample

    def window_zones(self, source, window_idx, y):
        """Read one window's zone maps and check that they describe its target.

        The new pixels the zone file assigns must be exactly the target's fire
        pixels. Zone files built from other archives, or read at the wrong step,
        therefore fail here instead of training on misplaced fires.

        Args:
            source (Path): the window's fire archive.
            window_idx (int): the window's number inside the fire.
            y (torch.Tensor): the window's two-channel target, (2, H, W).

        Returns:
            torch.Tensor: (2, H, W) int32, the zones and the target owners.

        Raises:
            ValueError: if the zone file's new pixels are not the target's.
        """
        name = fire_name(source)
        zone_window = window_idx * self.zones.strides[name]
        zones, owners = read_zone_window(self.zones.files[name], zone_window)
        if not np.array_equal(owners != 0, y[1].numpy() > 0):
            raise ValueError(f"{self.zones.files[name]} window {zone_window} does not match the target of {source} "
                             f"window {window_idx}: rebuild the zone files from these archives")
        return torch.from_numpy(np.stack((zones, owners)).astype(np.int32))

    def static_tessera(self, name):
        """Give one fire's normalized TESSERA map and validity mask; lazy (k > 1) maps are read from their file now."""
        if not self.lazy_tessera:
            embedding, valid = self.tessera_maps[name]
            return static_map(embedding, valid, self.tessera_stats)
        embedding, coverage = read_tessera(self.tessera_files[name], self.cells_per_pixel)
        valid = coverage.astype(np.float32) >= self.tessera_stats["min_coverage"]
        return static_map(embedding, valid, self.tessera_stats)


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
    loader_options = {"dataset": dataset, "batch_size": batch_size, "shuffle": shuffle,
                      "num_workers": num_workers, "pin_memory": pin_memory}
    if generator is not None:
        loader_options["generator"] = generator

    if num_workers > 0:  # one prefetched batch per worker, not two: each holds decompressed windows
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
    sample = dataset[0]
    data_shape = tuple(sample["data"].shape)
    label_shape = tuple(sample["labels"].shape)
    static_channels = sample["static"].shape[0] if "static" in sample else 0
    if static_channels and sample["static"].shape[-1] % data_shape[-1]:
        raise ValueError(f"{dataset_name} TESSERA map {tuple(sample['static'].shape)} is not a whole multiple "
                         f"of the window size {data_shape[-2:]}")
    if len(data_shape) != 4:
        raise ValueError(f"{dataset_name} sample must be (C,T,H,W), found {data_shape}")
    actual_channels = data_shape[0] + static_channels
    if actual_channels != expected_model_channels:
        raise ValueError(
            f"{dataset_name} FireDataset produces {actual_channels} channels after preprocessing, but the model "
            f"is configured for {expected_model_channels}. The 27-channel archive plus 17-class land-cover "
            f"expansion gives {BASE_MODEL_CHANNELS} channels; --tessera adds {TESSERA_CHANNELS} more."
        )
    print(f"{dataset_name} windows={len(dataset)}, data shape={data_shape}, label shape={label_shape}"
          + (f", static TESSERA channels={static_channels}" if static_channels else ""))
    return data_shape


def batch_inputs(batch, device):
    """Assemble a batch into the model's input on the device.

    The batch's 43 channels are moved to the device. In TESSERA mode the static
    (129, H, W) map is copied onto every day there and appended, giving 172
    channels; doing the copy on the GPU keeps host RAM at one map per sample.
    A finer map (k cells per pixel) cannot be appended pixel by pixel, so it is
    returned beside the data for the model's TESSERA stem.

    Args:
        batch (dict[str, torch.Tensor]): a DataLoader batch with "data" (and "static").
        device (torch.device | str): where the model runs.

    Returns:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]: (B, C, T, H, W) on device,
        or (data (B, 43, T, H, W), static (B, 129, k H, k W)) for a finer map.
    """
    data = batch["data"].to(device, non_blocking=True)
    if "static" in batch:
        static = batch["static"].to(device, non_blocking=True)
        if static.shape[-2:] != data.shape[-2:]:
            return data, static
        static = static.unsqueeze(2).expand(-1, -1, data.shape[2], -1, -1)
        data = torch.cat((data, static), dim=1)
    return data
