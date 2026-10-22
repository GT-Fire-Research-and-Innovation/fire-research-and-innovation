"""Data statistics and the data root's window_config.json that carries them.

Level 1 (data). `dataset count-pixels` and `compute-normalization` (data/windows_master.py)
write them; training/experiment.py and cli/check.py read them before anything else.

Every data root holds <data-root>/window_config.json, a copy of its window repo's
settings (data/hub.py) that download, data.zip and the TESSERA view carry along. Two
statistics blocks are added to it:

    pixel_counts    per split: background, fire and other target pixels. Next-day growth
                    covers well under 1 % of pixels, so --ce-weight auto weighs fire up by
                    train's background-to-fire ratio (the paper's 446.7836 is that ratio
                    for its training data; other windows, years or label rules differ).
    normalization   per-channel mean and std of the training inputs, which standardize
                    every window instead of the paper's NORMALIZATION_MEAN/STD.

Both are exactly repeatable. Counts follow read_window's rule (label == 0 background,
label > 0 fire), read only the label_### members and are whole numbers; augmentation only
moves pixels, so they are what the loss sees every epoch. Moments are folded window by
window with Chan's parallel update, which stays accurate when a channel's mean dwarfs its
spread (brightness temperatures near 300 K vary by about 10 K), and fires merge in sorted
order whatever the thread count. Memory stays at one decompressed window per thread
(27 x T x 256 x 256 float32, about 42 MB for T = 6). Every stored value counts, including
the zeros window building writes for missing data.

Training refuses -ts/-it that differ from the file. Each block lists the fires it was
computed from: a block missing a fire now in the split is not used (the data changed
since), while fires the block has and the split lacks are only noted (a TESSERA view
leaves some out). A statistic the file cannot supply (no file, no block, or a block
missing fires) raises MissingStatistics; training then falls back to the official
TS-SatFire values and records that in W&B.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from firepred.config.dataset import N_STORED_CHANNELS, STORED_CHANNEL_NAMES, WINDOW_CONFIG_FILE
from firepred.data.archive import find_fire_sources, fire_name, open_fire


# ===== Target pixel counts ===== #


# Threads reading fires at once: decompression releases the GIL, so a few threads help on slow disks.
MAX_COUNT_WORKERS = 8


@dataclass(frozen=True)
class LabelCounts:
    """Hold the exact target pixel counts of one split.

    Every window's target is counted, as the loss sees it. Pixels that are
    neither background nor fire (NaN or negative values) are counted apart,
    since the loss gives them neither class.

    Attributes:
        split_dir (Path): the split folder that was counted.
        fires (int): fires (archives or unzipped folders).
        windows (int): windows across those fires.
        background (int): target pixels equal to 0.
        fire (int): target pixels above 0.
        other (int): target pixels that are neither (NaN or negative).
        names (tuple[str, ...]): the fires counted ("p<id>"), sorted.
    """

    split_dir: Path
    fires: int
    windows: int
    background: int
    fire: int
    other: int = 0
    names: tuple = ()

    @property
    def fire_fraction(self):
        """Give the share of labelled pixels that are fire."""
        labelled = self.background + self.fire
        return self.fire / labelled if labelled else 0.0

    @property
    def background_per_fire(self):
        """Give the ratio of background to fire pixels: the fire class weight that balances the two."""
        if self.fire == 0:
            raise ValueError(f"{self.split_dir} has no fire pixels; the background-to-fire ratio is undefined")
        return self.background / self.fire

    def record(self):
        """Describe the counts as one split's entry in the pixel_counts block of window_config.json.

        The ratio is written as well, for people reading the file; reading the
        entry back recomputes it from the counts. The fire list lets training
        refuse counts that miss a fire it would train on.

        Returns:
            dict: fires, windows, the three counts and background per fire (None without fire).
        """
        return {
            "fires": list(self.names),
            "windows": self.windows,
            "background": self.background,
            "fire": self.fire,
            "other": self.other,
            "background_per_fire": self.background_per_fire if self.fire else None,
        }

    @classmethod
    def from_record(cls, record, split_dir):
        """Read one split's entry of the pixel_counts block back."""
        names = tuple(record["fires"])
        return cls(Path(split_dir), len(names), int(record["windows"]), int(record["background"]),
                   int(record["fire"]), int(record["other"]), names)

    def summary(self):
        """Describe the counts in one line for the console.

        The line names the split, the windows, both counts and the ratio. Pixels
        of neither class are mentioned only when there are some.

        Returns:
            str: the summary.
        """
        ratio = f"{self.background_per_fire:.4f}" if self.fire else "undefined (no fire)"
        other = f", {self.other:,} neither" if self.other else ""
        return (
            f"{self.split_dir}: {self.windows:,} windows in {self.fires:,} fires, "
            f"{self.background:,} background and {self.fire:,} fire pixels{other} "
            f"(fire {self.fire_fraction:.4%}); background per fire = {ratio}"
        )


def count_fire_labels(source):
    """Count one fire's target pixels by class, window by window.

    Only the fire's label_### members are read. The counts are whole numbers.

    Args:
        source (Path): a p<id>.npz archive or unzipped p<id>/ folder.

    Returns:
        tuple[int, int, int, int]: (windows, background, fire, other).

    Raises:
        RuntimeError: on an old-format archive without num_windows.
    """
    with open_fire(source) as archive:
        if "num_windows" not in archive.files:
            raise RuntimeError(f"{source} uses the old NPZ format. Expected num_windows plus label_### members.")
        windows = int(np.asarray(archive["num_windows"]).item())
        background = fire = pixels = 0
        for window_idx in range(windows):
            label = archive[f"label_{window_idx:03d}"]
            # read_window's rule: label == 0 is background, label > 0 is fire.
            background += int(np.count_nonzero(label == 0))
            fire += int(np.count_nonzero(label > 0))
            pixels += int(label.size)
    return windows, background, fire, pixels - background - fire


def count_labels(split_dir, workers=None):
    """Count the background and fire pixels of every window in a split.

    The fires are read in parallel threads, but the counts are whole numbers
    added up in fire order, so the result is the same on every run and machine.
    Only the targets are read, so even a large split takes seconds.

    Args:
        split_dir (str | Path): a split folder such as <data-root>/train.
        workers (int | None): reading threads (None picks up to MAX_COUNT_WORKERS).

    Returns:
        LabelCounts: the split's counts.

    Raises:
        RuntimeError: if the split holds no fires.
    """
    split_dir = Path(split_dir)
    sources = find_fire_sources(split_dir)
    if not sources:
        raise RuntimeError(f"No .npz archives or unzipped fire folders found in {split_dir}")
    workers = workers or min(MAX_COUNT_WORKERS, os.cpu_count() or 1, len(sources))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        per_fire = list(pool.map(count_fire_labels, sources))
    windows, background, fire, other = (sum(column) for column in zip(*per_fire))
    return LabelCounts(split_dir, len(sources), windows, background, fire, other,
                       tuple(fire_name(source) for source in sources))


# ===== Input normalization ===== #


# Threads reading fires at once. Each holds one decompressed window (about 42 MB
# for 6 days), so 4 threads stay well under a notebook's RAM.
MAX_NORMALIZATION_WORKERS = 4


@dataclass(frozen=True)
class ChannelMoments:
    """Hold running per-channel moments: how many values, their mean, and their squared deviations.

    Two sets of moments merge exactly (Chan et al.'s parallel update), so a
    split's moments are built window by window and fire by fire without ever
    holding more than one window.

    Attributes:
        count (int): values per channel so far.
        mean (np.ndarray): (C,) float64 per-channel mean.
        m2 (np.ndarray): (C,) float64 per-channel sum of squared deviations from the mean.
    """

    count: int
    mean: np.ndarray
    m2: np.ndarray

    @classmethod
    def empty(cls, channels):
        """Start moments that hold no value yet."""
        return cls(0, np.zeros(channels), np.zeros(channels))

    @classmethod
    def of_window(cls, window):
        """Measure one window's moments, one channel at a time in float64.

        Only one channel is widened to float64 at once (about 3 MB for 6 days),
        so measuring adds little to the window itself. The deviations are taken
        from this window's own mean, which keeps them small and exact.

        Args:
            window (np.ndarray): one window, shape (C, ...), any float type.

        Returns:
            ChannelMoments: the window's moments.
        """
        channels = window.shape[0]
        mean = np.empty(channels)
        m2 = np.empty(channels)
        for channel in range(channels):
            values = window[channel].astype(np.float64).ravel()
            mean[channel] = values.mean()
            m2[channel] = np.square(values - mean[channel]).sum()
        return cls(int(window[0].size), mean, m2)

    def merge(self, other):
        """Combine two sets of moments as if their values had been measured together.

        The result is exact up to rounding. The same inputs merged in the same
        order always give the same numbers.

        Args:
            other (ChannelMoments): moments of other values.

        Returns:
            ChannelMoments: the moments of both.
        """
        if other.count == 0:
            return self
        if self.count == 0:
            return other
        count = self.count + other.count
        delta = other.mean - self.mean
        mean = self.mean + delta * (other.count / count)
        m2 = self.m2 + other.m2 + np.square(delta) * (self.count * other.count / count)
        return ChannelMoments(count, mean, m2)

    @property
    def std(self):
        """Give the per-channel population standard deviation, like np.std (and compute_tessera_stats)."""
        return np.sqrt(self.m2 / self.count)


@dataclass(frozen=True)
class NormalizationStats:
    """Hold the per-channel statistics that standardize every window: (x - mean) / std.

    The angle and land-cover channels are measured too but never standardized
    (fire_dataset.NORMALIZE_SKIP), exactly as with the paper's numbers.

    Attributes:
        split_dir (Path): the split folder that was measured.
        fires (tuple[str, ...]): the fires measured ("p<id>"), sorted.
        windows (int): windows across those fires.
        values_per_channel (int): values each channel's statistics are over.
        mean (tuple[float, ...]): per-channel means (27).
        std (tuple[float, ...]): per-channel standard deviations (27); a channel
            that never varies gets 1.0, so it is only shifted.
    """

    split_dir: Path
    fires: tuple
    windows: int
    values_per_channel: int
    mean: tuple
    std: tuple

    def record(self):
        """Describe the statistics as the normalization block of window_config.json.

        Floats are written with full precision, so reading the block back gives
        exactly these numbers. The fire list lets training refuse statistics
        that miss a fire it would train on.

        Returns:
            dict: split, fires, windows, values per channel, channel names, mean, std.
        """
        return {
            "split": self.split_dir.name,
            "fires": list(self.fires),
            "windows": self.windows,
            "values_per_channel": self.values_per_channel,
            "channels": list(STORED_CHANNEL_NAMES[:len(self.mean)]),
            "mean": list(self.mean),
            "std": list(self.std),
        }

    @classmethod
    def from_record(cls, record, split_dir):
        """Read a normalization block of window_config.json back."""
        return cls(Path(split_dir), tuple(record["fires"]), int(record["windows"]),
                   int(record["values_per_channel"]), tuple(map(float, record["mean"])),
                   tuple(map(float, record["std"])))

    def summary(self):
        """Describe the statistics in one line for the console."""
        return (f"{self.split_dir}: per-channel mean and std of {self.windows:,} windows in "
                f"{len(self.fires):,} fires ({self.values_per_channel:,} values per channel)")


def fire_moments(source, stored_channels=N_STORED_CHANNELS):
    """Measure one fire's input windows, decompressing one window at a time.

    The fire is never loaded whole: each data_### member is read, measured
    and released before the next. Labels are not read.

    Args:
        source (Path): a p<id>.npz archive or unzipped p<id>/ folder.
        stored_channels (int): stored channels per window (27).

    Returns:
        tuple[int, ChannelMoments]: (windows, the fire's moments).

    Raises:
        RuntimeError: on an old-format archive without num_windows.
        ValueError: if a window has another channel count.
    """
    moments = ChannelMoments.empty(stored_channels)
    with open_fire(source) as archive:
        if "num_windows" not in archive.files:
            raise RuntimeError(f"{source} uses the old NPZ format. Expected num_windows plus data_### members.")
        windows = int(np.asarray(archive["num_windows"]).item())
        for window_idx in range(windows):
            window = archive[f"data_{window_idx:03d}"]
            if window.shape[0] != stored_channels:
                raise ValueError(f"{source}:data_{window_idx:03d} has {window.shape[0]} channels; "
                                 f"expected {stored_channels}")
            moments = moments.merge(ChannelMoments.of_window(window))
            del window
    return windows, moments


def split_normalization(split_dir, workers=None, stored_channels=N_STORED_CHANNELS):
    """Measure the per-channel mean and std of every input window in a split.

    Fires are read in parallel threads (decompression releases the GIL), one window per
    thread at a time. They are merged in fire order, so the result is the same
    on every run and for any thread count.

    Args:
        split_dir (str | Path): a split folder such as <data-root>/train.
        workers (int | None): reading threads (None picks up to MAX_NORMALIZATION_WORKERS).
        stored_channels (int): stored channels per window (27).

    Returns:
        NormalizationStats: the split's statistics.

    Raises:
        RuntimeError: if the split holds no fires or no windows.
    """
    split_dir = Path(split_dir)
    sources = find_fire_sources(split_dir)
    if not sources:
        raise RuntimeError(f"No .npz archives or unzipped fire folders found in {split_dir}")
    workers = workers or min(MAX_NORMALIZATION_WORKERS, os.cpu_count() or 1, len(sources))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        per_fire = list(pool.map(partial(fire_moments, stored_channels=stored_channels), sources))
    total = ChannelMoments.empty(stored_channels)
    for _, moments in per_fire:  # fire order, so the floating-point sums never change
        total = total.merge(moments)
    if total.count == 0:
        raise RuntimeError(f"NPZ files in {split_dir} contain no windows")
    std = total.std
    std = np.where(std > 0, std, 1.0)  # a constant channel is only shifted, never divided by 0
    return NormalizationStats(
        split_dir,
        tuple(fire_name(source) for source in sources),
        sum(windows for windows, _ in per_fire),
        total.count,
        tuple(float(value) for value in total.mean),
        tuple(float(value) for value in std),
    )


# ===== The data root's window_config.json ===== #


PIXEL_COUNTS = "pixel_counts"
NORMALIZATION = "normalization"
STATISTICS = (PIXEL_COUNTS, NORMALIZATION)
# The `python -m firepred.cli.dataset` subcommand that writes each block.
COMMANDS = {PIXEL_COUNTS: "count-pixels", NORMALIZATION: "compute-normalization"}


def window_config_path(data_root):
    """Give where a data root keeps its copy of window_config.json."""
    return Path(data_root) / WINDOW_CONFIG_FILE


def read_window_config_file(data_root):
    """Read a data root's window_config.json, or None when there is none."""
    path = window_config_path(data_root)
    return json.loads(path.read_text()) if path.is_file() else None


def write_window_config_file(data_root, config):
    """Write a data root's window_config.json atomically (beside it, then renamed), so it is never half written."""
    path = window_config_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.partial")
    temp.write_text(json.dumps(config, indent=2) + "\n")
    os.replace(temp, path)
    return path


def window_settings_of(config):
    """Give a window config without its statistics and notes: the settings two copies must share."""
    return {key: value for key, value in config.items() if key not in (*STATISTICS, "notes")}


def keep_statistics(config, previous):
    """Carry a previous copy's statistics into a fresh config that lacks them (its own statistics win)."""
    merged = dict(config)
    for key in STATISTICS:
        if key not in merged and previous and key in previous:
            merged[key] = previous[key]
    return merged


class MissingStatistics(RuntimeError):
    """Say that window_config.json cannot supply a statistic the run asked for.

    The file may be missing, lack the block or the split's entry, or hold a block
    computed without some fires now on disk. Training then falls back to the
    official TS-SatFire values and records why (training/experiment.py).
    """


def fire_coverage(record, split_dir, what, path):
    """Check that a statistics block covers every fire of a split.

    A fire the block lacks means the data changed after the statistics were
    computed. Fires the block has and the split lacks are returned instead.

    Args:
        record (dict): the block (it lists its "fires").
        split_dir (Path): the split being trained on.
        what (str): the block's key, for the message.
        path (Path): the window config, for the message.

    Returns:
        tuple[str, ...]: fires the block has but the split lacks (only worth a note).

    Raises:
        MissingStatistics: if the split has fires the block was not computed from.
    """
    present = {fire_name(source) for source in find_fire_sources(split_dir)}
    recorded = set(record["fires"])
    missing = sorted(present - recorded)
    if missing:
        raise MissingStatistics(
            f"{path}: {what} were computed without {len(missing)} fires now in {split_dir} "
            f"(e.g. {missing[:5]}); run `python -m firepred.cli.dataset {COMMANDS[what]} "
            f"--data-root {path.parent}` again"
        )
    return tuple(sorted(recorded - present))


@dataclass(frozen=True)
class WindowConfig:
    """Hold a data root's window_config.json: its window settings and data-grab statistics.

    Training reads one before anything else. Its methods check the settings
    and hand out the statistics.

    Attributes:
        path (Path): the file.
        values (dict): its contents ({} when there is no file).
        found (bool): whether the file exists.
    """

    path: Path
    values: dict
    found: bool = True

    @classmethod
    def read(cls, data_root):
        """Read a data root's window_config.json, or stand in for a missing one.

        A missing file gives an empty config: nothing is checked against it and
        every statistic it is asked for raises MissingStatistics. Nothing is
        checked yet.

        Args:
            data_root (str | Path): the data root.

        Returns:
            WindowConfig: the file, or an empty one with found=False.
        """
        values = read_window_config_file(data_root)
        return cls(window_config_path(data_root), values or {}, values is not None)

    def check_windows(self, ts_length, ts_interval=None):
        """Refuse a run whose -ts/-it differ from the recorded window settings (unrecorded ones are not checked)."""
        for key, flag, asked in (("ts_length", "-ts", ts_length), ("ts_interval", "-it", ts_interval)):
            recorded = self.values.get(key)
            if asked is not None and recorded is not None and int(recorded) != int(asked):
                raise ValueError(f"{flag} {asked} does not match the data: {self.path} records {key} = {recorded}")
        return self

    def block(self, key):
        """Give one statistics block, or raise MissingStatistics naming the command that writes it.

        The blocks are added by data-grab commands after download. Without the
        file, or without the block, training falls back to the official values.

        Args:
            key (str): PIXEL_COUNTS or NORMALIZATION.

        Returns:
            dict: the block.

        Raises:
            MissingStatistics: if there is no file, or it has no such block.
        """
        if not self.found:
            raise MissingStatistics(f"No {WINDOW_CONFIG_FILE} in {self.path.parent}")
        if key not in self.values:
            raise MissingStatistics(
                f"{self.path} has no {key}. Run `python -m firepred.cli.dataset {COMMANDS[key]} "
                f"--data-root {self.path.parent}` first."
            )
        return self.values[key]

    def label_counts(self, split_dir):
        """Give a split's recorded pixel counts, checked against the fires on disk.

        The split folder's name (train, val or test) picks the entry. Counts that
        miss a fire on disk are refused.

        Args:
            split_dir (str | Path): the split, e.g. <data-root>/train; its name picks the entry.

        Returns:
            tuple[LabelCounts, tuple[str, ...]]: (the counts, recorded fires the split lacks).

        Raises:
            MissingStatistics: if there are no counts for the split, or they miss one of its fires.
        """
        split_dir = Path(split_dir)
        counts = self.block(PIXEL_COUNTS)
        if split_dir.name not in counts:
            raise MissingStatistics(
                f"{self.path} has no {PIXEL_COUNTS} for {split_dir.name}. Run `python -m firepred.cli.dataset "
                f"{COMMANDS[PIXEL_COUNTS]} --data-root {self.path.parent}` first."
            )
        record = counts[split_dir.name]
        extra = fire_coverage(record, split_dir, PIXEL_COUNTS, self.path)
        return LabelCounts.from_record(record, split_dir), extra

    def normalization(self, split_dir):
        """Give the recorded input statistics, refused if they miss a fire of the split, plus fires the split lacks."""
        record = self.block(NORMALIZATION)
        extra = fire_coverage(record, Path(split_dir), NORMALIZATION, self.path)
        return NormalizationStats.from_record(record, split_dir), extra
