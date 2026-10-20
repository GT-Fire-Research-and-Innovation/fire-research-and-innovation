"""How rare fire is in a split: exact background and fire pixel counts of its targets.

Level 1 (data). Used by training/experiment.py (--ce-weight auto) and
cli/check.py (`check labels`).

Next-day fire growth covers well under 1 % of the pixels, so the Dice + CE
loss weighs fire pixels up by the ratio of background to fire pixels. The
paper's (1.0, 446.7836) is that ratio for its training data; a dataset built
with other windows, years or label rules has its own. count_labels finds it
for any data root.

The counts follow read_window's rule exactly (label == 0 is background,
label > 0 is fire), and only the label_### members are read, never the input
windows. They are whole numbers, so the result does not depend on the order
or the threads that read the fires: the same data always gives the same
ratio. Augmentation (flips and 90-degree turns) only moves pixels around, so
the counts are also exactly what the loss sees in every epoch.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from firepred.data.archive import find_fire_sources, open_fire

# Threads reading fires at once: zlib releases the GIL, so a few threads help on slow disks.
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
    """

    split_dir: Path
    fires: int
    windows: int
    background: int
    fire: int
    other: int = 0

    @property
    def fire_fraction(self):
        """Give the share of labelled pixels that are fire.

        Only background and fire pixels are counted. It is 0 when there are none.

        Returns:
            float: fire / (background + fire).
        """
        labelled = self.background + self.fire
        return self.fire / labelled if labelled else 0.0

    @property
    def background_per_fire(self):
        """Give the ratio of background to fire pixels: the fire class weight that balances the two.

        With class weights (1, ratio), all fire pixels together weigh as much as
        all background pixels. Both counts are whole numbers, so the division is
        exactly repeatable.

        Returns:
            float: background / fire.

        Raises:
            ValueError: if the split has no fire pixel.
        """
        if self.fire == 0:
            raise ValueError(f"{self.split_dir} has no fire pixels; the background-to-fire ratio is undefined")
        return self.background / self.fire

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
    return LabelCounts(split_dir, len(sources), windows, background, fire, other)
