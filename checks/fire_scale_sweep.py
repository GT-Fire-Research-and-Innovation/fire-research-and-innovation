"""Sweep the fire separation distance over a data root's archives and print how fire tracking reacts.

    python checks/fire_scale_sweep.py --data-root ./data
    python checks/fire_scale_sweep.py --data-root ./data --splits test --scales 1.5 2.25 3 --perturb

Only the targets and the first window's night heat are read, so a whole data
root takes minutes. For each distance the table shows how many fires, new
ignitions, splits and merges tracking finds, how often a split merges back
soon after (flicker), and the share of new target pixels no fire owns. With
--perturb it also shows how stable the fires stay when a day is dropped, 1%
speckle is added or the map shifts by one pixel. Pick the distance where the
fire count has levelled off, flicker is near 0 and stability is high.
"""

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firepred.config.dataset import SPLITS, TS_LENGTH, VIIRS_PIXEL_KM  # noqa: E402
from firepred.data.archive import find_fire_sources, fire_name, read_fire_labels, read_seed_heat  # noqa: E402
from firepred.data.fire_tracks import burn_history, pixel_days, split_settings, track_fire  # noqa: E402
from firepred.data.fire_zones import zone_file  # noqa: E402

DEFAULT_SCALES_KM = tuple(round(VIIRS_PIXEL_KM * pixels, 3) for pixels in range(2, 17))  # 0.75 .. 6 km
PERTURBATIONS = ("drop_day", "speckle", "shift")


@dataclass(frozen=True)
class ScaleReport:
    """Hold what fire tracking found over all archives at one separation distance.

    Counts are summed over archives. Stability is nan when it was not measured.

    Attributes:
        scale_km (float): the separation distance.
        fires (int): every fire found.
        new_fires (int): ignitions after the first one of each archive.
        splits (int): fires that split.
        merges (int): fires absorbed by an older fire.
        flip_flops (int): split branches that merged back into a sibling within memory_days.
        unowned_target_share (float): share of new target pixels no fire owns.
        stability (float): mean adjusted Rand index of the fires under small damage to the data.
    """

    scale_km: float
    fires: int
    new_fires: int
    splits: int
    merges: int
    flip_flops: int
    unowned_target_share: float
    stability: float


def load_histories(data_root, splits, length=TS_LENGTH):
    """Read every archive's burn history once, so each distance reuses them.

    Args:
        data_root (str | Path): the data root.
        splits (tuple[str, ...]): the splits to read.
        length (int): input days per window.

    Returns:
        list[tuple[str, BurnHistory]]: (p<id>, history) per archive.
    """
    histories = []
    for split in splits:
        for source in find_fire_sources(Path(data_root) / split):
            labels, label_sel = read_fire_labels(source)
            histories.append((fire_name(source), burn_history(labels, read_seed_heat(source), label_sel, length)))
    return histories


def count_flip_flops(events, settings):
    """Count split branches that merged into a sibling within memory_days of the split.

    A split that undoes itself that quickly is flicker, not two fires.
    """
    parents = {event.event_id: event.parent_id for event in events}
    return sum(1 for event in events
               if event.kind == "split" and event.merged_into
               and parents.get(event.merged_into) == event.parent_id
               and event.end_day - event.start_day <= settings.memory_days)


def pair_count(counts):
    """Count the pairs that can be drawn within each count, summed."""
    counts = np.asarray(counts, dtype=np.int64)
    return int((counts * (counts - 1) // 2).sum())


def adjusted_rand_index(labels_a, labels_b):
    """Measure how alike two labelings of the same pixels are: 1 identical, about 0 unrelated.

    Label values do not matter, only which pixels share a label. Fewer than
    two pixels count as identical.

    Args:
        labels_a (np.ndarray): (pixels,) labels.
        labels_b (np.ndarray): (pixels,) labels of the same pixels.

    Returns:
        float: the adjusted Rand index.
    """
    count = len(labels_a)
    if count < 2:
        return 1.0
    _, rows = np.unique(labels_a, return_inverse=True)
    _, cols = np.unique(labels_b, return_inverse=True)
    table = np.zeros((rows.max() + 1, cols.max() + 1), dtype=np.int64)
    np.add.at(table, (rows, cols), 1)
    together = pair_count(table)
    row_pairs, col_pairs = pair_count(table.sum(axis=1)), pair_count(table.sum(axis=0))
    expected = row_pairs * col_pairs / (count * (count - 1) // 2)
    best = (row_pairs + col_pairs) / 2
    return 1.0 if best == expected else float((together - expected) / (best - expected))


def perturb_history(history, kind, rng):
    """Damage a burn history slightly, the way real data goes wrong.

    "drop_day" loses one target day (a missed overpass), "speckle" adds 1%
    stray pixels on random target days, "shift" moves everything one pixel
    down. A history without target days comes back unchanged.

    Args:
        history (BurnHistory): the history.
        kind (str): one of PERTURBATIONS.
        rng (np.random.Generator): where random draws come from.

    Returns:
        BurnHistory: the damaged copy.
    """
    if history.last_day < history.first_label_day:
        return history
    burn_day = history.burn_day.copy()
    if kind == "drop_day":
        burn_day[burn_day == rng.integers(history.first_label_day, history.last_day + 1)] = -1
        return replace(history, burn_day=burn_day)
    if kind == "speckle":
        empty = np.argwhere(pixel_days(history) < 0)
        count = min(len(empty), max(1, round(0.01 * np.count_nonzero(burn_day >= 0))))
        chosen = empty[rng.choice(len(empty), size=count, replace=False)]
        burn_day[chosen[:, 0], chosen[:, 1]] = rng.integers(history.first_label_day, history.last_day + 1, size=count)
        return replace(history, burn_day=burn_day)
    return replace(history, burn_day=np.roll(burn_day, 1, axis=0), seed_day=np.roll(history.seed_day, 1, axis=0))


def stability_score(history, settings, rng):
    """Measure how much the fires change when the data is damaged slightly, averaged over PERTURBATIONS.

    The final owner maps are compared on the pixels seen in both runs. A
    shifted run is shifted back first.

    Args:
        history (BurnHistory): the history.
        settings (SplitSettings): the thresholds.
        rng (np.random.Generator): where random draws come from.

    Returns:
        float: the mean adjusted Rand index (1 = unchanged).
    """
    original = track_fire(history, settings).owners
    if len(original) == 0:
        return 1.0
    scores = []
    for kind in PERTURBATIONS:
        damaged = perturb_history(history, kind, rng)
        owners = track_fire(damaged, settings).owners[-1]
        seen = (pixel_days(history) >= 0) & (pixel_days(damaged) >= 0)
        if kind == "shift":
            owners = np.roll(owners, -1, axis=0)
            seen = (pixel_days(history) >= 0) & np.roll(pixel_days(damaged) >= 0, -1, axis=0)
        scores.append(adjusted_rand_index(original[-1][seen], owners[seen]))
    return float(np.mean(scores))


def scale_report(histories, scale_km, perturb=False, seed=0):
    """Track every archive at one separation distance and sum what was found.

    Args:
        histories (list[tuple[str, BurnHistory]]): load_histories' result.
        scale_km (float): the separation distance.
        perturb (bool): also measure stability (four tracking runs per archive instead of one).
        seed (int): random seed for the damage.

    Returns:
        ScaleReport: the counts.
    """
    settings = split_settings(scale_km)
    rng = np.random.default_rng(seed)
    fires = new_fires = splits = merges = flip_flops = owned_new = unowned_new = 0
    stability = []
    for _, history in histories:
        tracks = track_fire(history, settings)
        events = tracks.events
        ignitions = sum(1 for event in events if event.kind == "ignition")
        fires += len(events)
        new_fires += max(0, ignitions - 1)
        splits += len({event.parent_id for event in events if event.kind == "split"})
        merges += sum(1 for event in events if event.merged_into)
        flip_flops += count_flip_flops(events, settings)
        owners = zone_file(tracks, history, settings).target_owners
        owned_new += np.count_nonzero(owners > 0)
        unowned_new += np.count_nonzero(owners < 0)
        if perturb:
            stability.append(stability_score(history, settings, rng))
    total_new = owned_new + unowned_new
    return ScaleReport(settings.scale_km, fires, new_fires, splits, merges, flip_flops,
                       unowned_new / total_new if total_new else 0.0,
                       float(np.mean(stability)) if stability else float("nan"))


def sweep_scales(histories, scales_km, perturb=False, seed=0):
    """Run scale_report for every distance, smallest first."""
    return [scale_report(histories, scale_km, perturb, seed) for scale_km in sorted(scales_km)]


def format_sweep(reports):
    """Lay the reports out as a text table, one row per distance.

    Args:
        reports (list[ScaleReport]): sweep_scales' result.

    Returns:
        str: the table.
    """
    lines = [f"{'km':>6} {'pixels':>6} {'fires':>6} {'new':>5} {'splits':>6} {'merges':>6} "
             f"{'flicker':>7} {'unowned':>8} {'stable':>7}"]
    for report in reports:
        lines.append(f"{report.scale_km:6.3f} {report.scale_km / VIIRS_PIXEL_KM:6.1f} {report.fires:6d} "
                     f"{report.new_fires:5d} {report.splits:6d} {report.merges:6d} {report.flip_flops:7d} "
                     f"{report.unowned_target_share:8.1%} {report.stability:7.3f}")
    return "\n".join(lines)


def main(argv=None):
    """Parse the flags, read the archives once, sweep the distances and print the table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--scales", nargs="+", type=float, default=list(DEFAULT_SCALES_KM), help="Distances in km")
    parser.add_argument("-ts", "--length", type=int, default=TS_LENGTH, help="Input days per window")
    parser.add_argument("--perturb", action="store_true", help="Also measure stability (slower)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    histories = load_histories(args.data_root, tuple(args.splits), args.length)
    print(f"{len(histories)} archives from {args.data_root} ({', '.join(args.splits)})")
    print(format_sweep(sweep_scales(histories, args.scales, args.perturb, args.seed)))


if __name__ == "__main__":
    main()
