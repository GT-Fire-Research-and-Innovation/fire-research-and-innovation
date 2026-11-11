"""MASTER: add a zone file to every fire archive of a data root, built from the archives alone.

Level 1 (data), master. For each window archive it reads the targets (and the
night heat of the first window), tracks the fires (data/fire_tracks.py) and
writes zones/p<id>.npz (data/fire_zones.py). The archives are never changed,
so zones can be added to an existing dataset and rebuilt with another
separation distance in minutes. Like the other masters, it prints its progress.

    python -m firepred.cli.dataset zones --data-root ./data [--scale-km 2.25]
"""

from dataclasses import fields
from pathlib import Path

import numpy as np

from firepred.config.dataset import (
    DAILY_FORMAT_VERSION,
    DATA_DIR,
    FIRE_SEPARATION_KM,
    SEED_NIGHT_I4_KELVIN,
    SPLITS,
    TS_INTERVAL,
    TS_LENGTH,
    ZONE_DIR,
)
from firepred.data.archive import find_fire_sources, fire_name, read_fire_labels, read_seed_heat, read_zone_file
from firepred.data.archive import write_zone_file
from firepred.data.fire_tracks import burn_history, split_settings, track_fire
from firepred.data.fire_zones import zone_file
from firepred.data.stats import WindowConfig


def build_fire_zone_file(source, settings, length=TS_LENGTH, seed_kelvin=SEED_NIGHT_I4_KELVIN):
    """Build one fire archive's zones: read its targets and seed heat, track its fires, map their zones.

    Only the targets and the first window's night I4 are read. Nothing is
    written; build_zones writes the result.

    Args:
        source (str | Path): a p<id>.npz archive or unzipped p<id>/ folder.
        settings (SplitSettings): the thresholds (fire_tracks.split_settings).
        length (int): input days per window, the day of window 0's target.
        seed_kelvin (float): the night I4 that marks a seed before the first target.

    Returns:
        ZoneFile: the archive's zones.
    """
    labels, label_sel = read_fire_labels(source, length)
    history = burn_history(labels, read_seed_heat(source, length=length), label_sel, length, seed_kelvin)
    return zone_file(track_fire(history, settings), history, settings)


def require_daily_windows(data_root, length=TS_LENGTH, interval=TS_INTERVAL):
    """Refuse archives whose targets do not date every burn after the first target day.

    Burn days are rebuilt from the targets, so a window must start every day
    (interval 1), and the window length says which day each target is. The data
    root's window_config.json, when there is one, must record the same settings.
    Daily archives (format 3) store every day, so they are read at interval 1
    whatever -it they record; only the length must match.

    Args:
        data_root (str | Path): the data root.
        length (int): input days per window.
        interval (int): days between window starts.

    Raises:
        ValueError: on an interval other than 1, or settings window_config.json contradicts.
    """
    config = WindowConfig.read(data_root)
    if config.values.get("archive_format_version") == DAILY_FORMAT_VERSION:
        config.check_windows(length)
        return
    if interval != 1:
        raise ValueError(f"Zones need a window starting every day (-it 1); got -it {interval}")
    config.check_windows(length, interval)


def require_same_settings(path, settings):
    """Refuse to add zones to a zones folder whose existing file was built with other settings.

    Zone files built with different settings (separation distance, or a window
    length that caps memory_days differently) must never mix, just as window
    repos never mix window settings. Other settings belong in another zones folder.

    Args:
        path (Path): an existing zone file.
        settings (SplitSettings): the settings of this run.

    Raises:
        ValueError: if the file's settings differ.
    """
    existing = read_zone_file(path).settings
    if existing != settings:
        changed = ", ".join(f"{field.name} {getattr(existing, field.name)} (now {getattr(settings, field.name)})"
                            for field in fields(settings) if getattr(existing, field.name) != getattr(settings, field.name))
        raise ValueError(f"{path} was built with other settings: {changed}. "
                         "Pass --overwrite, or another --zones-root for these settings.")


def describe_zone_file(name, zones):
    """Summarize one zone file in one line: windows, fires, splits, merges and unowned new pixels.

    The last number is the share of new target pixels that no fire reaches:
    new ignitions the inputs could not predict. Those pixels are left out of
    every fire's loss.

    Args:
        name (str): the fire, p<id>.
        zones (ZoneFile): its zones.

    Returns:
        str: the summary line.
    """
    events = zones.events
    splits = len({event.parent_id for event in events if event.kind == "split"})
    merges = sum(1 for event in events if event.merged_into)
    new_pixels = np.count_nonzero(zones.target_owners)
    unowned = np.count_nonzero(zones.target_owners < 0)
    share = unowned / new_pixels if new_pixels else 0.0
    return (f"{name}: {len(zones.zones)} windows, {len(events)} fires ({splits} splits, {merges} merges), "
            f"{share:.1%} of new target pixels belong to no fire")


def build_zones(data_root=DATA_DIR, zones_root=None, splits=SPLITS, scale_km=FIRE_SEPARATION_KM,
                length=TS_LENGTH, interval=TS_INTERVAL, overwrite=False):
    """Write a zone file for every fire archive of the chosen splits.

    Zone files go to zones_root (default <data_root>/zones) as p<id>.npz, the
    same names as the archives. Fires that already have one are skipped unless
    overwrite is set, and an existing file built with another distance stops
    the run.

    Args:
        data_root (str | Path): the data root with train/, val/ and test/.
        zones_root (str | Path | None): where the zone files go.
        splits (tuple[str, ...]): the splits to process.
        scale_km (float): the separation distance (FIRE_SEPARATION_KM).
        length (int): input days per window the archives were built with.
        interval (int): days between window starts; must be 1 for format-2 archives
            (daily archives are read at 1 whatever it is).
        overwrite (bool): rebuild zone files that exist.

    Returns:
        tuple[str, ...]: one summary line per fire.
    """
    data_root = Path(data_root)
    zones_root = Path(zones_root) if zones_root else data_root / ZONE_DIR
    require_daily_windows(data_root, length, interval)
    settings = split_settings(scale_km, length)
    zones_root.mkdir(parents=True, exist_ok=True)
    print(f"Zones at {settings.scale_km:g} km: link {settings.link_pixels:g} px, "
          f"memory {settings.memory_days} days -> {zones_root}")
    notes = []
    for split in splits:
        for source in find_fire_sources(data_root / split):
            name = fire_name(source)
            path = zones_root / f"{name}.npz"
            if path.exists() and not overwrite:
                require_same_settings(path, settings)
                note = f"{name}: zone file exists; skipped"
            else:
                zones = build_fire_zone_file(source, settings, length)
                write_zone_file(path, zones)
                note = describe_zone_file(name, zones)
            print(note)
            notes.append(note)
    return tuple(notes)
