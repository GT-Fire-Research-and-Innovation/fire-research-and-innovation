"""The records shared by fire tracking, zone files and their readers: settings, fire events, zone files.

Level 1 (data). Data only: frozen dataclasses with no methods and nothing
imported beyond numpy, so archive.py can read and write zone files without
loading scipy. The functions that fill them live in fire_tracks.py
(tracking) and fire_zones.py (zones); archive.py stores them.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitSettings:
    """Hold every threshold of fire tracking, all derived from one separation distance.

    fire_tracks.split_settings computes them, so changing scale_km changes them
    all together. Distances are in pixels, times in days.

    Attributes:
        scale_km (float): the separation distance everything comes from.
        link_pixels (float): pixels at most this far apart belong to the same fire.
        merge_pixels (float): two fires merge only when their fronts come this close.
        memory_days (int): how long a quiet pixel still connects a fire (clouds, missed passes).
        split_days (int): how many recent days count as a fire's front.
        min_pixels (int): fewest burned pixels for a new fire or a split branch.
        min_growth_days (int): days a new fire must grow on, and a split must hold, before it counts.
    """

    scale_km: float
    link_pixels: float
    merge_pixels: float
    memory_days: int
    split_days: int
    min_pixels: int
    min_growth_days: int


@dataclass(frozen=True)
class FireEvent:
    """Describe one fire found by tracking: when it lived and where it came from.

    A fire starts as an ignition or as one branch of a split. It ends when it
    stops growing, when it splits (its branches carry on), or when it merges
    into an older fire.

    Attributes:
        event_id (int): the fire's number within its archive, from 1.
        kind (str): "ignition" or "split".
        parent_id (int): the fire it split from (0 for an ignition).
        start_day (int): the first day of its pixels.
        end_day (int): the day it ended, or -1 while it still burns.
        merged_into (int): the fire that absorbed it (0 if it was never merged).
    """

    event_id: int
    kind: str
    parent_id: int
    start_day: int
    end_day: int
    merged_into: int


@dataclass(frozen=True)
class ZoneFile:
    """Hold one fire archive's zones: which fire owns each area and each target pixel, per window.

    zones come from the input days only, so they can be built at inference.
    target_owners also use the target day and are for training only.

    Attributes:
        zones (np.ndarray): (windows, H, W) int16; fire id within reach of that fire, 0 elsewhere.
        target_owners (np.ndarray): (windows, H, W) int16; fire id of each target pixel,
            -1 for a target pixel no fire reaches, 0 off target.
        events (tuple[FireEvent, ...]): every fire found, in id order.
        settings (SplitSettings): the thresholds used.
        label_sel (int): the label rule of the source archive (-1 if unrecorded).
        first_label_day (int): the day of window 0's target (the window length).
        format_version (int): ZONE_FORMAT_VERSION when written.
    """

    zones: np.ndarray
    target_owners: np.ndarray
    events: tuple
    settings: SplitSettings
    label_sel: int
    first_label_day: int
    format_version: int
