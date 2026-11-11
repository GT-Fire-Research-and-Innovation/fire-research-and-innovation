"""Turn one archive's fire tracks into zones: which fire owns each area and each target pixel, per window.

Level 1 (data). Pure functions: no files, no printing.

Window w uses input days w .. w + T - 1 and predicts day w + T, where T is the
window length (the history's first_label_day). Its zones come from the owner
map at the end of its last input day, so they use no later day and are built
the same way at inference:

    zones          a pixel within link distance of a fire's recent pixels belongs to
                   the nearest such fire; every other pixel is 0
    target_owners  (training only) each new pixel of the target day belongs to the fire
                   its group reaches at link distance, chaining through other new pixels
                   so a fast run stays with its fire; a group no fire reaches is -1, a new
                   ignition nothing in the inputs could predict

A training sample is one fire in one window (fire_sample): its target is that
fire's new pixels, and its loss skips the other fires' zones and every new
pixel that is not its own. The inputs are never changed, so training
(--zones-dir) runs the model once per window and splits it into its fires in the
loss (training/objectives.fire_samples, the same rule for a batch).
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

from firepred.config.dataset import ZONE_FORMAT_VERSION
from firepred.data.fire_records import ZoneFile
from firepred.data.fire_tracks import link_clusters, nearest_owner, recent_pixels


def window_count(history):
    """Count the windows of the archive a burn history came from."""
    return history.last_day - history.first_label_day + 1


def last_input_day(history, window):
    """Give the last input day of a window, the day before its target."""
    return window + history.first_label_day - 1


def recent_fire_pixels(tracks, history, day, settings):
    """Mark the recent pixels of every fire on a day with that fire's id.

    Recent means first seen within the last memory_days, as in tracking.
    Unowned recent pixels (groups not yet confirmed as fires) are 0.

    Args:
        tracks (FireTracks): the archive's tracks.
        history (BurnHistory): the burn history.
        day (int): the day.
        settings (SplitSettings): the thresholds.

    Returns:
        np.ndarray: (H, W) int32, fire ids with 0 elsewhere.
    """
    return np.where(recent_pixels(history, day, settings.memory_days), tracks.owners[day], 0)


def window_zones(tracks, history, window, settings):
    """Map which fire owns each area of a window, from its input days only.

    Every pixel within link distance of a fire's recent pixels belongs to the
    nearest fire, so fires never share a pixel. Pixels farther than that from
    every fire are 0.

    Args:
        tracks (FireTracks): the archive's tracks.
        history (BurnHistory): the burn history.
        window (int): the window number.
        settings (SplitSettings): the thresholds.

    Returns:
        np.ndarray: (H, W) int16 zones.
    """
    fires = recent_fire_pixels(tracks, history, last_input_day(history, window), settings)
    zones = np.zeros(fires.shape, dtype=np.int16)
    if not fires.any():
        return zones
    distance, (rows, cols) = distance_transform_edt(fires == 0, return_indices=True)
    within_reach = distance <= settings.link_pixels
    zones[within_reach] = fires[rows[within_reach], cols[within_reach]]
    return zones


def window_target_owners(tracks, history, window, settings):
    """Say which fire each new pixel of a window's target day belongs to.

    The target's new pixels and the fires' recent pixels are grouped at link
    distance. New pixels in a group with a fire go to the nearest fire of that
    group, and new pixels in a group without one are -1.

    Args:
        tracks (FireTracks): the archive's tracks.
        history (BurnHistory): the burn history.
        window (int): the window number.
        settings (SplitSettings): the thresholds.

    Returns:
        np.ndarray: (H, W) int16; fire id per new pixel, -1 for no fire, 0 off target.
    """
    day = last_input_day(history, window)
    fires = recent_fire_pixels(tracks, history, day, settings)
    new_pixels = history.burn_day == day + 1
    owners = np.zeros(fires.shape, dtype=np.int16)
    groups = link_clusters(new_pixels | (fires > 0), settings.link_pixels)
    for number in range(1, groups.max() + 1):
        group = groups == number
        group_new = group & new_pixels
        if not group_new.any():
            continue
        if (fires[group] > 0).any():
            owners[group_new] = nearest_owner(fires, group)[group_new]
        else:
            owners[group_new] = -1
    return owners


def zone_file(tracks, history, settings):
    """Bundle the zones and target owners of every window with the fires and settings that made them.

    The result is everything a zone file stores. archive.write_zone_file
    writes it next to the window archive.

    Args:
        tracks (FireTracks): the archive's tracks.
        history (BurnHistory): the burn history.
        settings (SplitSettings): the thresholds.

    Returns:
        ZoneFile: zones, target owners, fires and settings.
    """
    windows = range(window_count(history))
    empty = np.zeros((0, *history.burn_day.shape), dtype=np.int16)
    zones = [window_zones(tracks, history, window, settings) for window in windows]
    target_owners = [window_target_owners(tracks, history, window, settings) for window in windows]
    return ZoneFile(
        zones=np.stack(zones) if zones else empty,
        target_owners=np.stack(target_owners) if target_owners else empty,
        events=tracks.events,
        settings=settings,
        label_sel=history.label_sel,
        first_label_day=history.first_label_day,
        format_version=ZONE_FORMAT_VERSION,
    )


def fires_in_window(zones):
    """List the fires that own part of a window, in id order: one training sample each."""
    return tuple(int(fire_id) for fire_id in np.unique(zones) if fire_id > 0)


def fire_sample(zones, target_owners, fire_id):
    """Build one fire's target and loss mask for one window.

    The target is the fire's own new pixels. The loss skips the other fires'
    zones and every new pixel that is not this fire's (other fires' growth and
    new ignitions), so two samples of one window never teach opposite things.

    Args:
        zones (np.ndarray): (H, W) zones of the window.
        target_owners (np.ndarray): (H, W) target owners of the window.
        fire_id (int): the fire.

    Returns:
        tuple[np.ndarray, np.ndarray]: (target, loss mask), both (H, W) bool.
    """
    target = target_owners == fire_id
    other_zones = (zones > 0) & (zones != fire_id)
    other_new_pixels = (target_owners != 0) & (target_owners != fire_id)
    return target, ~(other_zones | other_new_pixels)
