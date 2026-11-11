"""Follow every fire of one archive day by day: new ignitions, merges and splits, all set by one distance.

Level 1 (data). Pure functions over numpy arrays: no files, no printing, and
arguments are never modified (arrays are copied before they change).

The input is a burn history: the day each pixel first burned, rebuilt from a
window archive's targets (with TS_INTERVAL = 1, target w holds exactly the
pixels first burned on day w + TS_LENGTH), plus "seed" pixels for the days
before the first target, taken from night I4 heat. Seeds count like burned
pixels, so a fire already burning before its first target is tracked from
day 0, while scattered warm pixels never group into enough pixels to start one.

Tracking is a fold over days, state = advance_one_day(state, ...). Each day:

    cluster   group recent pixels; pixels within link distance belong together
    confirm   a group with no fire, enough pixels and growth on 2+ days -> new fire
    claim     unowned pixels in a fire's group -> the nearest fire
    merge     fires whose fronts come within merge distance -> the older absorbs the younger
    split     a fire whose front breaks into 2+ large parts for 2 days -> one new fire per part
    end       a fire with no growth for memory_days ends

Every threshold comes from one distance (split_settings): gaps shorter than it
are noise to bridge (speckle, clouds, broken front lines), longer gaps
separate fires. Ties go to the older fire, then the lower id, then the first
pixel in row-major order, so the same input always gives the same fires.
"""

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from firepred.config.dataset import (
    FAST_SPREAD_KM_PER_DAY,
    FIRE_SEPARATION_KM,
    MIN_GROWTH_DAYS,
    SEED_NIGHT_I4_KELVIN,
    SLOW_SPREAD_KM_PER_DAY,
    TS_LENGTH,
    VIIRS_PIXEL_KM,
)
from firepred.data.fire_records import FireEvent, SplitSettings


# ----- Records ----- #


@dataclass(frozen=True)
class BurnHistory:
    """Hold the day each pixel of one fire archive first burned, and the heat seeds before that.

    Days count from the archive's first input day (day 0). Burn days start at
    first_label_day, the day of the first target; seed days come before it.

    Attributes:
        burn_day (np.ndarray): (H, W) int16, the day the pixel first burned, -1 if never.
        seed_day (np.ndarray): (H, W) int16, the first hot day before first_label_day, -1 if none.
        first_label_day (int): the day of window 0's target (the window length).
        last_day (int): the day of the last target.
        label_sel (int): the label rule that built the targets (-1 if unknown).
    """

    burn_day: np.ndarray
    seed_day: np.ndarray
    first_label_day: int
    last_day: int
    label_sel: int


@dataclass(frozen=True)
class DayClusters:
    """Hold one day's groups of recent pixels at the two tracking distances.

    Groups are numbered 1, 2, ... by their first pixel in row-major order, and
    0 marks pixels outside every group.

    Attributes:
        linked (np.ndarray): (H, W) int32, groups of the last memory_days at link distance.
        close (np.ndarray): (H, W) int32, groups of the last split_days (the fronts) at merge distance.
    """

    linked: np.ndarray
    close: np.ndarray


@dataclass(frozen=True)
class TrackState:
    """Hold everything tracking knows at the end of one day.

    Each step of advance_one_day returns a new state. Nothing is changed in place.

    Attributes:
        day (int): the day just processed (-1 before the first).
        owner (np.ndarray): (H, W) int32, the fire each pixel belongs to, 0 for none.
        events (tuple[FireEvent, ...]): every fire so far, in id order.
        pending_splits (tuple[tuple[int, int], ...]): (fire id, days its front has been split so far).
        next_id (int): the id the next new fire gets.
    """

    day: int
    owner: np.ndarray
    events: tuple
    pending_splits: tuple
    next_id: int


@dataclass(frozen=True)
class FireTracks:
    """Hold the result of tracking one archive: who owned each pixel on each day, and every fire.

    owners[t] is the owner map at the end of day t. It is built from days up
    to t only, so the zones of a window never see its future.

    Attributes:
        owners (np.ndarray): (days, H, W) int32, owner maps for days 0 .. last_day.
        events (tuple[FireEvent, ...]): every fire found, in id order.
    """

    owners: np.ndarray
    events: tuple


# ----- Settings and history ----- #


def split_settings(scale_km=FIRE_SEPARATION_KM):
    """Derive every tracking threshold from one separation distance.

    Smaller distances split more and clean less; larger ones clean more and
    split less. Distances become 375 m pixels, and the slow and fast spread
    rates turn them into days.

    Args:
        scale_km (float): the separation distance in km.

    Returns:
        SplitSettings: the thresholds.

    Raises:
        ValueError: if scale_km is not positive.
    """
    if scale_km <= 0:
        raise ValueError(f"scale_km must be positive, got {scale_km}")
    # Rounded first, so 6.0000000001 pixels stays 6 when rounded up below.
    pixels = round(scale_km / VIIRS_PIXEL_KM, 6)
    return SplitSettings(
        scale_km=float(scale_km),
        link_pixels=pixels,
        merge_pixels=pixels / 2,
        memory_days=max(1, math.floor(scale_km / SLOW_SPREAD_KM_PER_DAY + 0.5)),  # rounded half up
        split_days=max(1, math.ceil(round(scale_km / FAST_SPREAD_KM_PER_DAY, 6))),
        min_pixels=max(1, math.ceil(pixels)),
        min_growth_days=MIN_GROWTH_DAYS,
    )


def burn_history(labels, night_heat, label_sel=-1, first_label_day=TS_LENGTH, seed_kelvin=SEED_NIGHT_I4_KELVIN):
    """Rebuild the day each pixel first burned from a fire's targets, plus heat seeds before them.

    Target w holds the pixels first burned on day first_label_day + w, so the
    targets together date every burn from that day on. Earlier days are only
    visible as heat: a pixel whose night I4 passes seed_kelvin becomes a seed
    on the first day it does, unless the targets date its burn later.

    Args:
        labels (np.ndarray): (windows, H, W) bool next-day targets.
        night_heat (np.ndarray): (days, H, W) night I4 running maximum of the days before the first target.
        label_sel (int): the label rule that built the targets (-1 if unknown).
        first_label_day (int): the day of window 0's target (the window length).
        seed_kelvin (float): the night I4 that marks a seed.

    Returns:
        BurnHistory: burn and seed days.
    """
    labels = np.asarray(labels, dtype=bool)
    burn_day = np.full(labels.shape[1:], -1, dtype=np.int16)
    for window, target in enumerate(labels):
        burn_day[target & (burn_day < 0)] = first_label_day + window

    seed_day = np.full(labels.shape[1:], -1, dtype=np.int16)
    hot = np.asarray(night_heat) > seed_kelvin
    if len(hot):
        first_hot = hot.argmax(axis=0)
        seeded = hot.any(axis=0) & (burn_day < 0)
        seed_day[seeded] = first_hot[seeded]
    return BurnHistory(burn_day, seed_day, int(first_label_day), int(first_label_day) + len(labels) - 1,
                       int(label_sel))


def pixel_days(history):
    """Give every pixel the day it was first seen: its burn day, else its seed day, else -1."""
    return np.where(history.burn_day >= 0, history.burn_day, history.seed_day)


def recent_pixels(history, day, days_back):
    """Mark the pixels first seen in the last days_back days up to and including day.

    Nothing after day is ever looked at. That is what keeps tracking causal,
    so inference can rebuild the same fires from the days it has.

    Args:
        history (BurnHistory): the burn history.
        day (int): the current day.
        days_back (int): how many days count as recent.

    Returns:
        np.ndarray: (H, W) bool.
    """
    days = pixel_days(history)
    return (days >= 0) & (days <= day) & (days > day - days_back)


# ----- Geometry ----- #


def link_clusters(mask, link_pixels):
    """Group the pixels of a mask so that any two within link_pixels of each other share a group.

    This is single-linkage clustering cut at link_pixels, done exactly with a
    k-d tree (a dilation would join pixels slightly farther apart). Groups are
    numbered 1, 2, ... in order of their first pixel in row-major order.

    Args:
        mask (np.ndarray): (H, W) bool.
        link_pixels (float): the largest gap, in pixels, that still joins two pixels.

    Returns:
        np.ndarray: (H, W) int32, the group of each mask pixel, 0 outside the mask.
    """
    clusters = np.zeros(mask.shape, dtype=np.int32)
    points = np.argwhere(mask)
    if len(points) == 0:
        return clusters
    pairs = cKDTree(points).query_pairs(link_pixels, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(points), len(points)))
    # Labels follow the first point of each group, and points are in row-major order.
    _, group = connected_components(graph, directed=False)
    clusters[points[:, 0], points[:, 1]] = group + 1
    return clusters


def nearest_owner(owner, mask):
    """Give every pixel of a mask the id of the closest owned pixel inside that mask.

    Only owners inside the mask compete, so a pixel is never handed to a fire
    outside its own group. Pixels outside the mask, and masks with no owner, get 0.

    Args:
        owner (np.ndarray): (H, W) int, fire ids with 0 for none.
        mask (np.ndarray): (H, W) bool, the pixels to assign.

    Returns:
        np.ndarray: (H, W) of owner's dtype.
    """
    result = np.zeros(owner.shape, dtype=owner.dtype)
    owned = np.argwhere(mask & (owner > 0))
    if len(owned) == 0:
        return result
    queries = np.argwhere(mask)
    _, closest = cKDTree(owned).query(queries)
    result[queries[:, 0], queries[:, 1]] = owner[owned[closest, 0], owned[closest, 1]]
    return result


def cluster_size(cluster_mask):
    """Count the pixels of a group, seed pixels included."""
    return int(np.count_nonzero(cluster_mask))


def growth_days(history, cluster_mask):
    """Count the different days on which a group's pixels were first seen."""
    days = pixel_days(history)[cluster_mask]
    return len(np.unique(days[days >= 0]))


def large_front_parts(front, settings):
    """Split a fire's front into parts at link distance and keep the parts with at least min_pixels.

    A front that falls apart into two or more large parts is a split. Small
    stray parts are noise and stay with the nearest large part.

    Args:
        front (np.ndarray): (H, W) bool, the fire's pixels of the last split_days.
        settings (SplitSettings): the thresholds.

    Returns:
        list[np.ndarray]: (H, W) bool masks, in group order.
    """
    parts = link_clusters(front, settings.link_pixels)
    masks = [parts == number for number in range(1, parts.max() + 1)]
    return [mask for mask in masks if cluster_size(mask) >= settings.min_pixels]


# ----- Fire events ----- #


def burning_fire_ids(events):
    """List the ids of the fires that have not ended, in id order."""
    return [event.event_id for event in events if event.end_day < 0]


def update_event(events, event_id, **changes):
    """Return the events with one event's fields changed."""
    return tuple(replace(event, **changes) if event.event_id == event_id else event for event in events)


def oldest_fire(events, fire_ids):
    """Pick the fire that started first among fire_ids (the lower id on a tie)."""
    starts = {event.event_id: event.start_day for event in events}
    return min(fire_ids, key=lambda fire_id: (starts[fire_id], fire_id))


# ----- One day of tracking ----- #


def start_tracking(history):
    """Make the empty state before day 0: no fires, no owners."""
    return TrackState(day=-1, owner=np.zeros(history.burn_day.shape, dtype=np.int32), events=(),
                      pending_splits=(), next_id=1)


def cluster_recent_pixels(history, day, settings):
    """Group the day's recent pixels at link distance, and the fronts at merge distance.

    Groups at link distance (pixels of the last memory_days) decide which
    pixels belong to one fire. Groups at merge distance (pixels of the last
    split_days only) decide which fires touch; old burned ground between two
    fires does not count as touching.

    Args:
        history (BurnHistory): the burn history.
        day (int): the day.
        settings (SplitSettings): the thresholds.

    Returns:
        DayClusters: both groupings.
    """
    memory = recent_pixels(history, day, settings.memory_days)
    fronts = recent_pixels(history, day, settings.split_days)
    return DayClusters(linked=link_clusters(memory, settings.link_pixels),
                       close=link_clusters(fronts, settings.merge_pixels))


def confirm_new_fires(state, clusters, history, settings):
    """Start a new fire for every group without a fire that has enough pixels and grew on enough days.

    Groups that fail the test stay without a fire. If they never pass it, they
    age out of memory: that is how speckle is cleaned. A new fire's start day
    is its earliest pixel's day.

    Args:
        state (TrackState): the state so far.
        clusters (DayClusters): the day's groups.
        history (BurnHistory): the burn history.
        settings (SplitSettings): the thresholds.

    Returns:
        TrackState: the state with the new fires.
    """
    owner = state.owner.copy()
    events = list(state.events)
    next_id = state.next_id
    days = pixel_days(history)
    for number in range(1, clusters.linked.max() + 1):
        group = clusters.linked == number
        if (owner[group] > 0).any():
            continue
        if cluster_size(group) < settings.min_pixels:
            continue
        if growth_days(history, group) < settings.min_growth_days:
            continue
        owner[group] = next_id
        events.append(FireEvent(next_id, "ignition", 0, int(days[group].min()), -1, 0))
        next_id += 1
    return replace(state, owner=owner, events=tuple(events), next_id=next_id)


def claim_unowned_pixels(state, clusters):
    """Hand every unowned pixel in a group that has a fire to the nearest fire of that group.

    This keeps a fire's new pixels, nearby speckle and bridged gaps with that
    fire. When a group holds two fires, each pixel goes to the closer one.

    Args:
        state (TrackState): the state so far.
        clusters (DayClusters): the day's groups.

    Returns:
        TrackState: the state with the pixels claimed.
    """
    owner = state.owner.copy()
    for number in range(1, clusters.linked.max() + 1):
        group = clusters.linked == number
        unowned = group & (owner == 0)
        if not unowned.any() or not (owner[group] > 0).any():
            continue
        nearest = nearest_owner(owner, group)
        owner[unowned] = nearest[unowned]
    return replace(state, owner=owner)


def merge_touching_fires(state, clusters):
    """Merge fires whose fronts share a group at merge distance; the oldest fire absorbs the others.

    Fires merge only when their recent fronts come within half the link
    distance. Between half and the full distance nothing changes, so two
    fires touching and parting from one day to the next do not flicker
    between one and two fires. Absorbed fires end that day.

    Args:
        state (TrackState): the state so far.
        clusters (DayClusters): the day's groups.

    Returns:
        TrackState: the state after merging.
    """
    owner = state.owner.copy()
    events = state.events
    for number in range(1, clusters.close.max() + 1):
        burning = set(burning_fire_ids(events))
        fire_ids = sorted(int(fire_id) for fire_id in np.unique(owner[clusters.close == number]) if fire_id in burning)
        if len(fire_ids) < 2:
            continue
        survivor = oldest_fire(events, fire_ids)
        for absorbed in fire_ids:
            if absorbed != survivor:
                owner[owner == absorbed] = survivor
                events = update_event(events, absorbed, end_day=state.day, merged_into=survivor)
    return replace(state, owner=owner, events=events)


def split_diverged_fires(state, history, settings):
    """Split every fire whose front has been in two or more large parts for min_growth_days days.

    The fire ends and each large part becomes a new fire whose parent is the
    split fire, so every branch keeps the shared history before the split.
    The fire's recent pixels go to the nearest branch. Older pixels keep the
    parent's id, as a record of where it burned.

    Args:
        state (TrackState): the state so far.
        history (BurnHistory): the burn history.
        settings (SplitSettings): the thresholds.

    Returns:
        TrackState: the state after splitting.
    """
    owner = state.owner.copy()
    events = state.events
    next_id = state.next_id
    days_split = dict(state.pending_splits)
    still_pending = []
    fronts = recent_pixels(history, state.day, settings.split_days)
    memory = recent_pixels(history, state.day, settings.memory_days)
    for fire_id in burning_fire_ids(events):
        parts = large_front_parts((owner == fire_id) & fronts, settings)
        days = days_split.get(fire_id, 0) + 1 if len(parts) >= 2 else 0
        if days < settings.min_growth_days:
            if days:
                still_pending.append((fire_id, days))
            continue
        branches = np.zeros_like(owner)
        for part in parts:
            branches[part] = next_id
            events += (FireEvent(next_id, "split", fire_id, state.day, -1, 0),)
            next_id += 1
        fire_pixels = (owner == fire_id) & memory
        nearest = nearest_owner(branches, fire_pixels)
        owner[fire_pixels] = nearest[fire_pixels]
        events = update_event(events, fire_id, end_day=state.day)
    return replace(state, owner=owner, events=events, next_id=next_id, pending_splits=tuple(still_pending))


def end_quiet_fires(state, history, settings):
    """End every fire that has no pixel in the last memory_days days.

    A fire hidden by clouds for a few days keeps its id because its pixels stay
    in memory. Only a fire quiet for the whole memory ends.

    Args:
        state (TrackState): the state so far.
        history (BurnHistory): the burn history.
        settings (SplitSettings): the thresholds.

    Returns:
        TrackState: the state with quiet fires ended.
    """
    memory = recent_pixels(history, state.day, settings.memory_days)
    events = state.events
    for fire_id in burning_fire_ids(events):
        if not ((state.owner == fire_id) & memory).any():
            events = update_event(events, fire_id, end_day=state.day)
    return replace(state, events=events)


def advance_one_day(state, history, settings):
    """Track one more day: cluster, confirm, claim, merge, split, end.

    The steps run in this order every day. Each one returns a new state, so any
    step can be tested on its own.

    Args:
        state (TrackState): the state at the end of the previous day.
        history (BurnHistory): the burn history.
        settings (SplitSettings): the thresholds.

    Returns:
        TrackState: the state at the end of the new day.
    """
    state = replace(state, day=state.day + 1)
    clusters = cluster_recent_pixels(history, state.day, settings)
    state = confirm_new_fires(state, clusters, history, settings)
    state = claim_unowned_pixels(state, clusters)
    state = merge_touching_fires(state, clusters)
    state = split_diverged_fires(state, history, settings)
    return end_quiet_fires(state, history, settings)


def track_fire(history, settings):
    """Track every fire of one archive from day 0 to its last target day.

    The owner map of every day is kept. Each window's zones are then built
    from exactly the days that window saw.

    Args:
        history (BurnHistory): the burn history.
        settings (SplitSettings): the thresholds.

    Returns:
        FireTracks: the daily owner maps and every fire.
    """
    state = start_tracking(history)
    owners = []
    for _ in range(history.last_day + 1):
        state = advance_one_day(state, history, settings)
        owners.append(state.owner)
    if not owners:
        return FireTracks(np.zeros((0, *history.burn_day.shape), dtype=np.int32), state.events)
    return FireTracks(np.stack(owners), state.events)
