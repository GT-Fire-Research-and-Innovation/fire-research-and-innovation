"""Fire tracking and zone files: new ignitions, splits, merges, cleaning, causality and the files on disk.

Every scenario is drawn by hand on a small grid with the default separation
distance (2.25 km: link 6 px, merge 3 px, memory 11 days, front 3 days, at
least 6 pixels and 2 growth days for a new fire). Fronts are 4 rows tall.
"""

import numpy as np
import pytest

from firepred.data.archive import read_zone_file, write_fire_archive, write_zone_file
from firepred.data.fire_tracks import BurnHistory, link_clusters, split_settings, track_fire
from firepred.data.fire_zones import fire_sample, window_target_owners, window_zones, zone_file
from firepred.data.zones_master import build_zones

SETTINGS = split_settings(2.25)
SIZE = 64
FIRST_LABEL_DAY = 6
ROWS = slice(30, 34)


def history_of(burn_day, last_day, seed_day=None):
    """Make a burn history from a hand-drawn burn-day map (and optional seed-day map)."""
    seeds = np.full((SIZE, SIZE), -1, dtype=np.int16) if seed_day is None else seed_day
    return BurnHistory(burn_day.astype(np.int16), seeds.astype(np.int16), FIRST_LABEL_DAY, last_day, -1)


def empty_days():
    """A burn-day map with nothing burned."""
    return np.full((SIZE, SIZE), -1, dtype=np.int16)


def paint(days, columns_by_day, rows=ROWS):
    """Return a copy of a burn-day map with columns first burned on the given days."""
    days = days.copy()
    for day, (first, last) in columns_by_day.items():
        days[rows, first:last + 1] = day
    return days


def kinds(events):
    """The (id, kind, parent, start, end, merged_into) of every event, for readable asserts."""
    return [(e.event_id, e.kind, e.parent_id, e.start_day, e.end_day, e.merged_into) for e in events]


# ----- Settings and geometry ----- #


def test_split_settings_formulas():
    """Every threshold follows from the one distance."""
    assert (SETTINGS.link_pixels, SETTINGS.merge_pixels, SETTINGS.memory_days, SETTINGS.split_days,
            SETTINGS.min_pixels, SETTINGS.min_growth_days) == (6.0, 3.0, 11, 3, 6, 2)
    small = split_settings(0.75)
    assert (small.link_pixels, small.merge_pixels, small.memory_days, small.split_days, small.min_pixels) == \
        (2.0, 1.0, 4, 1, 2)
    with pytest.raises(ValueError):
        split_settings(0)


def test_link_clusters_is_exact_single_linkage():
    """Pixels exactly the link distance apart join; slightly farther apart they do not."""
    mask = np.zeros((20, 20), dtype=bool)
    mask[2, 2] = mask[2, 8] = True          # 6.0 px apart: one group
    mask[12, 2] = mask[13, 8] = True        # 6.08 px apart: two groups
    groups = link_clusters(mask, 6.0)
    assert groups[2, 2] == groups[2, 8] == 1
    assert groups[12, 2] != groups[13, 8]


# ----- Tracking scenarios ----- #


def test_far_ignition_becomes_new_fire():
    """A fire starting far from the first one is a second ignition, dated from its first pixel."""
    days = paint(empty_days(), {6: (10, 13), 7: (14, 16), 8: (17, 19)}, rows=slice(10, 14))
    days = paint(days, {9: (40, 43), 10: (44, 46)}, rows=slice(40, 44))
    tracks = track_fire(history_of(days, last_day=12), SETTINGS)
    assert kinds(tracks.events) == [(1, "ignition", 0, 6, -1, 0), (2, "ignition", 0, 9, -1, 0)]
    assert tracks.owners[-1][41, 41] == 2


def test_diverging_front_makes_two_children():
    """A front that breaks into two large parts for two days ends its fire and starts two branches."""
    days = paint(empty_days(), {6: (28, 35), 7: (24, 27), 8: (20, 23), 9: (16, 19), 10: (12, 15), 11: (8, 11)})
    days = paint(days, {7: (36, 39), 8: (40, 43), 9: (44, 47), 10: (48, 51), 11: (52, 55)})
    tracks = track_fire(history_of(days, last_day=12), SETTINGS)
    assert kinds(tracks.events) == [(1, "ignition", 0, 6, 10, 0), (2, "split", 1, 10, -1, 0),
                                    (3, "split", 1, 10, -1, 0)]
    final = tracks.owners[-1]
    assert final[31, 9] == 2 and final[31, 53] == 3


def test_merge_keeps_older_id():
    """Two fires whose fronts meet merge into the older one; the younger ends that day."""
    days = paint(empty_days(), {6: (5, 7), 7: (8, 10), 8: (11, 13), 9: (14, 16), 10: (17, 19), 11: (20, 22),
                                12: (23, 25)})
    days = paint(days, {8: (40, 42), 9: (37, 39), 10: (34, 36), 11: (31, 33), 12: (28, 30)})
    tracks = track_fire(history_of(days, last_day=13), SETTINGS)
    assert kinds(tracks.events) == [(1, "ignition", 0, 6, -1, 0), (2, "ignition", 0, 8, 12, 1)]
    assert set(np.unique(tracks.owners[-1][days >= 0])) == {1}


def test_single_pixel_speckle_never_confirmed():
    """Lone pixels far from the fire never become fires and own no target pixel."""
    days = paint(empty_days(), {6: (10, 13), 7: (14, 16), 8: (17, 19), 9: (20, 22)})
    days[50, 50] = 8
    days[55, 20] = 9
    history = history_of(days, last_day=12)
    tracks = track_fire(history, SETTINGS)
    assert len(tracks.events) == 1
    assert (tracks.owners[:, 50, 50] == 0).all()
    assert window_target_owners(tracks, history, 8 - FIRST_LABEL_DAY, SETTINGS)[50, 50] == -1


def test_scattered_seed_heat_never_creates_fire():
    """Warm pixels before the first target, spread too thin to group, start no fire."""
    seeds = np.full((SIZE, SIZE), -1, dtype=np.int16)
    seeds[::8, ::8] = 2
    tracks = track_fire(history_of(empty_days(), last_day=10, seed_day=seeds), SETTINGS)
    assert tracks.events == ()


def test_seed_cluster_starts_fire_before_first_target():
    """A compact patch of early heat over two days is a fire already burning when the targets begin."""
    seeds = np.full((SIZE, SIZE), -1, dtype=np.int16)
    seeds[ROWS, 10:14] = 2
    seeds[ROWS, 14:17] = 3
    days = paint(empty_days(), {6: (17, 19)})
    history = history_of(days, last_day=8, seed_day=seeds)
    tracks = track_fire(history, SETTINGS)
    assert kinds(tracks.events) == [(1, "ignition", 0, 2, -1, 0)]
    assert window_target_owners(tracks, history, 0, SETTINGS)[31, 18] == 1


def test_cloud_gap_does_not_split_or_restart():
    """Two days without detections keep the same fire: no new ignition, no split, no end."""
    days = paint(empty_days(), {6: (10, 13), 7: (14, 16), 8: (17, 19), 11: (20, 22), 12: (23, 25)})
    tracks = track_fire(history_of(days, last_day=13), SETTINGS)
    assert kinds(tracks.events) == [(1, "ignition", 0, 6, -1, 0)]
    assert tracks.owners[-1][31, 24] == 1


# ----- Determinism and causality ----- #


def test_window_zones_ignore_future_labels():
    """A window's zones and target owners do not change when only later days change."""
    days = paint(empty_days(), {6: (10, 13), 7: (14, 16), 8: (17, 19), 9: (20, 22), 10: (23, 25)})
    later_fire = paint(days, {10: (40, 47), 11: (48, 55)}, rows=slice(50, 54))
    first, second = history_of(days, last_day=12), history_of(later_fire, last_day=12)
    first_tracks, second_tracks = track_fire(first, SETTINGS), track_fire(second, SETTINGS)
    for window in range(4):  # target days 6 .. 9, before the later fire
        assert (window_zones(first_tracks, first, window, SETTINGS)
                == window_zones(second_tracks, second, window, SETTINGS)).all()
        assert (window_target_owners(first_tracks, first, window, SETTINGS)
                == window_target_owners(second_tracks, second, window, SETTINGS)).all()


def test_rerun_is_identical():
    """The same history always gives the same fires and owner maps."""
    days = paint(empty_days(), {6: (28, 35), 7: (24, 27), 8: (20, 23), 9: (16, 19), 10: (12, 15)})
    days = paint(days, {7: (36, 39), 8: (40, 43), 9: (44, 47), 10: (48, 51)})
    history = history_of(days, last_day=11)
    first, second = track_fire(history, SETTINGS), track_fire(history, SETTINGS)
    assert first.events == second.events
    assert (first.owners == second.owners).all()


# ----- Training samples ----- #


def test_fire_sample_keeps_fires_apart():
    """A fire's sample targets only its own new pixels and skips other fires' zones and new pixels."""
    zones = np.zeros((4, 6), dtype=np.int16)
    zones[:, :2], zones[:, 4:] = 1, 2
    owners = np.zeros((4, 6), dtype=np.int16)
    owners[0, 1], owners[0, 4], owners[3, 3] = 1, 2, -1
    target, mask = fire_sample(zones, owners, 1)
    assert target.sum() == 1 and target[0, 1]
    assert not mask[:, 4:].any()                   # fire 2's zone
    assert not mask[3, 3]                          # a new ignition
    assert mask[:, :4].sum() == 15                 # fire 1's zone and background, minus the ignition


# ----- Files ----- #


def test_zone_file_round_trip(tmp_path):
    """A zone file reads back exactly as written: zones, target owners, fires and settings."""
    days = paint(empty_days(), {6: (28, 35), 7: (24, 27), 8: (20, 23), 9: (16, 19), 10: (12, 15)})
    days = paint(days, {7: (36, 39), 8: (40, 43), 9: (44, 47), 10: (48, 51)})
    history = history_of(days, last_day=11)
    zones = zone_file(track_fire(history, SETTINGS), history, SETTINGS)
    saved = read_zone_file(write_zone_file(tmp_path / "p1.npz", zones))
    assert saved.events == zones.events and saved.settings == zones.settings
    assert (saved.zones == zones.zones).all() and (saved.target_owners == zones.target_owners).all()
    assert (saved.label_sel, saved.first_label_day) == (-1, FIRST_LABEL_DAY)


def test_build_zones_from_archives(tmp_path):
    """The master writes a zone file per archive, skips it next time and refuses another distance."""
    labels = np.zeros((4, 32, 32), dtype=np.float32)
    labels[0, 10:14, 10:14] = 1
    labels[1, 10:14, 14:17] = 1
    labels[2, 10:14, 17:20] = 1
    data = np.zeros((4, 27, 6, 32, 32), dtype=np.float32)
    (tmp_path / "test").mkdir()
    write_fire_archive(tmp_path / "test" / "p7.npz", data, labels, label_sel=1)

    first = build_zones(tmp_path, splits=("test",))
    saved = read_zone_file(tmp_path / "zones" / "p7.npz")
    assert len(saved.zones) == 4 and saved.label_sel == 1 and len(saved.events) == 1
    assert "skipped" in build_zones(tmp_path, splits=("test",))[0] and "skipped" not in first[0]
    with pytest.raises(ValueError):
        build_zones(tmp_path, splits=("test",), scale_km=3.0)
