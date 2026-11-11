"""Fire split zones on daily archives: the same zone files as on the window archives they came from."""

import json

import numpy as np
import pytest

from test_daily import daily_fire, v2_archive


def growing_fire(rng, num_days, size=64):
    """Days and targets of a fire that spreads from the centre, plus a far ignition from day 8."""
    days, _ = daily_fire(rng, num_days, size=size)
    rows, cols = np.mgrid[:size, :size]
    radius = np.hypot(rows - size // 3, cols - size // 3)
    far = np.hypot(rows - 52, cols - 52)
    burned = [(radius <= day) | ((far <= day - 8) if day >= 8 else np.zeros_like(radius, bool))
              for day in range(num_days + 1)]
    targets = np.stack([burned[0]] + [burned[d] & ~burned[d - 1] for d in range(1, num_days + 1)])
    heat = np.where(radius <= 2, 330.0, 290.0).astype(np.float32)
    days[6] = np.maximum.accumulate(np.broadcast_to(heat, days[6].shape) + rng.normal(0, 1, days[6].shape), axis=0)
    return days, targets


def same_zone_file(a, b):
    """Zone files are equal: arrays bit for bit, fires, settings and label rule."""
    return (a.zones.tobytes() == b.zones.tobytes() and a.target_owners.tobytes() == b.target_owners.tobytes()
            and a.events == b.events and a.settings == b.settings and a.label_sel == b.label_sel
            and a.first_label_day == b.first_label_day)


def test_zones_from_daily_archives_match_window_archives(tmp_path):
    """build_zones writes identical zone files from a format-2 root and the daily root converted from it."""
    pytest.importorskip("scipy")
    from firepred.data.archive import read_zone_file
    from firepred.data.windows_master import convert_daily
    from firepred.data.zones_master import build_zones

    rng = np.random.default_rng(8)
    v2 = tmp_path / "v2"
    for name in ("p1", "p2"):
        v2_archive(v2 / "train" / f"{name}.npz", *growing_fire(rng, 14))
    (v2 / "window_config.json").write_text(json.dumps({"ts_length": 6, "ts_interval": 1}))
    convert_daily(data_root=v2, dest_root=tmp_path / "v3", splits=("train",), compresslevel=1, work_dir=tmp_path)

    build_zones(v2, splits=("train",))
    build_zones(tmp_path / "v3", splits=("train",))
    for name in ("p1", "p2"):
        old = read_zone_file(v2 / "zones" / f"{name}.npz")
        new = read_zone_file(tmp_path / "v3" / "zones" / f"{name}.npz")
        assert len(old.events) >= 2  # the far ignition is its own fire
        assert same_zone_file(old, new)


def test_daily_roots_need_no_daily_stride(tmp_path):
    """Zones refuse format-2 windows that skip days but read a daily root at -it 1 whatever it records."""
    from firepred.data.archive import read_fire_labels
    from firepred.data.windows_master import convert_daily
    from firepred.data.zones_master import require_daily_windows

    rng = np.random.default_rng(9)
    v2 = tmp_path / "v2"
    days, targets = growing_fire(rng, 12)
    v2_archive(v2 / "train" / "p1.npz", days, targets)
    (v2 / "window_config.json").write_text(json.dumps({"ts_length": 6, "ts_interval": 1}))
    convert_daily(data_root=v2, dest_root=tmp_path / "v3", splits=("train",), compresslevel=1, work_dir=tmp_path)
    config = json.loads((tmp_path / "v3" / "window_config.json").read_text())
    (tmp_path / "v3" / "window_config.json").write_text(json.dumps({**config, "ts_interval": 3}))

    with pytest.raises(ValueError, match="-it 1"):
        require_daily_windows(v2, 6, 3)
    require_daily_windows(tmp_path / "v3", 6, 3)  # daily: only the length is checked
    with pytest.raises(ValueError, match="-ts"):
        require_daily_windows(tmp_path / "v3", 7, 3)
    labels, label_sel = read_fire_labels(tmp_path / "v3" / "train" / "p1.npz", 6)
    assert labels.shape[0] == 7 and label_sel == 0
    assert labels.tobytes() == targets[6:13].tobytes()
