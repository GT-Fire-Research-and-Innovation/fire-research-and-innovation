"""The window archive format: write, find, open, unzip."""

import numpy as np

from firepred.data.archive import find_fire_sources, fire_name, open_fire, unzip_data_root, write_fire_archive


def test_write_then_read_round_trip(tmp_path):
    """write_fire_archive writes every window as its own member and returns the path; reading gives them back."""
    data = np.arange(2 * 27 * 2 * 4 * 4, dtype=np.float32).reshape(2, 27, 2, 4, 4)
    labels = np.zeros((2, 4, 4), dtype=np.float32)
    labels[1, 0, 0] = 1
    path = write_fire_archive(tmp_path / "p1.npz", data, labels, label_sel=0)
    assert path == tmp_path / "p1.npz"
    with open_fire(path) as archive:
        assert int(archive["num_windows"]) == 2 and int(archive["format_version"]) == 2
        assert int(archive["label_sel"]) == 0
        np.testing.assert_array_equal(archive["data_001"], data[1])
        np.testing.assert_array_equal(archive["label_001"], labels[1])


def test_unzipped_folders_replace_archives_and_read_the_same(data_root, tmp_path):
    """Unzipped fire folders read exactly like the archives, and a second unzip call reuses them."""
    root, _ = data_root
    unzipped = unzip_data_root(root, tmp_path / "npy")
    assert unzipped == tmp_path / "npy"
    archives = find_fire_sources(root / "train")
    folders = find_fire_sources(unzipped / "train")
    assert [fire_name(p) for p in archives] == [fire_name(p) for p in folders]
    assert all(p.is_dir() for p in folders)
    with open_fire(archives[0]) as a, open_fire(folders[0]) as b:
        np.testing.assert_array_equal(a["data_000"], b["data_000"])
    # Second call reuses the folders.
    assert unzip_data_root(root, tmp_path / "npy") == tmp_path / "npy"
