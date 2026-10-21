"""Local files: ROI CSVs -> fire splits, zips, data roots, the code version.

Level 1 (data, dataset building). Pure functions of files already on disk (downloading
is data/hub.py); none prints. Used by the masters, data/hub.py and data/tessera.py.

    read_roi -> RoiTable        train/val are the 2017-2020 fires (val is the fixed VAL_IDS
                                list), test the 2021 fires [released]; label rules, start
                                years. Which fires get TESSERA is data/tessera.py.
    zip_folders, unzip_into     bundle folders (data.zip, tessera.zip); unpack with a path check
    extract_zip, extract_data   unpack once per session -> train/val/test root
    code_version                git commit of this checkout (provenance)
"""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from firepred.config.dataset import (
    CODE_MANIFEST_FILE,
    REPO_ROOT,
    SPLITS,
    TEST_YEARS,
    TRAIN_VAL_YEARS,
    VAL_IDS,
)
from firepred.data.archive import zipfile


# ===== ROI CSVs: which fires go where ===== #


@dataclass(frozen=True)
class RoiTable:
    """Hold everything the pipeline needs from the ROI CSVs, read once.

    The ROI CSVs list every fire with its id, dates and label rule. Reading them
    once into this table and passing it around means no step downloads or
    parses them again. The table is frozen, so no step can change it.

    Attributes:
        train_ids (tuple[str, ...]): training fire ids, sorted by id.
        val_ids (tuple[str, ...]): validation fire ids (the fixed VAL_IDS list).
        test_ids (tuple[str, ...]): test fire ids, in CSV order.
        label_sel (dict[str, int]): label rule per fire, where a CSV records one.
        start_years (dict[str, int]): the year each fire started.
    """

    train_ids: tuple
    val_ids: tuple
    test_ids: tuple
    label_sel: dict = field(default_factory=dict)
    start_years: dict = field(default_factory=dict)

    def ids_by_split(self):
        """Group the fire ids by split."""
        return dict(zip(SPLITS, (self.train_ids, self.val_ids, self.test_ids)))

    def split_of(self):
        """Look up which split each fire belongs to."""
        return {fire_id: split for split, ids in self.ids_by_split().items() for fire_id in ids}


def split_ids(roi_paths, train_val_years=TRAIN_VAL_YEARS, test_years=TEST_YEARS, val_ids=VAL_IDS):
    """Split the fire ids into train, validation and test from the ROI CSVs.

    The year in each CSV's file name decides where its fires go: 2017-2020
    fires are for training and validation, 2021 fires are for testing
    [released]. Validation is a fixed list of fire ids; every other 2017-2020
    fire trains.

    Args:
        roi_paths (list[str]): local paths of the ROI CSV files.
        train_val_years (tuple[str, ...]): file-name years of train/val fires.
        test_years (tuple[str, ...]): file-name years of test fires.
        val_ids (list[str]): the fire ids used for validation.

    Returns:
        tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]: (train, val,
        test) fire ids; train and val sorted by id, test in CSV order.
    """
    train_val = [
        pd.read_csv(path, dtype={"Id": str})
        for path in roi_paths
        if any(year in os.path.basename(path) for year in train_val_years)
    ]
    df = pd.concat(train_val, ignore_index=True).sort_values(by=["Id"])
    df["Id"] = df["Id"].astype(str)
    train = tuple(df[~df.Id.isin(val_ids)]["Id"].values.astype(str).tolist())
    val = tuple(df[df.Id.isin(val_ids)]["Id"].values.astype(str).tolist())

    test_frames = [
        pd.read_csv(path, dtype={"Id": str})
        for path in roi_paths
        if any(year in os.path.basename(path) for year in test_years)
    ]
    test = tuple(pd.concat(test_frames, ignore_index=True)["Id"].values.astype(str).tolist())
    return train, val, test


def fire_label_sel(roi_paths):
    """Read which burned-area label rule each fire uses.

    Some ROI CSVs have a label_sel column: 0 means active fire plus NIFC
    perimeters, 1 means active fire only. A fire listed in two CSVs must have
    the same rule in both.

    Args:
        roi_paths (list[str]): local paths of the ROI CSV files.

    Returns:
        dict[str, int]: {fire id: label_sel} for fires that have one.

    Raises:
        ValueError: if two CSVs give the same fire different rules.
    """
    label_sel = {}
    for path in roi_paths:
        df = pd.read_csv(path, dtype={"Id": str})
        if "Id" not in df.columns or "label_sel" not in df.columns:
            continue
        for fire_id, value in zip(df["Id"], df["label_sel"]):
            if pd.isna(value):
                continue
            fire_id, value = str(fire_id), int(value)
            if label_sel.setdefault(fire_id, value) != value:
                raise ValueError(f"ROI CSVs disagree on label_sel for {fire_id}")
    return label_sel


def fire_start_years(roi_paths):
    """Read the year each fire started.

    The start year decides which TESSERA embeddings a fire can use (the year
    before the fire). Fires without a start date are left out.

    Args:
        roi_paths (list[str]): local paths of the ROI CSV files.

    Returns:
        dict[str, int]: {fire id: start year}.
    """
    years = {}
    for path in roi_paths:
        df = pd.read_csv(path, dtype={"Id": str})
        if "Id" not in df.columns or "start_date" not in df.columns:
            continue
        for fire_id, start_date in zip(df["Id"], df["start_date"]):
            if pd.notna(start_date):
                years[str(fire_id)] = int(pd.to_datetime(start_date).year)
    return years


def read_roi(roi_paths):
    """Read the ROI CSVs once into a RoiTable (splits, label rules, start years) that every later step uses."""
    train, val, test = split_ids(roi_paths)
    return RoiTable(train, val, test, fire_label_sel(roi_paths), fire_start_years(roi_paths))


def label_rules(roi, splits, use_roi_label_sel, train_val_label_sel):
    """Pick the burned-area label rule for every fire of the chosen splits.

    A fire uses its CSV's rule when it has one (the paper's per-fire choice for
    test fires); every other fire uses train_val_label_sel. With
    use_roi_label_sel False, every fire uses train_val_label_sel.

    Args:
        roi (RoiTable): the fires and their recorded rules.
        splits (tuple[str, ...]): which splits to cover.
        use_roi_label_sel (bool): whether CSV rules take precedence.
        train_val_label_sel (int): the rule used otherwise (0 or 1).

    Returns:
        dict[str, int]: {fire id: label_sel}.
    """
    from_roi = roi.label_sel if use_roi_label_sel else {}
    ids = roi.ids_by_split()
    return {
        str(fire_id): int(from_roi.get(str(fire_id), train_val_label_sel))
        for split in splits
        for fire_id in ids[split]
    }


def fire_id_of(archive_name):
    """Turn an archive name such as "train/p20562846.npz" into its fire id, "20562846"."""
    return os.path.basename(archive_name).removesuffix(".npz").removeprefix("p")


# ===== Zips, data roots and the code version ===== #


def find_data_root(base):
    """Find the one folder under a path that holds train/, val/ and test/ archives.

    An unzipped data.zip may put the data root at any depth. This searches for
    the folder with all three split folders and checks that each contains
    window archives.

    Args:
        base (str | Path): where to start looking.

    Returns:
        Path: the data root.

    Raises:
        RuntimeError: if there is not exactly one such folder, or a split is empty.
    """
    base = Path(base)
    # The root may sit at any depth inside the unzipped folder; there must be
    # exactly one folder holding train/, val/ and test/.
    candidates = [base] + [p.parent for p in base.rglob("train") if p.is_dir()]
    matches = sorted({p.resolve() for p in candidates
                      if all((p / split).is_dir() for split in SPLITS)})
    if len(matches) != 1:
        raise RuntimeError(f"Expected one train/val/test root under {base}; found {matches}")
    root = matches[0]
    for split in SPLITS:
        if not next((root / split).glob("*.npz"), None):
            raise RuntimeError(f"No NPZ files directly inside {root / split}")
    return root


def count_archives(root):
    """Count the window archives in each split folder of a data root."""
    root = Path(root)
    return {split: sum(1 for _ in (root / split).glob("*.npz")) for split in SPLITS}


def check_members_inside(archive, destination):
    """Refuse a zip whose members would be written outside the target folder.

    A malicious zip can name a member "../../file" to escape the folder it is
    unpacked into ("zip slip"). Python already guards against this; this check
    makes it explicit and fails loudly instead of quietly renaming.

    Args:
        archive (zipfile.ZipFile): the open zip to check.
        destination (str | Path): the folder it will be extracted into.

    Returns:
        list[str]: the member names checked.

    Raises:
        ValueError: if any member would land outside destination.
    """
    root = Path(destination).resolve()
    names = []
    for item in archive.infolist():
        if not (root / item.filename).resolve().is_relative_to(root):
            raise ValueError(f"Archive member escapes extraction folder: {item.filename}")
        names.append(item.filename)
    return names


def zip_folders(folders, output_path, extra_files=()):
    """Zip every file under the given folders, plus some loose files, into one zip file.

    Each member keeps its folder name (train/p123.npz), so unzipping recreates
    the same layout; loose files go at the zip's root under their own name
    (window_config.json). Files are stored without recompression, because the
    archives inside are already Zstandard-compressed and would not shrink again.
    Folders that do not exist are skipped.

    Args:
        folders (list[str | Path]): folders to include.
        output_path (str | Path): the zip file to write.
        extra_files (Iterable[str | Path]): loose files to add at the root.

    Returns:
        list[str]: the member names written.
    """
    members = []
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_STORED) as bundle:
        for folder in map(Path, folders):
            if not folder.exists():
                continue
            for file_path in folder.rglob("**/*"):
                if file_path.is_file():
                    name = file_path.relative_to(folder.parent)
                    bundle.write(file_path, name)
                    members.append(str(name))
        for file_path in map(Path, extra_files):
            bundle.write(file_path, file_path.name)
            members.append(file_path.name)
    return members


def unzip_into(zip_path, destination):
    """Unpack a zip into a folder after checking its member paths."""
    destination = Path(destination)
    with zipfile.ZipFile(zip_path, "r") as bundle:
        check_members_inside(bundle, destination)
        bundle.extractall(destination)
    return destination


def extract_zip(archive_path, destination):
    """Unpack a zip once per session, checking disk space and member paths.

    A marker file records which zip was unpacked: the same zip again does
    nothing, while a different zip is refused instead of being mixed in. The
    disk must have room for the unpacked files plus 1 GiB.

    Args:
        archive_path (str | Path): the zip file.
        destination (str | Path): the folder to unpack into.

    Returns:
        Path: destination.

    Raises:
        RuntimeError: if a different zip was unpacked there, or the disk is too small.
    """
    archive_path, destination = Path(archive_path), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    # A marker records which zip was unpacked here: the same zip again is a
    # no-op, a different zip is refused rather than mixed in.
    marker = destination / ".extraction-complete.json"
    stat = archive_path.stat()
    signature = {"path": str(archive_path.resolve()), "size": stat.st_size,
                 "mtime_ns": stat.st_mtime_ns}
    if marker.exists():
        if json.loads(marker.read_text()) == signature:
            return destination
        raise RuntimeError("The archive changed. Choose a fresh extraction folder first.")
    with zipfile.ZipFile(archive_path) as archive:
        # Check free disk space (with 1 GiB to spare) before unpacking.
        needed = sum(item.file_size for item in archive.infolist())
        free = shutil.disk_usage(destination).free
        if needed + 1024**3 > free:
            raise RuntimeError(
                f"Insufficient disk space: {needed / 1024**3:.1f} GiB needed, {free / 1024**3:.1f} GiB free. "
                "Use an attached, extracted Kaggle Dataset."
            )
        check_members_inside(archive, destination)
        archive.extractall(destination)
    marker.write_text(json.dumps(signature))
    return destination


def extract_data(archive_path, destination):
    """Unpack a data.zip and return its train/val/test root."""
    return find_data_root(extract_zip(archive_path, destination))


def code_version():
    """Read which code this is: the git commit, or a packed copy's recorded version.

    The version is stored with every TESSERA file and window_config.json, so
    anyone can see which code built them. A copy packed by tools/pack_code.py
    has no git history; its code_version.json says which commit it was packed
    from and whether it held uncommitted changes. Without either it is empty.

    Returns:
        str: the commit hash, the packed copy's version, or "" when neither can be read.
    """
    try:
        return subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        pass
    try:
        return json.loads((REPO_ROOT / CODE_MANIFEST_FILE).read_text())["code_version"]
    except (OSError, ValueError, KeyError):
        return ""
