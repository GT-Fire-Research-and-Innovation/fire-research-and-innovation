"""Which fires go where: ROI metadata, train/val/test split, label rule, fire ids.

Level 1 (data, dataset building). Pure functions of ROI CSV files already on
disk (downloading them is data/hub.py). Used by the two master modules. The
split [released]: train/val are the 2017-2020 fires (validation is the fixed
VAL_IDS list), test is the 2021 fires. Which fires get TESSERA is
data/tessera.py.
"""

import os
from dataclasses import dataclass, field

import pandas as pd

from firepred.config.dataset import SPLITS, TEST_YEARS, TRAIN_VAL_YEARS, VAL_IDS


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
        """Group the fire ids by split.

        Most steps loop over splits, so this gives the three id lists under their
        split names. The order is train, val, test.

        Returns:
            dict[str, tuple[str, ...]]: {"train": ids, "val": ids, "test": ids}.
        """
        return dict(zip(SPLITS, (self.train_ids, self.val_ids, self.test_ids)))

    def split_of(self):
        """Look up which split each fire belongs to.

        This is the reverse of ids_by_split. It is used to put a downloaded archive
        into the right train/val/test folder.

        Returns:
            dict[str, str]: {fire id: split name} for every listed fire.
        """
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
    """Read the ROI CSVs into one RoiTable.

    This is the one call the masters make: it splits the fires, reads their
    label rules and their start years. Every later step uses the table
    instead of the files.

    Args:
        roi_paths (list[str]): local paths of the ROI CSV files.

    Returns:
        RoiTable: the splits, label rules and start years.
    """
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
    """Turn an archive file name into its fire id.

    Window archives are named p<fire id>.npz and may sit in a split folder.
    This strips the folder, the extension and the leading "p".

    Args:
        archive_name (str): e.g. "train/p20562846.npz".

    Returns:
        str: the fire id, e.g. "20562846".
    """
    return os.path.basename(archive_name).removesuffix(".npz").removeprefix("p")
