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
    """Everything the pipeline needs from the ROI CSVs, read once.

    train_ids / val_ids / test_ids: fire IDs per split, in pipeline order
    label_sel:   {fire id: label rule} where a CSV records one
    start_years: {fire id: year the fire started}
    """

    train_ids: tuple
    val_ids: tuple
    test_ids: tuple
    label_sel: dict = field(default_factory=dict)
    start_years: dict = field(default_factory=dict)

    def ids_by_split(self):
        """{"train": ids, "val": ids, "test": ids}."""
        return dict(zip(SPLITS, (self.train_ids, self.val_ids, self.test_ids)))

    def split_of(self):
        """{fire id: split name} for every listed fire."""
        return {fire_id: split for split, ids in self.ids_by_split().items() for fire_id in ids}


def split_ids(roi_paths, train_val_years=TRAIN_VAL_YEARS, test_years=TEST_YEARS, val_ids=VAL_IDS):
    """(train, val, test) fire-ID tuples from the ROI CSVs, by the year in each file name.

    Train/val come from the train/val years, sorted by Id (val is the fixed
    `val_ids` list); test keeps the CSV order of the test-year files.
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
    """{fire Id: label_sel} from every ROI CSV that has the column; a fire listed twice must agree."""
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
    """{fire Id: start year} from the ROI CSVs that have a start_date column."""
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
    """A RoiTable from ROI CSV files on disk."""
    train, val, test = split_ids(roi_paths)
    return RoiTable(train, val, test, fire_label_sel(roi_paths), fire_start_years(roi_paths))


def label_rules(roi, splits, use_roi_label_sel, train_val_label_sel):
    """{fire id: label_sel} for every fire of `splits`.

    The ROI CSV's label_sel where it has one (the paper's per-fire choice for
    test fires), else `train_val_label_sel`; with use_roi_label_sel=False
    every fire uses `train_val_label_sel`.
    """
    from_roi = roi.label_sel if use_roi_label_sel else {}
    ids = roi.ids_by_split()
    return {
        str(fire_id): int(from_roi.get(str(fire_id), train_val_label_sel))
        for split in splits
        for fire_id in ids[split]
    }


def fire_id_of(archive_name):
    """'train/p20562846.npz' -> '20562846' (the archive's fire ID)."""
    return os.path.basename(archive_name).removesuffix(".npz").removeprefix("p")
