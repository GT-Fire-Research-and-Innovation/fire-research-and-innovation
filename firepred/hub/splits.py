"""Which fires go where: ROI metadata, train/val/test split, label rule, start years.

Level 2 (hub). Used by hub/publish_windows.py, hub/publish_tessera.py and
hub/download.py. The split [released]: train/val are the 2017-2020 fires
(validation is the fixed VAL_IDS list), test is the 2021 fires.
"""

import os

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from firepred.config.dataset import SPLITS, SRC_REPO, TEST_YEARS, TRAIN_VAL_YEARS, VAL_IDS
from firepred.prep.tessera_grid import has_prior_year_tessera


def download_roi_csvs(src_repo=SRC_REPO, token=None, roi_dir="hf_roi"):
    """Download every ROI CSV (fire metadata) from `src_repo` into `roi_dir`."""
    os.makedirs(roi_dir, exist_ok=True)

    # obtains all roi files (metadata that maps onto fires)
    # need to pull them to get a dataframe of ids for training.
    api = HfApi(token=token)
    files = []
    for file_name in api.list_repo_files(src_repo, repo_type="dataset"):
        if file_name.endswith(".csv"):
            files.append(file_name)

    local_paths = []
    for file_name in files:
        local_path = hf_hub_download(
            filename=file_name,
            repo_id=src_repo,
            repo_type="dataset",
            local_dir=roi_dir,
            token=token,
        )
        local_paths.append(local_path)

    return local_paths


def fire_label_sel(roi_paths):
    """Map fire Id -> label_sel from every ROI CSV that has the column."""
    label_sel = {}
    for path in roi_paths:
        df = pd.read_csv(path, dtype={"Id": str})
        if "Id" not in df.columns or "label_sel" not in df.columns:
            continue
        # CSVs without the column are skipped; a fire listed twice must agree.
        for fire_id, value in zip(df["Id"], df["label_sel"]):
            if pd.isna(value):
                continue
            fire_id, value = str(fire_id), int(value)
            if label_sel.setdefault(fire_id, value) != value:
                raise ValueError(f"ROI CSVs disagree on label_sel for {fire_id}")
    return label_sel


def obtain_ids(src_repo=SRC_REPO, token=None, roi_dir="hf_roi"):
    """Download the ROI CSVs from `src_repo` and split fire IDs.

    Train/val come from 2017-2020 fires (val is the fixed VAL_IDS list);
    test comes from 2021 fires.
    """
    local_paths = download_roi_csvs(src_repo, token, roi_dir)

    # --- training and validation dataframe ---
    dfs = []
    for path in local_paths:
        print(path)
        if any(year in os.path.basename(path) for year in TRAIN_VAL_YEARS):
            dfs.append(pd.read_csv(path, dtype={'Id': str}))

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(by=['Id'])
    df['Id'] = df['Id'].astype(str)

    # Validation = the fixed VAL_IDS list; every other 2017-2020 fire trains.
    train_df = df[~df.Id.isin(VAL_IDS)]
    val_df = df[df.Id.isin(VAL_IDS)]

    train_ids = train_df['Id'].values.astype(str)
    val_ids = val_df['Id'].values.astype(str)

    # --- testing dataframe ---
    dfs_test = []
    for path in local_paths:
        print(path)
        if any(year in os.path.basename(path) for year in TEST_YEARS):
            dfs_test.append(pd.read_csv(path, dtype={'Id': str}))

    df_test = pd.concat(dfs_test, ignore_index=True)
    test_ids = df_test['Id'].values.astype(str)

    # validate that the vars are initialized
    print(f"Train: {len(train_ids)}")
    print(f"Val: {len(val_ids)}")
    print(f"Test: {len(test_ids)}")

    return train_ids, val_ids, test_ids


def fire_start_years(roi_paths):
    """Map fire Id -> start year from the ROI CSVs."""
    years = {}
    for path in roi_paths:
        df = pd.read_csv(path, dtype={"Id": str})
        if "Id" not in df.columns or "start_date" not in df.columns:
            continue
        for fire_id, start_date in zip(df["Id"], df["start_date"]):
            if pd.notna(start_date):
                years[str(fire_id)] = int(pd.to_datetime(start_date).year)
    return years


def tessera_excluded_npz(token=None, src_repo=SRC_REPO, roi_dir="hf_roi"):
    """Window archives to drop in TESSERA mode, e.g. ['train/p20778153.npz', ...].

    These are the fires with no prior-year TESSERA (all 2017 fires). Dropping
    them from both the TESSERA and the matched no-TESSERA runs keeps the
    comparison on the same fires.
    """
    years = fire_start_years(download_roi_csvs(src_repo, token, roi_dir))
    ids_by_split = dict(zip(SPLITS, obtain_ids(src_repo, token, roi_dir)))
    return sorted(
        f"{split}/p{fire_id}.npz"
        for split, fire_ids in ids_by_split.items()
        for fire_id in fire_ids
        if not has_prior_year_tessera(years.get(str(fire_id)))
    )
