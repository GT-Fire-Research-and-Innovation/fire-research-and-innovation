"""Hugging Face repository bookkeeping: what a repo holds and with which window settings.

Level 2 (hub). Used by the other hub/ modules.

Every window repo carries window_config.json: the settings that decided what
its archives contain (window length and stride, label rules, crop, channels,
format, split), plus a "notes" block for humans (what each label rule means,
compression level, code version). The archive file names encode none of this,
so the file is what keeps one repo to one set of settings.
"""

import json
import tempfile

from huggingface_hub import hf_hub_download

from firepred.config.channels import N_STORED_CHANNELS
from firepred.config.dataset import (
    ARCHIVE_FORMAT_VERSION,
    CROP_OFFSET,
    CROP_SIZE,
    NPZ_COMPRESSLEVEL,
    TEST_YEARS,
    TRAIN_VAL_LABEL_SEL,
    TRAIN_VAL_YEARS,
    VAL_IDS,
    WINDOW_CONFIG_FILE,
    window_repo_for,
)

# What each label_sel value means (prep/windows.build_fire_windows).
LABEL_SEL_MEANING = {
    "0": "burned area = accumulated active fire + NIFC perimeters (TS-SatFire paper)",
    "1": "burned area = accumulated active fire only, no NIFC perimeters (released TS-SatFire code)",
}


def list_fire_files(api, src_repo):
    """Scan source repository and group GeoTIFFs by fire and mode."""
    all_files = api.list_repo_files(
        src_repo,
        repo_type="dataset",
    )
    locations = {}

    for file_name in all_files:
        if not file_name.endswith(".tif"):
            continue

        # Paths end in <fire id>/<mode>/<file>.tif, where mode is the product
        # (e.g. VIIRS_Day); group them as {fire id: {mode: [files]}}.
        parts = file_name.split("/")
        if len(parts) < 3:
            continue

        location = parts[-3]
        mode = parts[-2]
        locations.setdefault(location, {}).setdefault(mode, []).append(
            file_name
        )

    # Sort by file name so every run sees each fire's files in the same order.
    for location in locations:
        for mode in locations[location]:
            locations[location][mode].sort()

    return locations


def window_settings(length, interval, train_val_label_sel=TRAIN_VAL_LABEL_SEL, use_roi_label_sel=True):
    """The settings that decide what a window repo's archives contain.

    test_label_sel is "roi" when each test fire uses the label_sel column of
    its ROI CSV (the paper's per-fire choice), else the one rule applied to
    every test fire.
    """
    return {
        "ts_length": int(length),
        "ts_interval": int(interval),
        "train_val_label_sel": int(train_val_label_sel),
        "test_label_sel": "roi" if use_roi_label_sel else int(train_val_label_sel),
        "crop_offset": CROP_OFFSET,
        "crop_size": CROP_SIZE,
        "stored_channels": N_STORED_CHANNELS,
        "archive_format_version": ARCHIVE_FORMAT_VERSION,
        "train_val_years": list(TRAIN_VAL_YEARS),
        "test_years": list(TEST_YEARS),
        "val_ids": list(VAL_IDS),
    }


def window_config_notes():
    """Human-readable context stored beside the settings; never compared."""
    from firepred.prep.tessera_grid import code_version  # git commit; lazy: pulls in rasterio

    return {
        "label_sel": LABEL_SEL_MEANING,
        "test_label_sel": '"roi": each test fire uses the label_sel column of the 2021 ROI CSV; '
                          "a number: every test fire uses that rule",
        "npz_compresslevel": NPZ_COMPRESSLEVEL,
        "code_version": code_version(),
    }


def read_window_config(api, window_repo, token=None):
    """Return the parsed window_config.json of `window_repo`, or None if it has none."""
    files = set(api.list_repo_files(window_repo, repo_type="dataset"))
    if WINDOW_CONFIG_FILE in files:
        # Fetch into a fresh temp folder so an old local copy is never read.
        path = hf_hub_download(
            repo_id=window_repo,
            filename=WINDOW_CONFIG_FILE,
            repo_type="dataset",
            token=token,
            local_dir=tempfile.mkdtemp(prefix="window_config_"),
        )
        with open(path) as handle:
            return json.load(handle)
    return None


def _upload_window_config(api, window_repo, config):
    api.upload_file(
        path_or_fileobj=json.dumps(config, indent=2).encode(),
        path_in_repo=WINDOW_CONFIG_FILE,
        repo_id=window_repo,
        repo_type="dataset",
    )


def ensure_window_config(api, window_repo, settings, overwrite=False, token=None):
    """Record `settings` (window_settings()) in `window_repo`, refusing to mix settings.

    - No window_config.json yet: write settings plus notes.
    - Existing file, no `overwrite`: never rewritten. A setting that differs
      raises ValueError; settings the file predates are reported and left out.
    - `overwrite` (process --overwrite, which also rebuilds every fire it
      processes): the file is replaced with this run's settings.
    """
    existing = read_window_config(api, window_repo, token)
    config = {**settings, "notes": window_config_notes()}
    if existing is None:
        _upload_window_config(api, window_repo, config)
        return

    # Compare with what the repo already records: settings that differ, and
    # settings the file predates (written by older code).
    changed = {
        key: (existing[key], value)
        for key, value in settings.items()
        if key in existing and existing[key] != value
    }
    missing = [key for key in settings if key not in existing]
    # Without --overwrite the file is never rewritten: a conflict stops the run
    # before anything is built; an older file is kept as it is.
    if not overwrite:
        if changed:
            details = ", ".join(f"{key}: repo has {old!r}, asked for {new!r}" for key, (old, new) in changed.items())
            raise ValueError(
                f"{window_repo} holds windows built with other settings ({details}). For other "
                f"window lengths use window_repo_for({settings['ts_length']}, {settings['ts_interval']}) = "
                f"{window_repo_for(settings['ts_length'], settings['ts_interval'])!r}; to rebuild this "
                "repo with the new settings, pass --overwrite."
            )
        if missing:
            print(
                f"{window_repo}/{WINDOW_CONFIG_FILE} predates {missing}; keeping it "
                "unchanged (pass --overwrite to rebuild and record them)"
            )
        return

    # --overwrite: replace the file with this run's settings.
    if changed:
        print(
            f"WARNING: overwriting {window_repo}/{WINDOW_CONFIG_FILE} ({sorted(changed)} change). "
            "Fires not rebuilt in this run keep archives made with the old settings."
        )
    _upload_window_config(api, window_repo, config)
