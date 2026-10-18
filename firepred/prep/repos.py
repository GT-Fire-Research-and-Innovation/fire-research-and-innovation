"""Hugging Face I/O: list, download and upload repo files; window_config.json.

Level 1 (prep). The only module that talks to Hugging Face. `huggingface_hub`
is imported inside each call (never at import time), so a notebook can pin
its cache folders after importing the masters. There is no global login:
every call takes the token explicitly. Every function returns what it
fetched or wrote.

Every window repo carries window_config.json: the settings that decided what
its archives contain (window length and stride, label rules, crop, channels,
format, split), plus a "notes" block for humans (what each label rule means,
compression level, code version). The archive file names encode none of this,
so the file is what keeps one repo to one set of settings.
"""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from firepred.config.data import (
    ARCHIVE_FORMAT_VERSION,
    CROP_OFFSET,
    CROP_SIZE,
    N_STORED_CHANNELS,
    NPZ_COMPRESSLEVEL,
    ROI_CACHE_DIR,
    TEST_YEARS,
    TRAIN_VAL_LABEL_SEL,
    TRAIN_VAL_YEARS,
    VAL_IDS,
    WINDOW_CONFIG_FILE,
    window_repo_for,
)
from firepred.prep.staging import unzip_into
from firepred.prep.tessera_grid import code_version

# What each label_sel value means (prep/windows.build_fire_windows).
LABEL_SEL_MEANING = {
    "0": "burned area = accumulated active fire + NIFC perimeters (TS-SatFire paper)",
    "1": "burned area = accumulated active fire only, no NIFC perimeters (released TS-SatFire code)",
}


def _hub():
    """The huggingface_hub module, imported on first use."""
    import huggingface_hub

    return huggingface_hub


def _api(token):
    """An HfApi client for `token`."""
    return _hub().HfApi(token=token)


# ----- Reading ----- #


def list_repo_files(repo_id, token=None):
    """Every file path in a dataset repo."""
    return list(_api(token).list_repo_files(repo_id, repo_type="dataset"))


def group_fire_files(file_names):
    """{fire id: {mode: [sorted .tif paths]}} from repo paths ending <fire id>/<mode>/<file>.tif."""
    locations = {}
    for file_name in file_names:
        parts = file_name.split("/")
        if not file_name.endswith(".tif") or len(parts) < 3:
            continue
        locations.setdefault(parts[-3], {}).setdefault(parts[-2], []).append(file_name)
    # Sorted, so every run sees each fire's files in the same order.
    return {
        location: {mode: sorted(paths) for mode, paths in modes.items()}
        for location, modes in locations.items()
    }


def list_fire_files(src_repo, token=None):
    """The source repo's GeoTIFFs grouped by fire and mode (see group_fire_files)."""
    return group_fire_files(list_repo_files(src_repo, token))


def download_file(repo_id, filename, token=None, local_dir=None, cache_dir=None):
    """Download one repo file and return its local path (str).

    With `local_dir` the file lands at local_dir/filename; otherwise in the
    Hugging Face cache (or `cache_dir`).
    """
    options = {"local_dir": str(local_dir)} if local_dir is not None else {}
    if cache_dir is not None:
        options["cache_dir"] = str(cache_dir)
    return _hub().hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="dataset", token=token, **options
    )


def download_roi_csvs(src_repo, token=None, roi_dir=ROI_CACHE_DIR):
    """Download every ROI CSV (fire metadata) of `src_repo` into `roi_dir`; return the local paths."""
    Path(roi_dir).mkdir(parents=True, exist_ok=True)
    return [
        download_file(src_repo, name, token, local_dir=roi_dir)
        for name in list_repo_files(src_repo, token)
        if name.endswith(".csv")
    ]


def pull_zip(repo_id, zip_name, data_root, token=None):
    """Download `zip_name` from `repo_id` into `data_root` and unpack it there; return data_root."""
    zip_path = download_file(repo_id, zip_name, token, local_dir=data_root)
    return unzip_into(zip_path, data_root)


# ----- Writing ----- #


def ensure_repo(repo_id, token=None, private=False):
    """Create the dataset repo if it does not exist (an existing one keeps its visibility); return its id."""
    _api(token).create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)
    return repo_id


def upload_folder(folder, repo_id, path_in_repo, token=None):
    """Upload every file under `folder` to repo_id/path_in_repo in one commit; return their names."""
    folder = Path(folder)
    names = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
    _api(token).upload_folder(
        folder_path=str(folder), repo_id=repo_id, repo_type="dataset", path_in_repo=path_in_repo
    )
    return names


def upload_file(path_or_bytes, path_in_repo, repo_id, token=None):
    """Upload one file (a path or bytes) to repo_id/path_in_repo; return path_in_repo."""
    source = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else str(path_or_bytes)
    _api(token).upload_file(
        path_or_fileobj=source, path_in_repo=path_in_repo, repo_id=repo_id, repo_type="dataset", token=token
    )
    return path_in_repo


# ----- window_config.json ----- #


@dataclass(frozen=True)
class WindowConfigDecision:
    """What ensure_window_config did: wrote the file or kept it, and why."""

    written: bool
    changed: tuple = ()
    missing: tuple = ()
    message: str = ""


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
    return {
        "label_sel": LABEL_SEL_MEANING,
        "test_label_sel": '"roi": each test fire uses the label_sel column of the 2021 ROI CSV; '
                          "a number: every test fire uses that rule",
        "npz_compresslevel": NPZ_COMPRESSLEVEL,
        "code_version": code_version(),
    }


def read_window_config(window_repo, token=None):
    """The parsed window_config.json of `window_repo`, or None if it has none."""
    if WINDOW_CONFIG_FILE not in set(list_repo_files(window_repo, token)):
        return None
    # Fetch into a fresh temp folder so an old local copy is never read.
    with tempfile.TemporaryDirectory(prefix="window_config_") as folder:
        path = download_file(window_repo, WINDOW_CONFIG_FILE, token, local_dir=folder)
        return json.loads(Path(path).read_text())


def decide_window_config(existing, settings, overwrite, window_repo):
    """Whether to (re)write window_config.json, given what the repo has. Pure.

    - No file yet: write.
    - Existing file, no `overwrite`: never rewritten. A setting that differs
      raises ValueError; settings the file predates are reported and left out.
    - `overwrite`: replace the file with these settings.
    """
    if existing is None:
        return WindowConfigDecision(written=True, message="new window_config.json")
    changed = tuple(sorted(key for key, value in settings.items() if key in existing and existing[key] != value))
    missing = tuple(key for key in settings if key not in existing)
    if overwrite:
        message = (
            f"WARNING: overwriting {window_repo}/{WINDOW_CONFIG_FILE} ({list(changed)} change). "
            "Fires not rebuilt in this run keep archives made with the old settings."
            if changed else f"rewriting {window_repo}/{WINDOW_CONFIG_FILE}"
        )
        return WindowConfigDecision(True, changed, missing, message)
    if changed:
        details = ", ".join(
            f"{key}: repo has {existing[key]!r}, asked for {settings[key]!r}" for key in changed
        )
        raise ValueError(
            f"{window_repo} holds windows built with other settings ({details}). For other "
            f"window lengths use window_repo_for({settings['ts_length']}, {settings['ts_interval']}) = "
            f"{window_repo_for(settings['ts_length'], settings['ts_interval'])!r}; to rebuild this "
            "repo with the new settings, pass --overwrite."
        )
    message = (
        f"{window_repo}/{WINDOW_CONFIG_FILE} predates {list(missing)}; keeping it unchanged "
        "(pass --overwrite to rebuild and record them)" if missing else "window_config.json matches"
    )
    return WindowConfigDecision(False, changed, missing, message)


def ensure_window_config(window_repo, settings, overwrite=False, token=None):
    """Record `settings` (window_settings()) in `window_repo`, refusing to mix settings.

    Returns the WindowConfigDecision; raises ValueError on a conflict without
    `overwrite` (see decide_window_config).
    """
    decision = decide_window_config(read_window_config(window_repo, token), settings, overwrite, window_repo)
    if decision.written:
        config = {**settings, "notes": window_config_notes()}
        upload_file(json.dumps(config, indent=2).encode(), WINDOW_CONFIG_FILE, window_repo, token)
    return decision
