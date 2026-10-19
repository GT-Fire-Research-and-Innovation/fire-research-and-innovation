"""Hugging Face I/O: list, download and upload repo files; window_config.json.

Level 1 (data, dataset building). The only module that talks to Hugging Face. `huggingface_hub`
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

from firepred.config.dataset import (
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
from firepred.data.files import code_version, unzip_into

# What each label_sel value means (prep/windows.build_fire_windows).
LABEL_SEL_MEANING = {
    "0": "burned area = accumulated active fire + NIFC perimeters (TS-SatFire paper)",
    "1": "burned area = accumulated active fire only, no NIFC perimeters (released TS-SatFire code)",
}


def _hub():
    """Import huggingface_hub the first time it is needed.

    Importing it lazily lets a notebook import the masters, point the Hugging
    Face cache somewhere else, and only then load the library. Every Hugging
    Face call in this module goes through here.

    Returns:
        module: the huggingface_hub module.
    """
    import huggingface_hub

    return huggingface_hub


def _api(token):
    """Create a Hugging Face API client for a token.

    There is no global login: each call builds a client with the token it was
    given. A None token works for public repos.

    Args:
        token (str | None): a Hugging Face token.

    Returns:
        huggingface_hub.HfApi: the client.
    """
    return _hub().HfApi(token=token)


# ----- Reading ----- #


def list_repo_files(repo_id, token=None):
    """List every file path in a Hugging Face dataset repo.

    Paths are relative to the repo root, e.g. "train/p123.npz". Listing reads no
    file contents.

    Args:
        repo_id (str): the repo, "owner/name".
        token (str | None): a Hugging Face token.

    Returns:
        list[str]: the file paths.
    """
    return list(_api(token).list_repo_files(repo_id, repo_type="dataset"))


def group_fire_files(file_names):
    """Group a source repo's GeoTIFF paths by fire and image type.

    Source paths end in <fire id>/<mode>/<file>.tif, where mode is the product
    (VIIRS_Day, VIIRS_Night, FirePred). Each fire's files are sorted, so every
    run sees the days in the same order. Non-GeoTIFF paths are ignored.

    Args:
        file_names (list[str]): repo file paths.

    Returns:
        dict[str, dict[str, list[str]]]: {fire id: {mode: [sorted paths]}}.
    """
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
    """List the source repo's GeoTIFFs grouped by fire and image type.

    This combines list_repo_files and group_fire_files. The masters use it to
    find each fire's daily images.

    Args:
        src_repo (str): the raw GeoTIFF repo.
        token (str | None): a Hugging Face token.

    Returns:
        dict[str, dict[str, list[str]]]: {fire id: {mode: [sorted paths]}}.
    """
    return group_fire_files(list_repo_files(src_repo, token))


def download_file(repo_id, filename, token=None, local_dir=None, cache_dir=None):
    """Download one file from a dataset repo.

    With local_dir the file lands at local_dir/filename (keeping its folders);
    otherwise it goes to the Hugging Face cache, or to cache_dir. Temporary
    downloads use cache_dir so they can be deleted afterwards.

    Args:
        repo_id (str): the repo, "owner/name".
        filename (str): the file path inside the repo.
        token (str | None): a Hugging Face token.
        local_dir (str | Path | None): a folder to download into.
        cache_dir (str | Path | None): a cache folder to use instead.

    Returns:
        str: the local path of the downloaded file.
    """
    options = {"local_dir": str(local_dir)} if local_dir is not None else {}
    if cache_dir is not None:
        options["cache_dir"] = str(cache_dir)
    return _hub().hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="dataset", token=token, **options
    )


def download_roi_csvs(src_repo, token=None, roi_dir=ROI_CACHE_DIR):
    """Download every ROI CSV of the source repo.

    The ROI CSVs list the fires, their dates and their label rules. They go
    into roi_dir and are read with fires.read_roi.

    Args:
        src_repo (str): the raw GeoTIFF repo, which also holds the CSVs.
        token (str | None): a Hugging Face token.
        roi_dir (str | Path): the local folder for the CSVs.

    Returns:
        list[str]: the local CSV paths.
    """
    Path(roi_dir).mkdir(parents=True, exist_ok=True)
    return [
        download_file(src_repo, name, token, local_dir=roi_dir)
        for name in list_repo_files(src_repo, token)
        if name.endswith(".csv")
    ]


def pull_zip(repo_id, zip_name, data_root, token=None):
    """Download a zip from a repo into a folder and unpack it there.

    Used for data.zip and tessera.zip. The zip's member paths are checked
    before unpacking, so nothing lands outside the folder.

    Args:
        repo_id (str): the repo holding the zip.
        zip_name (str): the zip's name in the repo, e.g. "data.zip".
        data_root (str | Path): the folder to download into and unpack in.
        token (str | None): a Hugging Face token.

    Returns:
        Path: data_root.
    """
    zip_path = download_file(repo_id, zip_name, token, local_dir=data_root)
    return unzip_into(zip_path, data_root)


# ----- Writing ----- #


def ensure_repo(repo_id, token=None, private=False):
    """Create a dataset repo if it does not exist yet.

    An existing repo is left as it is, including its visibility. A new repo is
    private only when asked.

    Args:
        repo_id (str): the repo, "owner/name".
        token (str | None): a token with write access.
        private (bool): make a newly created repo private.

    Returns:
        str: repo_id.
    """
    _api(token).create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)
    return repo_id


def upload_folder(folder, repo_id, path_in_repo, token=None):
    """Upload every file under a local folder to a repo folder in one commit.

    One commit per batch keeps the repo history short and uploads fast. The
    folder's layout is kept under path_in_repo.

    Args:
        folder (str | Path): the local folder.
        repo_id (str): the repo, "owner/name".
        path_in_repo (str): the repo folder to upload into, e.g. "train".
        token (str | None): a token with write access.

    Returns:
        list[str]: the uploaded file names, relative to folder.
    """
    folder = Path(folder)
    names = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
    _api(token).upload_folder(
        folder_path=str(folder), repo_id=repo_id, repo_type="dataset", path_in_repo=path_in_repo
    )
    return names


def upload_file(path_or_bytes, path_in_repo, repo_id, token=None):
    """Upload one file to a repo.

    The file can be a local path or bytes in memory (window_config.json is
    uploaded from memory). It replaces any file at the same repo path.

    Args:
        path_or_bytes (str | Path | bytes): the file or its contents.
        path_in_repo (str): where it goes in the repo.
        repo_id (str): the repo, "owner/name".
        token (str | None): a token with write access.

    Returns:
        str: path_in_repo.
    """
    source = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else str(path_or_bytes)
    _api(token).upload_file(
        path_or_fileobj=source, path_in_repo=path_in_repo, repo_id=repo_id, repo_type="dataset", token=token
    )
    return path_in_repo


# ----- window_config.json ----- #


@dataclass(frozen=True)
class WindowConfigDecision:
    """Record what ensure_window_config decided about a repo's window_config.json.

    The decision says whether the file was written, which recorded settings
    differ, and which ones the file predates. The message is what the master
    prints.

    Attributes:
        written (bool): True when the file was (re)written.
        changed (tuple[str, ...]): settings whose recorded value differs.
        missing (tuple[str, ...]): settings the existing file does not record.
        message (str): a human-readable explanation.
    """

    written: bool
    changed: tuple = ()
    missing: tuple = ()
    message: str = ""


def window_settings(length, interval, train_val_label_sel=TRAIN_VAL_LABEL_SEL, use_roi_label_sel=True):
    """Collect the settings that decide what a window repo's archives contain.

    Archive names record none of these, so the repo stores them in
    window_config.json and refuses windows built with other settings. The test
    label rule is "roi" when each test fire uses its CSV's rule.

    Args:
        length (int): input days per window.
        interval (int): days between window starts.
        train_val_label_sel (int): the train/val label rule (0 or 1).
        use_roi_label_sel (bool): whether test fires use their CSV's rule.

    Returns:
        dict: the settings (window length and stride, label rules, crop, channels, format, split).
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
    """Collect the human-readable notes stored beside the window settings.

    The notes explain the label rules and record the zlib level and code
    version. They are never compared, only kept for people reading the file.

    Returns:
        dict: the notes.
    """
    return {
        "label_sel": LABEL_SEL_MEANING,
        "test_label_sel": '"roi": each test fire uses the label_sel column of the 2021 ROI CSV; '
                          "a number: every test fire uses that rule",
        "npz_compresslevel": NPZ_COMPRESSLEVEL,
        "code_version": code_version(),
    }


def read_window_config(window_repo, token=None):
    """Read a window repo's window_config.json.

    The file is downloaded into a temporary folder, so an old local copy is
    never read, and the folder is removed afterwards. A repo without the file
    gives None.

    Args:
        window_repo (str): the window repo.
        token (str | None): a Hugging Face token.

    Returns:
        dict | None: the parsed settings, or None.
    """
    if WINDOW_CONFIG_FILE not in set(list_repo_files(window_repo, token)):
        return None
    # Fetch into a fresh temp folder so an old local copy is never read.
    with tempfile.TemporaryDirectory(prefix="window_config_") as folder:
        path = download_file(window_repo, WINDOW_CONFIG_FILE, token, local_dir=folder)
        return json.loads(Path(path).read_text())


def decide_window_config(existing, settings, overwrite, window_repo):
    """Decide whether to write window_config.json, given what the repo already has.

    A repo without the file gets it. An existing file is never rewritten
    without overwrite: a differing setting is an error, and settings the file
    predates are reported and left out. This function only decides; it reads
    and writes nothing.

    Args:
        existing (dict | None): the repo's current settings (read_window_config).
        settings (dict): this run's settings (window_settings).
        overwrite (bool): replace the file with this run's settings.
        window_repo (str): the repo name, for messages.

    Returns:
        WindowConfigDecision: whether to write, and why.

    Raises:
        ValueError: if a setting differs and overwrite is False.
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
    """Record a run's window settings in a repo, refusing to mix settings.

    The repo's current file is read, decide_window_config decides, and the
    file (settings plus notes) is uploaded when needed. This runs before any
    fire is built, so a conflicting repo fails at once.

    Args:
        window_repo (str): the window repo.
        settings (dict): this run's settings (window_settings).
        overwrite (bool): replace the file with this run's settings.
        token (str | None): a token with write access.

    Returns:
        WindowConfigDecision: what was done, and why.

    Raises:
        ValueError: if a setting differs and overwrite is False.
    """
    decision = decide_window_config(read_window_config(window_repo, token), settings, overwrite, window_repo)
    if decision.written:
        config = {**settings, "notes": window_config_notes()}
        upload_file(json.dumps(config, indent=2).encode(), WINDOW_CONFIG_FILE, window_repo, token)
    return decision
