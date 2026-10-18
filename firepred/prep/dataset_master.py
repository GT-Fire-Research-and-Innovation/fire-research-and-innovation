"""MASTER for datasets: every window-data step a notebook or the CLI runs.

Level 1 (prep). Notebooks call only this module and prep/tessera_master.py.
Each function runs one pipeline step end to end and returns a frozen value
(reports have .summary()). Progress is printed here and nowhere below.

    workspace           every local path a notebook uses, from a few settings
    pin_caches          keep Hugging Face / temp caches under the workspace
    read_secret         a token from Kaggle secrets, Colab userdata or the environment
    process_windows     raw GeoTIFFs -> one window archive per fire in the window repo
    download_windows    window archives (+ TESSERA files) into a local data root
    upload_data_zip     zip a data root and upload it as data.zip
    rebuild_data_zip    download a window repo and upload it as data.zip
    pull_data_zip       download data.zip and unpack it into a data root
    stage_training_data training notebooks: find/download/extract data (+ TESSERA view)

    SRC_REPO raw GeoTIFFs --one fire at a time--> prep/windows.build_fire_windows
        --> data/archive.write_fire_archive p<id>.npz --uploaded in batches--> window repo
"""

import gc
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from firepred.config.data import (
    DATA_DIR,
    DATA_ZIP,
    ROI_CACHE_DIR,
    SPLITS,
    SRC_REPO,
    TESSERA_REPO,
    TRAIN_VAL_LABEL_SEL,
    TS_INTERVAL,
    TS_LENGTH,
    WINDOW_UPLOAD_BATCH,
    WORK_DIR,
    ZIP_FOLDERS,
    window_repo_for,
)
from firepred.data.archive import write_fire_archive
from firepred.prep import repos, staging, tessera_master
from firepred.prep.roi import fire_id_of, label_rules, read_roi, tessera_excluded_npz
from firepred.prep.windows import build_fire_windows


# ----- Paths and credentials ----- #


@dataclass(frozen=True)
class Workspace:
    """Every local path the notebooks use, derived from one work folder and the window settings."""

    work_dir: Path
    ts_length: int
    ts_interval: int
    use_tessera: bool
    window_repo: str
    data_root: Path            # data-grab notebook: built/downloaded window data
    roi_dir: Path              # downloaded ROI CSVs
    cache_dir: Path            # Hugging Face and temp caches (pin_caches)
    download_dir: Path         # downloaded zips
    extract_dir: Path          # training notebooks: unpacked data.zip
    tessera_extract_dir: Path  # unpacked tessera.zip
    view_dir: Path             # symlinked TESSERA training root


def workspace(work_dir, ts_length=TS_LENGTH, ts_interval=TS_INTERVAL, use_tessera=False, window_repo=None):
    """A Workspace under `work_dir` for windows of `ts_length` days every `ts_interval` days."""
    work = Path(work_dir)
    tag = f"ts{ts_length}-it{ts_interval}"
    return Workspace(
        work_dir=work,
        ts_length=ts_length,
        ts_interval=ts_interval,
        use_tessera=use_tessera,
        window_repo=window_repo or window_repo_for(ts_length, ts_interval),
        data_root=work / (f"data_ts{ts_length}_it{ts_interval}" + ("_tessera" if use_tessera else "")),
        roi_dir=work / "roi",
        cache_dir=work / "cache",
        download_dir=work / "downloads",
        extract_dir=work / f"data-{tag}",
        tessera_extract_dir=work / "tessera",
        view_dir=work / f"data-{tag}-tessera-view",
    )


def pin_caches(cache_dir):
    """Point HF_HOME, XDG_CACHE_HOME and TMPDIR into `cache_dir`; return the variables set.

    Call before anything imports huggingface_hub or geotessera (the masters
    import neither at import time), so large caches stay on the work disk.
    """
    cache_dir = Path(cache_dir)
    settings = {
        "HF_HOME": str(cache_dir / "huggingface"),
        "XDG_CACHE_HOME": str(cache_dir),
        "TMPDIR": str(cache_dir / "tmp"),
    }
    Path(settings["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    os.environ.update(settings)
    tempfile.tempdir = None  # make tempfile re-read TMPDIR
    return settings


def read_secret(label, env_var=None):
    """A secret from Kaggle secrets, else Colab userdata, else the environment; None when absent."""
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret(label)
    except Exception:
        pass
    try:
        from google.colab import userdata

        return userdata.get(label)
    except Exception:
        pass
    return os.environ.get(env_var or label)


# ----- Reports ----- #


@dataclass(frozen=True)
class WindowBuildReport:
    """process_windows' result: fires built per split, fires skipped, files uploaded."""

    repo: str
    built: dict
    skipped: dict = field(default_factory=dict)
    uploaded: tuple = ()
    config: str = ""

    def summary(self):
        """Counts per split plus skipped fires."""
        counts = ", ".join(f"{split} {len(ids)}" for split, ids in self.built.items())
        lines = [f"Windows in {self.repo}: built {counts}; skipped {len(self.skipped)}", self.config]
        lines += [f"  {fire_id}: {reason}" for fire_id, reason in self.skipped.items()]
        return "\n".join(line for line in lines if line)


@dataclass(frozen=True)
class DownloadReport:
    """download_windows' result: what was fetched, already there, skipped or excluded."""

    data_root: Path
    downloaded: tuple
    present: tuple
    skipped: tuple
    excluded: tuple
    missing_tessera: tuple = ()
    window_config: dict = None

    def summary(self):
        """Counts, plus a warning for archives without a TESSERA file."""
        text = (f"{self.data_root}: downloaded {len(self.downloaded)}, already present {len(self.present)}, "
                f"skipped {len(self.skipped)}, excluded {len(self.excluded)}")
        if self.missing_tessera:
            text += f"\nWARNING: {len(self.missing_tessera)} window archives have no TESSERA file: " \
                    f"{list(self.missing_tessera)}"
        return text


@dataclass(frozen=True)
class ZipUpload:
    """upload_data_zip's result: the local zip, its members and where it went."""

    zip_path: Path
    members: tuple
    repo: str
    path_in_repo: str

    def summary(self):
        """One line."""
        return f"Uploaded {self.zip_path} ({len(self.members)} files) to {self.repo}/{self.path_in_repo}"


@dataclass(frozen=True)
class StagedData:
    """stage_training_data's result: the extracted data, the training root and its archive counts."""

    data_root: Path
    train_root: Path
    tessera_dir: Path = None
    counts: dict = field(default_factory=dict)

    def summary(self):
        """Roots and archive counts per split."""
        counts = ", ".join(f"{split} {n}" for split, n in self.counts.items())
        return f"Data root: {self.data_root} | training root: {self.train_root} | archives: {counts}"


# ----- Building windows ----- #


def build_and_stage(fire_id, fire_files, label_sel, split_dir, src_repo, token, length, interval, temp_root, debug):
    """Download one fire's GeoTIFFs, build its windows and write p<id>.npz into `split_dir`.

    Returns (archive path or None when the fire has no windows, the build notes).
    The downloads live in a temporary folder deleted afterwards.
    """
    with tempfile.TemporaryDirectory(dir=temp_root) as temp_dir:
        local_map = {
            path: repos.download_file(src_repo, path, token, cache_dir=temp_dir)
            for paths in fire_files.values()
            for path in paths
        }
        windows = build_fire_windows(fire_id, local_map, length=length, interval=interval,
                                     label_sel=label_sel, debug=debug)
    if windows.count == 0:
        return None, windows.notes
    path = write_fire_archive(Path(split_dir) / f"p{fire_id}.npz", windows.inputs, windows.targets,
                              label_sel=label_sel)
    return path, windows.notes


def upload_split_batch(split_dir, split, window_repo, token):
    """Upload the staged archives of one split, then empty its staging folder; return the names."""
    print(f"Uploading staged {split} archives to {window_repo}")
    names = repos.upload_folder(split_dir, window_repo, split, token)
    shutil.rmtree(split_dir, ignore_errors=True)
    Path(split_dir).mkdir(parents=True, exist_ok=True)
    return [f"{split}/{name}" for name in names]


def process_windows(
    token,
    src_repo=SRC_REPO,
    window_repo=None,
    batch_size=WINDOW_UPLOAD_BATCH,
    length=TS_LENGTH,
    interval=TS_INTERVAL,
    splits=SPLITS,
    roi_dir=ROI_CACHE_DIR,
    work_dir=WORK_DIR,
    overwrite=False,
    debug=False,
    use_roi_label_sel=True,
    train_val_label_sel=TRAIN_VAL_LABEL_SEL,
    only_fires=None,
):
    """Stream raw fires from `src_repo` into windows of `length` input days in `window_repo`.

    A new window starts every `interval` days. `window_repo` defaults to
    window_repo_for(length, interval), so different settings never share a
    repo; its window_config.json is checked (or written) before any fire is
    built. Targets follow the TS-SatFire paper: train/val fires use
    `train_val_label_sel`, test fires their ROI label_sel
    (`use_roi_label_sel=False`: `train_val_label_sel` for every fire). Fires
    already in the repo are skipped unless `overwrite`; `only_fires` limits
    the run to those IDs (debugging). Archives are uploaded `batch_size` at a time.
    """
    window_repo = window_repo or window_repo_for(length, interval)
    print(f"Windows: {length} input days every {interval} days -> {window_repo}")
    repos.ensure_repo(window_repo, token)
    decision = repos.ensure_window_config(
        window_repo, repos.window_settings(length, interval, train_val_label_sel, use_roi_label_sel),
        overwrite=overwrite, token=token,
    )
    print(decision.message)

    roi = read_roi(repos.download_roi_csvs(src_repo, token, roi_dir))
    rules = label_rules(roi, splits, use_roi_label_sel, train_val_label_sel)
    locations = repos.list_fire_files(src_repo, token)
    existing = set(repos.list_repo_files(window_repo, token))
    wanted = None if only_fires is None else {str(fire_id) for fire_id in only_fires}

    work_dir = Path(work_dir)
    temp_root = work_dir / "temp"
    temp_root.mkdir(parents=True, exist_ok=True)
    built, skipped, uploaded = {}, {}, []
    for split in splits:
        split_dir = work_dir / "staging" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        fire_ids = [fire_id for fire_id in roi.ids_by_split()[split]
                    if fire_id in locations and (wanted is None or fire_id in wanted)]
        print(f"{split}: {len(fire_ids)} fires with GeoTIFFs")
        built[split], staged = [], 0
        for number, fire_id in enumerate(fire_ids, start=1):
            if f"{split}/p{fire_id}.npz" in existing and not overwrite:
                print(f"Skipping {fire_id}: already processed")
                continue
            print(f"[{split.upper()} | {number}/{len(fire_ids)}] Processing: {fire_id}")
            path, notes = build_and_stage(fire_id, locations[fire_id], rules.get(fire_id, train_val_label_sel),
                                          split_dir, src_repo, token, length, interval, temp_root, debug)
            print("\n".join(notes))
            if path is None:
                skipped[fire_id] = "no complete window"
            else:
                built[split].append(fire_id)
                staged += 1
            gc.collect()
            if staged >= batch_size:
                uploaded += upload_split_batch(split_dir, split, window_repo, token)
                staged = 0
        if staged > 0:
            uploaded += upload_split_batch(split_dir, split, window_repo, token)
        shutil.rmtree(split_dir, ignore_errors=True)

    report = WindowBuildReport(window_repo, {s: tuple(ids) for s, ids in built.items()}, skipped,
                               tuple(uploaded), decision.message)
    print(report.summary())
    return report


# ----- Moving window data ----- #


def download_windows(
    token,
    window_repo=None,
    src_repo=SRC_REPO,
    data_root=DATA_DIR,
    roi_dir=ROI_CACHE_DIR,
    include_tessera=False,
    tessera_repo=TESSERA_REPO,
    length=TS_LENGTH,
    interval=TS_INTERVAL,
):
    """Download every window archive of `window_repo` into <data_root>/{train,val,test}.

    Each archive goes to the split its fire ID belongs to (from the ROI CSVs),
    files already on disk are kept. With `include_tessera`, fires without
    prior-year TESSERA are left out and per-fire embeddings are downloaded
    from `tessera_repo` into <data_root>/tessera. Use a different `data_root`
    per mode so a full download never mixes with a TESSERA one.
    """
    window_repo = window_repo or window_repo_for(length, interval)
    data_root = Path(data_root)
    window_config = repos.read_window_config(window_repo, token)
    print(f"Window repo {window_repo}: {window_config}")
    roi = read_roi(repos.download_roi_csvs(src_repo, token, roi_dir))
    excluded = tessera_excluded_npz(roi) if include_tessera else []
    split_of = roi.split_of()
    for split in SPLITS:
        (data_root / split).mkdir(parents=True, exist_ok=True)

    downloaded, present, skipped = [], [], []
    for name in repos.list_repo_files(window_repo, token):
        # Only split folders hold window archives (tessera/p<id>.npz shares their names).
        if name.split("/")[0] not in SPLITS or not name.endswith(".npz"):
            continue
        if name in excluded:
            skipped.append(name)
            continue
        split = split_of.get(fire_id_of(name))
        if split is None:
            skipped.append(name)
            continue
        target = data_root / split / Path(name).name
        if target.exists():
            present.append(str(target.relative_to(data_root)))
            continue
        shutil.move(repos.download_file(window_repo, name, token, local_dir=data_root), target)
        downloaded.append(str(target.relative_to(data_root)))

    missing = ()
    if include_tessera:
        missing = tuple(download_tessera_files(data_root, tessera_repo, token))
    report = DownloadReport(data_root, tuple(downloaded), tuple(present), tuple(skipped), tuple(excluded),
                            missing, window_config)
    print(report.summary())
    return report


def download_tessera_files(data_root, tessera_repo, token):
    """Download tessera/* into <data_root>/tessera; return window archives that have no TESSERA file."""
    data_root = Path(data_root)
    for name in repos.list_repo_files(tessera_repo, token):
        if name.startswith("tessera/") and not (data_root / name).exists():
            repos.download_file(tessera_repo, name, token, local_dir=data_root)
    return [
        f"{split}/{archive.name}"
        for split in SPLITS
        for archive in sorted((data_root / split).glob("*.npz"))
        if not (data_root / "tessera" / archive.name).exists()
    ]


def upload_data_zip(token, data_root=DATA_DIR, zip_path=None, repo_id=None, zip_name=DATA_ZIP, folders=ZIP_FOLDERS):
    """Zip <data_root>/<folder> for each folder and upload it to `repo_id` as `zip_name`; return a ZipUpload.

    TESSERA mode uses repo_id=TESSERA_REPO, zip_name="data_tessera.zip", so
    the full data.zip in the window repo is never replaced by the reduced set.
    """
    repo_id = repo_id or window_repo_for()
    zip_path = Path(zip_path) if zip_path else Path(data_root).parent / zip_name
    members = staging.zip_folders([Path(data_root) / folder for folder in folders], zip_path)
    repos.upload_file(zip_path, zip_name, repo_id, token)
    report = ZipUpload(zip_path, tuple(members), repo_id, zip_name)
    print(report.summary())
    return report


def rebuild_data_zip(token, window_repo=None, length=TS_LENGTH, interval=TS_INTERVAL, src_repo=SRC_REPO,
                     work_dir=WORK_DIR, roi_dir=ROI_CACHE_DIR):
    """Download every archive of the window repo and upload it back as data.zip; return a ZipUpload."""
    window_repo = window_repo or window_repo_for(length, interval)
    zip_root = Path(work_dir) / f"zip_ts{length}_it{interval}"
    download_windows(token, window_repo=window_repo, src_repo=src_repo, data_root=zip_root, roi_dir=roi_dir)
    return upload_data_zip(token, data_root=zip_root, zip_path=f"{zip_root}.zip", repo_id=window_repo)


def pull_data_zip(data_root, token=None, repo_id=None, zip_name=DATA_ZIP):
    """Download `zip_name` from `repo_id` into `data_root` and unpack it there; return data_root."""
    data_root = repos.pull_zip(repo_id or window_repo_for(), zip_name, data_root, token)
    print(f"Extracted {zip_name} into {data_root}")
    return Path(data_root)


def stage_training_data(work, token=None, window_repo=None, tessera_repo=TESSERA_REPO,
                        attached_data_root=None, attached_zip=None,
                        attached_tessera_dir=None, attached_tessera_zip=None):
    """Training notebooks: make the data (and TESSERA view) ready; return StagedData.

    Data priority: an attached extracted folder, an attached data.zip, else
    data.zip from `window_repo` (default: the workspace's window repo).
    In TESSERA mode (work.use_tessera) the training root is a symlink view
    without the excluded fires (tessera_master.stage_tessera).
    """
    if attached_data_root:
        data_root = staging.find_data_root(attached_data_root)
    else:
        archive = Path(attached_zip) if attached_zip else Path(
            repos.download_file(window_repo or work.window_repo, DATA_ZIP, token, local_dir=work.download_dir)
        )
        data_root = staging.extract_data(archive, work.extract_dir)

    tessera_dir, train_root = None, data_root
    if work.use_tessera:
        view = tessera_master.stage_tessera(data_root, work, token, tessera_repo,
                                            attached_tessera_dir, attached_tessera_zip)
        train_root, tessera_dir = view.root, view.root / "tessera"
    staged = StagedData(data_root, train_root, tessera_dir, staging.count_archives(train_root))
    print(staged.summary())
    return staged
