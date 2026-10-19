"""MASTER for datasets: every window-data step a notebook or the CLI runs.

Level 1 (data, dataset building). Notebooks call only this module and data/tessera_master.py.
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

from firepred.config.dataset import (
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
from firepred.data import files, hub, tessera_master
from firepred.data.fires import fire_id_of, label_rules, read_roi
from firepred.data.tessera import tessera_excluded_npz
from firepred.data.windows import build_fire_windows


# ----- Paths and credentials ----- #


@dataclass(frozen=True)
class Workspace:
    """Hold every local path a notebook uses, derived from one work folder.

    The data-grab and training notebooks used to build these paths by hand.
    workspace() derives them all from a work folder and the window settings, so
    the notebooks only choose the folder.

    Attributes:
        work_dir (Path): the work folder everything else lives under.
        ts_length (int): input days per window.
        ts_interval (int): days between window starts.
        use_tessera (bool): TESSERA mode.
        window_repo (str): the window repo for these settings.
        data_root (Path): data-grab notebook: built or downloaded window data.
        roi_dir (Path): downloaded ROI CSVs.
        cache_dir (Path): Hugging Face and temporary caches (pin_caches).
        download_dir (Path): downloaded zips.
        extract_dir (Path): training notebooks: the unpacked data.zip.
        tessera_extract_dir (Path): the unpacked tessera.zip.
        view_dir (Path): the symlinked TESSERA training root.
    """

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
    """Derive a Workspace under a work folder for one window setting.

    Folder names include the window settings, so data built with different
    settings never shares a folder. The window repo defaults to the one for
    these settings.

    Args:
        work_dir (str | Path): the work folder, e.g. "/content" or "/kaggle/temp/fire-research".
        ts_length (int): input days per window.
        ts_interval (int): days between window starts.
        use_tessera (bool): TESSERA mode.
        window_repo (str | None): another window repo (default: window_repo_for).

    Returns:
        Workspace: the paths.
    """
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
    """Keep the Hugging Face and temporary caches under the work folder.

    Large downloads should land on the notebook's big work disk, not the small
    system disk. Call this before huggingface_hub is first used (the masters
    import it only on first use).

    Args:
        cache_dir (str | Path): the cache folder.

    Returns:
        dict[str, str]: the environment variables set (HF_HOME, XDG_CACHE_HOME, TMPDIR).
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
    """Read a secret (an API token) wherever the notebook runs.

    Kaggle secrets are tried first, then Colab's userdata, then an environment
    variable. A missing secret gives None, which works for public data.

    Args:
        label (str): the secret's name, e.g. "HF_TOKEN".
        env_var (str | None): the environment variable to fall back to (default: label).

    Returns:
        str | None: the secret, or None.
    """
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
    """Report what process_windows built, skipped and uploaded.

    The master returns this instead of only printing, so a notebook or test
    can inspect the outcome. Fires without a complete window are listed as
    skipped.

    Attributes:
        repo (str): the window repo.
        built (dict[str, tuple[str, ...]]): {split: fire ids built}.
        skipped (dict[str, str]): {fire id: why it has no archive}.
        uploaded (tuple[str, ...]): "split/p<id>.npz" names uploaded.
        config (str): what was decided about window_config.json.
    """

    repo: str
    built: dict
    skipped: dict = field(default_factory=dict)
    uploaded: tuple = ()
    config: str = ""

    def summary(self):
        """Summarize the build: counts per split and the skipped fires.

        The master prints this at the end. Each skipped fire is listed with its
        reason.

        Returns:
            str: the summary, one item per line.
        """
        counts = ", ".join(f"{split} {len(ids)}" for split, ids in self.built.items())
        lines = [f"Windows in {self.repo}: built {counts}; skipped {len(self.skipped)}", self.config]
        lines += [f"  {fire_id}: {reason}" for fire_id, reason in self.skipped.items()]
        return "\n".join(line for line in lines if line)


@dataclass(frozen=True)
class DownloadReport:
    """Report what download_windows fetched, kept, skipped and excluded.

    Archives already on disk are kept, not downloaded again. In TESSERA mode the
    report also lists archives still without a TESSERA file.

    Attributes:
        data_root (Path): where the archives went.
        downloaded (tuple[str, ...]): "split/p<id>.npz" archives downloaded.
        present (tuple[str, ...]): archives already on disk.
        skipped (tuple[str, ...]): repo files not placed (excluded or unknown fire).
        excluded (tuple[str, ...]): archives TESSERA mode leaves out.
        missing_tessera (tuple[str, ...]): archives without a TESSERA file.
        window_config (dict | None): the repo's window_config.json.
    """

    data_root: Path
    downloaded: tuple
    present: tuple
    skipped: tuple
    excluded: tuple
    missing_tessera: tuple = ()
    window_config: dict = None

    def summary(self):
        """Summarize the download, with a warning for archives without TESSERA.

        The first line gives the counts. A second line warns when TESSERA files
        are missing.

        Returns:
            str: the summary.
        """
        text = (f"{self.data_root}: downloaded {len(self.downloaded)}, already present {len(self.present)}, "
                f"skipped {len(self.skipped)}, excluded {len(self.excluded)}")
        if self.missing_tessera:
            text += f"\nWARNING: {len(self.missing_tessera)} window archives have no TESSERA file: " \
                    f"{list(self.missing_tessera)}"
        return text


@dataclass(frozen=True)
class ZipUpload:
    """Report what upload_data_zip zipped and where it went.

    The local zip stays on disk. Its members keep their split folders.

    Attributes:
        zip_path (Path): the local zip.
        members (tuple[str, ...]): the zipped files.
        repo (str): the repo it was uploaded to.
        path_in_repo (str): its name in the repo.
    """

    zip_path: Path
    members: tuple
    repo: str
    path_in_repo: str

    def summary(self):
        """Summarize the upload in one line.

        The line names the zip, how many files it holds and where it went. The
        master prints it.

        Returns:
            str: the summary.
        """
        return f"Uploaded {self.zip_path} ({len(self.members)} files) to {self.repo}/{self.path_in_repo}"


@dataclass(frozen=True)
class StagedData:
    """Report what stage_training_data prepared for training.

    The training root is what the trainer reads: the data root itself, or in
    TESSERA mode a symlinked view without the excluded fires. The counts let a
    notebook check the data at a glance.

    Attributes:
        data_root (Path): the extracted window data.
        train_root (Path): the folder to train from.
        tessera_dir (Path | None): the TESSERA files (TESSERA mode only).
        counts (dict[str, int]): archives per split in train_root.
    """

    data_root: Path
    train_root: Path
    tessera_dir: Path = None
    counts: dict = field(default_factory=dict)

    def summary(self):
        """Summarize the staged data in one line.

        The line gives both roots and the archive count of each split. The master
        prints it.

        Returns:
            str: the summary.
        """
        counts = ", ".join(f"{split} {n}" for split, n in self.counts.items())
        return f"Data root: {self.data_root} | training root: {self.train_root} | archives: {counts}"


# ----- Planning (pure) ----- #


@dataclass(frozen=True)
class WindowJob:
    """Describe one fire that process_windows will build.

    The job carries everything needed to build the fire. It says where the
    images are, which split the fire goes to and which label rule to use.

    Attributes:
        split (str): "train", "val" or "test".
        fire_id (str): the fire id.
        label_sel (int): the label rule (0 or 1).
        files (dict[str, list[str]]): {mode: [source repo GeoTIFF paths]}.
    """

    split: str
    fire_id: str
    label_sel: int
    files: dict


@dataclass(frozen=True)
class WindowPlan:
    """Describe what process_windows will do before it downloads anything.

    The plan lists the fires to build and the fires left alone, with why.
    Building it touches no file or network, so it can be tested on its own.

    Attributes:
        jobs (tuple[WindowJob, ...]): the fires to build, in order.
        already_built (tuple[str, ...]): "split/p<id>.npz" archives already in the repo.
        without_geotiffs (tuple[str, ...]): fires with no images in the source repo.
    """

    jobs: tuple
    already_built: tuple = ()
    without_geotiffs: tuple = ()

    def jobs_by_split(self, splits):
        """Group the planned jobs by split.

        process_windows builds and uploads one split at a time. Every requested
        split appears, possibly with no jobs.

        Args:
            splits (tuple[str, ...]): the splits, in build order.

        Returns:
            dict[str, tuple[WindowJob, ...]]: {split: its jobs, in plan order}.
        """
        return {split: tuple(job for job in self.jobs if job.split == split) for split in splits}

    def summary(self):
        """Summarize the plan in one line.

        The line counts the fires to build and those left alone. The master prints
        it before starting.

        Returns:
            str: the summary.
        """
        return (f"Plan: build {len(self.jobs)} fires; {len(self.already_built)} already in the repo, "
                f"{len(self.without_geotiffs)} without GeoTIFFs in the source repo")


def plan_windows(roi, rules, locations, existing, splits, overwrite=False, only_fires=None,
                 default_label_sel=TRAIN_VAL_LABEL_SEL):
    """Plan which fires to build: every fire with images that is not built yet.

    Fires already in the window repo are left alone unless overwrite is set,
    and fires with no images in the source repo are listed separately. This is
    pure: it reads and writes nothing.

    Args:
        roi (RoiTable): the fires of every split.
        rules (dict[str, int]): {fire id: label rule} (fires.label_rules).
        locations (dict): the source repo's GeoTIFFs by fire (hub.list_fire_files).
        existing (set[str]): file paths already in the window repo.
        splits (tuple[str, ...]): the splits to build.
        overwrite (bool): rebuild fires already in the repo.
        only_fires (list[str] | None): restrict to these ids (debugging).
        default_label_sel (int): the rule for fires missing from rules.

    Returns:
        WindowPlan: the jobs and the fires left alone.
    """
    wanted = None if only_fires is None else {str(fire_id) for fire_id in only_fires}
    jobs, already, missing = [], [], []
    ids = roi.ids_by_split()
    for split in splits:
        for fire_id in ids[split]:
            if wanted is not None and fire_id not in wanted:
                continue
            if fire_id not in locations:
                missing.append(fire_id)
            elif f"{split}/p{fire_id}.npz" in existing and not overwrite:
                already.append(f"{split}/p{fire_id}.npz")
            else:
                jobs.append(WindowJob(split, fire_id, int(rules.get(fire_id, default_label_sel)), locations[fire_id]))
    return WindowPlan(tuple(jobs), tuple(already), tuple(missing))


# ----- Building windows ----- #


@dataclass(frozen=True)
class SplitResult:
    """Report what build_split did for one split.

    process_windows combines one of these per split into its report. Fires
    without a complete window are skipped, not failed.

    Attributes:
        built (tuple[str, ...]): fire ids built.
        skipped (dict[str, str]): {fire id: why it has no archive}.
        uploaded (tuple[str, ...]): "split/p<id>.npz" names uploaded.
    """

    built: tuple
    skipped: dict
    uploaded: tuple


def build_and_stage(fire_id, fire_files, label_sel, split_dir, src_repo, token, length, interval, temp_root, debug):
    """Download one fire's images, build its windows and write its archive.

    The images go into a temporary folder that is deleted afterwards. A fire
    with too few days gets no archive; the notes say why.

    Args:
        fire_id (str): the fire id.
        fire_files (dict[str, list[str]]): {mode: [source repo GeoTIFF paths]}.
        label_sel (int): the label rule (0 or 1).
        split_dir (str | Path): where the finished archive waits for upload.
        src_repo (str): the raw GeoTIFF repo.
        token (str | None): a Hugging Face token.
        length (int): input days per window.
        interval (int): days between window starts.
        temp_root (str | Path): where the temporary download folder is made.
        debug (bool): add per-day label counts to the notes.

    Returns:
        tuple[Path | None, tuple[str, ...]]: (the archive, or None when the fire
        has no windows; the build notes).
    """
    with tempfile.TemporaryDirectory(dir=temp_root) as temp_dir:
        local_map = {
            path: hub.download_file(src_repo, path, token, cache_dir=temp_dir)
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
    """Upload one split's staged archives, then empty the staging folder.

    All archives go to window_repo/<split> in one commit. The folder is emptied
    so the next batch starts clean.

    Args:
        split_dir (str | Path): the staging folder of the split.
        split (str): the split name (the repo folder).
        window_repo (str): the window repo.
        token (str | None): a token with write access.

    Returns:
        list[str]: the uploaded "split/p<id>.npz" names.
    """
    print(f"Uploading staged {split} archives to {window_repo}")
    names = hub.upload_folder(split_dir, window_repo, split, token)
    shutil.rmtree(split_dir, ignore_errors=True)
    Path(split_dir).mkdir(parents=True, exist_ok=True)
    return [f"{split}/{name}" for name in names]


def build_split(jobs, split_dir, window_repo, src_repo, token, length, interval, temp_root, batch_size, debug=False):
    """Build every planned fire of one split, uploading archives in batches.

    Each job is built with build_and_stage. Every batch_size finished archives
    are uploaded in one commit, and the rest at the end. The staging folder is
    removed afterwards.

    Args:
        jobs (tuple[WindowJob, ...]): the split's planned fires.
        split_dir (str | Path): the split's staging folder.
        window_repo (str): the window repo.
        src_repo (str): the raw GeoTIFF repo.
        token (str | None): a token with write access to window_repo.
        length (int): input days per window.
        interval (int): days between window starts.
        temp_root (str | Path): where temporary download folders are made.
        batch_size (int): archives per upload commit.
        debug (bool): add per-day label counts to the notes.

    Returns:
        SplitResult: what was built, skipped and uploaded.
    """
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    built, skipped, uploaded, staged = [], {}, [], 0
    for number, job in enumerate(jobs, start=1):
        print(f"[{job.split.upper()} | {number}/{len(jobs)}] Processing: {job.fire_id}")
        path, notes = build_and_stage(job.fire_id, job.files, job.label_sel, split_dir, src_repo, token,
                                      length, interval, temp_root, debug)
        print("\n".join(notes))
        if path is None:
            skipped[job.fire_id] = "no complete window"
        else:
            built.append(job.fire_id)
            staged += 1
        gc.collect()
        if staged >= batch_size:
            uploaded += upload_split_batch(split_dir, jobs[0].split, window_repo, token)
            staged = 0
    if staged > 0:
        uploaded += upload_split_batch(split_dir, jobs[0].split, window_repo, token)
    shutil.rmtree(split_dir, ignore_errors=True)
    return SplitResult(tuple(built), skipped, tuple(uploaded))


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
    """Build window archives for every fire with images and upload them to the window repo.

    This is the dataset build. The repo's window_config.json is checked (or
    written) first, so a repo built with other settings fails at once. Then the
    ROI CSVs are read once, a plan picks the fires, and each split is built and
    uploaded in batches. Train/val fires use train_val_label_sel and test fires
    their CSV's rule, as in the paper. Fires already in the repo are skipped,
    so an interrupted build resumes.

    Args:
        token (str | None): a token with write access to window_repo.
        src_repo (str): the raw GeoTIFF repo.
        window_repo (str | None): the window repo (default: the one for length/interval).
        batch_size (int): archives per upload commit.
        length (int): input days per window.
        interval (int): days between window starts.
        splits (tuple[str, ...]): the splits to build.
        roi_dir (str | Path): the local folder for the ROI CSVs.
        work_dir (str | Path): where temporary and staged files go.
        overwrite (bool): rebuild fires already in the repo and rewrite window_config.json.
        debug (bool): print per-day label counts.
        use_roi_label_sel (bool): test fires use their CSV's rule.
        train_val_label_sel (int): the train/val label rule (0 or 1).
        only_fires (list[str] | None): restrict to these ids (debugging).

    Returns:
        WindowBuildReport: what was built, skipped and uploaded.

    Raises:
        ValueError: if the repo was built with other window settings.
    """
    window_repo = window_repo or window_repo_for(length, interval)
    print(f"Windows: {length} input days every {interval} days -> {window_repo}")
    hub.ensure_repo(window_repo, token)
    decision = hub.ensure_window_config(
        window_repo, hub.window_settings(length, interval, train_val_label_sel, use_roi_label_sel),
        overwrite=overwrite, token=token,
    )
    print(decision.message)

    roi = read_roi(hub.download_roi_csvs(src_repo, token, roi_dir))
    plan = plan_windows(
        roi, label_rules(roi, splits, use_roi_label_sel, train_val_label_sel), hub.list_fire_files(src_repo, token),
        set(hub.list_repo_files(window_repo, token)), splits, overwrite, only_fires, train_val_label_sel,
    )
    print(plan.summary())

    work_dir = Path(work_dir)
    temp_root = work_dir / "temp"
    temp_root.mkdir(parents=True, exist_ok=True)
    built, skipped, uploaded = {}, {}, []
    for split, jobs in plan.jobs_by_split(splits).items():
        result = build_split(jobs, work_dir / "staging" / split, window_repo, src_repo, token, length, interval,
                             temp_root, batch_size, debug)
        built[split] = result.built
        skipped.update(result.skipped)
        uploaded += result.uploaded

    report = WindowBuildReport(window_repo, built, skipped,
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
    """Download every window archive of the window repo into a local data root.

    Each archive goes to the train/val/test folder its fire belongs to (from the
    ROI CSVs); archives already on disk are kept. In TESSERA mode, fires without
    prior-year TESSERA are left out and the TESSERA files are downloaded too.
    Use a separate data root per mode, so the two never mix.

    Args:
        token (str | None): a Hugging Face token.
        window_repo (str | None): the window repo (default: the one for length/interval).
        src_repo (str): the raw GeoTIFF repo (for the ROI CSVs).
        data_root (str | Path): where the archives go.
        roi_dir (str | Path): the local folder for the ROI CSVs.
        include_tessera (bool): TESSERA mode.
        tessera_repo (str): the TESSERA repo.
        length (int): input days per window.
        interval (int): days between window starts.

    Returns:
        DownloadReport: what was fetched, kept, skipped and excluded.
    """
    window_repo = window_repo or window_repo_for(length, interval)
    data_root = Path(data_root)
    window_config = hub.read_window_config(window_repo, token)
    print(f"Window repo {window_repo}: {window_config}")
    roi = read_roi(hub.download_roi_csvs(src_repo, token, roi_dir))
    excluded = tessera_excluded_npz(roi) if include_tessera else []
    split_of = roi.split_of()
    for split in SPLITS:
        (data_root / split).mkdir(parents=True, exist_ok=True)

    downloaded, present, skipped = [], [], []
    for name in hub.list_repo_files(window_repo, token):
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
        shutil.move(hub.download_file(window_repo, name, token, local_dir=data_root), target)
        downloaded.append(str(target.relative_to(data_root)))

    missing = ()
    if include_tessera:
        missing = tuple(tessera_master.download_tessera_files(data_root, tessera_repo, token))
    report = DownloadReport(data_root, tuple(downloaded), tuple(present), tuple(skipped), tuple(excluded),
                            missing, window_config)
    print(report.summary())
    return report


def upload_data_zip(token, data_root=DATA_DIR, zip_path=None, repo_id=None, zip_name=DATA_ZIP, folders=ZIP_FOLDERS):
    """Zip a data root's folders and upload the zip to a repo.

    Training downloads this one zip instead of one file per fire. Missing
    folders are skipped. TESSERA mode uses the TESSERA repo and another zip
    name, so the full data.zip is never replaced by the reduced set.

    Args:
        token (str | None): a token with write access to repo_id.
        data_root (str | Path): the data root to zip.
        zip_path (str | Path | None): the local zip (default: next to data_root).
        repo_id (str | None): the repo (default: the window repo of the default settings).
        zip_name (str): the zip's name in the repo.
        folders (tuple[str, ...]): the data root's folders to include.

    Returns:
        ZipUpload: the zip, its members and where it went.
    """
    repo_id = repo_id or window_repo_for()
    zip_path = Path(zip_path) if zip_path else Path(data_root).parent / zip_name
    members = files.zip_folders([Path(data_root) / folder for folder in folders], zip_path)
    hub.upload_file(zip_path, zip_name, repo_id, token)
    report = ZipUpload(zip_path, tuple(members), repo_id, zip_name)
    print(report.summary())
    return report


def rebuild_data_zip(token, window_repo=None, length=TS_LENGTH, interval=TS_INTERVAL, src_repo=SRC_REPO,
                     work_dir=WORK_DIR, roi_dir=ROI_CACHE_DIR):
    """Rebuild data.zip from the window repo's archives and upload it.

    After new windows are built, data.zip must be remade. This downloads every
    archive into a fresh folder and uploads it as one zip.

    Args:
        token (str | None): a token with write access to the window repo.
        window_repo (str | None): the window repo (default: the one for length/interval).
        length (int): input days per window.
        interval (int): days between window starts.
        src_repo (str): the raw GeoTIFF repo (for the ROI CSVs).
        work_dir (str | Path): where the download folder and zip go.
        roi_dir (str | Path): the local folder for the ROI CSVs.

    Returns:
        ZipUpload: the zip, its members and where it went.
    """
    window_repo = window_repo or window_repo_for(length, interval)
    zip_root = Path(work_dir) / f"zip_ts{length}_it{interval}"
    download_windows(token, window_repo=window_repo, src_repo=src_repo, data_root=zip_root, roi_dir=roi_dir)
    return upload_data_zip(token, data_root=zip_root, zip_path=f"{zip_root}.zip", repo_id=window_repo)


def pull_data_zip(data_root, token=None, repo_id=None, zip_name=DATA_ZIP):
    """Download a data zip and unpack it into a data root.

    The zip's folders (train/, val/, test/, ...) are recreated under
    data_root. Member paths are checked before unpacking.

    Args:
        data_root (str | Path): where to download and unpack.
        token (str | None): a Hugging Face token.
        repo_id (str | None): the repo holding the zip (default: the window repo of the default settings).
        zip_name (str): the zip's name in the repo.

    Returns:
        Path: data_root.
    """
    data_root = hub.pull_zip(repo_id or window_repo_for(), zip_name, data_root, token)
    print(f"Extracted {zip_name} into {data_root}")
    return Path(data_root)


def stage_training_data(work, token=None, window_repo=None, tessera_repo=TESSERA_REPO,
                        attached_data_root=None, attached_zip=None,
                        attached_tessera_dir=None, attached_tessera_zip=None):
    """Make the training data ready in a notebook and return where it is.

    The data comes from an attached extracted folder, an attached data.zip, or
    the window repo's data.zip, in that order. In TESSERA mode the TESSERA
    files are staged too and the training root is a symlinked view without the
    excluded fires. This replaces the training notebooks' data cell.

    Args:
        work (Workspace): the notebook's paths.
        token (str | None): a Hugging Face token.
        window_repo (str | None): the window repo (default: the workspace's).
        tessera_repo (str): the TESSERA repo.
        attached_data_root (str | Path | None): an attached extracted data root.
        attached_zip (str | Path | None): an attached data.zip.
        attached_tessera_dir (str | Path | None): an attached folder of TESSERA files.
        attached_tessera_zip (str | Path | None): an attached tessera.zip.

    Returns:
        StagedData: the data root, the training root and archive counts.
    """
    if attached_data_root:
        data_root = files.find_data_root(attached_data_root)
    else:
        archive = Path(attached_zip) if attached_zip else Path(
            hub.download_file(window_repo or work.window_repo, DATA_ZIP, token, local_dir=work.download_dir)
        )
        data_root = files.extract_data(archive, work.extract_dir)

    tessera_dir, train_root = None, data_root
    if work.use_tessera:
        view = tessera_master.stage_tessera(data_root, work, token, tessera_repo,
                                            attached_tessera_dir, attached_tessera_zip)
        train_root, tessera_dir = view.root, view.root / "tessera"
    staged = StagedData(data_root, train_root, tessera_dir, files.count_archives(train_root))
    print(staged.summary())
    return staged
