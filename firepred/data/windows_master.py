"""MASTER for datasets: every window-data step a notebook or the CLI runs.

Level 1 (data, dataset building). Notebooks call only this module and data/tessera_master.py.
Each function runs one pipeline step end to end and returns a frozen value
(reports have .summary()). Progress is printed here and nowhere below.

    workspace           every local path a notebook uses, from a few settings
    pin_caches          keep Hugging Face / temp caches under the workspace
    read_secret         a token from Kaggle secrets, Colab userdata or the environment
    process_windows     raw GeoTIFFs -> one window archive per fire in the window repo
    download_windows    window archives (+ TESSERA files) and window_config.json into a local data root
    count_pixels        every split's background/fire target pixels -> the data root's window_config.json
    compute_normalization  per-channel input mean/std of train/, one window in memory per thread
                        -> the data root's window_config.json (training reads both before it starts)
    upload_data_zip     zip a data root and upload it as data.zip
    rebuild_data_zip    download a whole window repo, add count_pixels + compute_normalization to its
                        window_config.json (repo copy too), then upload it as data.zip
    pull_data_zip       download data.zip and unpack it into a data root
    stage_training_data training notebooks: find/download/extract data (+ TESSERA view)

    SRC_REPO raw GeoTIFFs --one fire at a time--> data/windows.build_fire_windows
        --> data/archive.write_fire_archive p<id>.npz --uploaded in batches--> window repo
"""

import gc
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from firepred.config.dataset import (
    DAILY_BLOCK_DAYS,
    DAILY_COMPRESSLEVEL,
    DAILY_DATA_DIR,
    DAILY_FORMAT_VERSION,
    DATA_DIR,
    DATA_ZIP,
    ROI_CACHE_DIR,
    SPLITS,
    SRC_REPO,
    TESSERA_REPO,
    TRAIN_VAL_LABEL_SEL,
    TS_INTERVAL,
    TS_LENGTH,
    WINDOW_CONFIG_FILE,
    WINDOW_UPLOAD_BATCH,
    WORK_DIR,
    ZIP_FOLDERS,
    ZONE_DIR,
    window_repo_for,
    with_dataset,
)
from firepred.data import files, hub, tessera_master
from firepred.data.archive import convert_to_daily, find_fire_sources, fire_name, write_fire_archive
from firepred.data.stats import (
    NORMALIZATION,
    PIXEL_COUNTS,
    count_labels,
    keep_statistics,
    read_window_config_file,
    split_normalization,
    window_config_path,
    window_settings_of,
    write_window_config_file,
)
from firepred.data.files import fire_id_of, label_rules, read_roi
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
        """Summarize the build: counts per split and the skipped fires."""
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
        zones (tuple[str, ...]): "zones/p<id>.npz" zone files downloaded or already on disk.
    """

    data_root: Path
    downloaded: tuple
    present: tuple
    skipped: tuple
    excluded: tuple
    missing_tessera: tuple = ()
    window_config: dict = None
    zones: tuple = ()

    def summary(self):
        """Summarize the download, with a warning for archives without TESSERA.

        The first line counts the archives. A line for zone files follows when the
        repo had any, and a warning names the archives still without a TESSERA file.

        Returns:
            str: the summary, one or more lines.
        """
        text = (f"{self.data_root}: downloaded {len(self.downloaded)}, already present {len(self.present)}, "
                f"skipped {len(self.skipped)}, excluded {len(self.excluded)}")
        if self.zones:
            text += f"\nZone files: {len(self.zones)} in {self.data_root / ZONE_DIR} (train with --zones-dir there)"
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
        """Summarize the upload in one line."""
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
        """Summarize the staged data in one line."""
        counts = ", ".join(f"{split} {n}" for split, n in self.counts.items())
        return f"Data root: {self.data_root} | training root: {self.train_root} | archives: {counts}"


@dataclass(frozen=True)
class PixelCountReport:
    """Report what count_pixels counted and where it wrote the counts.

    The master returns this instead of only printing. A notebook can read the
    counts from it.

    Attributes:
        config_path (Path): the data root's window_config.json.
        counts (dict[str, LabelCounts]): {split: its counts}.
        uploaded_to (str | None): the window repo the file was also written to.
    """

    config_path: Path
    counts: dict
    uploaded_to: str = None

    def summary(self):
        """Summarize the counts, one split per line, and where they went."""
        lines = [counts.summary() for counts in self.counts.values()]
        lines.append(f"Pixel counts written to {self.config_path}"
                     + (f" and {self.uploaded_to}/{WINDOW_CONFIG_FILE}" if self.uploaded_to else ""))
        return "\n".join(lines)


@dataclass(frozen=True)
class NormalizationReport:
    """Report what compute_normalization measured and where it wrote the statistics.

    The master returns this instead of only printing. A notebook can read the
    numbers from it.

    Attributes:
        config_path (Path): the data root's window_config.json.
        stats (NormalizationStats): the statistics.
        uploaded_to (str | None): the window repo the file was also written to.
    """

    config_path: Path
    stats: object
    uploaded_to: str = None

    def summary(self):
        """Summarize the statistics and where they went."""
        return "\n".join((
            self.stats.summary(),
            f"Normalization written to {self.config_path}"
            + (f" and {self.uploaded_to}/{WINDOW_CONFIG_FILE}" if self.uploaded_to else ""),
        ))


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
        """Group the planned jobs by split."""
        return {split: tuple(job for job in self.jobs if job.split == split) for split in splits}

    def summary(self):
        """Summarize the plan in one line."""
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
        rules (dict[str, int]): {fire id: label rule} (files.label_rules).
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
    """Upload one split's staged archives, then empty the staging folder."""
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


# ----- Data statistics: written into the data root's window_config.json ----- #


def data_root_window_config(data_root, window_repo, token=None):
    """Read the data root's window_config.json, fetching the repo's copy when there is none.

    Data unpacked from an older data.zip has no copy yet. The repo's file records
    the same window settings, and the copy names that repo as its dataset.

    Args:
        data_root (Path): the data root.
        window_repo (str): the window repo the data came from.
        token (str | None): a Hugging Face token.

    Returns:
        dict: the window config.

    Raises:
        RuntimeError: if neither the data root nor the repo has one.
    """
    config = read_window_config_file(data_root)
    if config is not None:
        return config
    config = hub.read_window_config(window_repo, token)
    if config is None:
        raise RuntimeError(f"Neither {data_root} nor {window_repo} has a {WINDOW_CONFIG_FILE}; "
                           "build the repo with `process` (it records the window settings)")
    print(f"{data_root} had no {WINDOW_CONFIG_FILE}; using {window_repo}'s")
    return with_dataset(config, window_repo)


def save_window_config(data_root, config, upload, window_repo, token):
    """Write the data root's window_config.json and, when asked, the window repo's.

    The repo's file is replaced only if it records the same window settings.
    Statistics therefore never land beside windows of another shape. An upload
    names the window repo as the dataset in both copies, so repos built before
    the dataset was recorded gain it here.

    Args:
        data_root (Path): the data root.
        config (dict): the whole file.
        upload (bool): also write it to the window repo.
        window_repo (str): the window repo.
        token (str | None): a token with write access (upload only).

    Returns:
        tuple[Path, str | None]: (the local file, the repo uploaded to or None).

    Raises:
        ValueError: if the repo's file records other window settings.
    """
    if upload:
        config = with_dataset(config, window_repo)
    path = write_window_config_file(data_root, config)
    if not upload:
        return path, None
    remote = hub.read_window_config(window_repo, token)
    if remote is not None and window_settings_of(remote) != window_settings_of(config):
        raise ValueError(f"{window_repo}/{WINDOW_CONFIG_FILE} records other window settings than {path}; "
                         "not uploading statistics to it")
    hub.upload_file(json.dumps(config, indent=2).encode(), WINDOW_CONFIG_FILE, window_repo, token)
    return path, window_repo


def count_pixels(data_root=DATA_DIR, splits=SPLITS, token=None, window_repo=None, length=TS_LENGTH,
                 interval=TS_INTERVAL, upload=False, workers=None):
    """Count every split's background, fire and other target pixels into window_config.json.

    Only the small label_### members are read, so this takes seconds. The
    counts go into the pixel_counts block of <data_root>/window_config.json
    (splits not counted now keep their earlier entries); --ce-weight auto trains
    with the train entry's background-to-fire ratio. Each entry lists its fires,
    so training refuses counts that miss a fire it would train on.

    Args:
        data_root (str | Path): the data root with train/, val/ and test/.
        splits (tuple[str, ...]): the splits to count; empty ones are skipped.
        token (str | None): a Hugging Face token (to fetch or upload window_config.json).
        window_repo (str | None): the window repo (default: the one for length/interval).
        length (int): input days per window (picks the default repo).
        interval (int): days between window starts (picks the default repo).
        upload (bool): also write the updated window_config.json to the window repo.
        workers (int | None): reading threads.

    Returns:
        PixelCountReport: the counts and where they were written.
    """
    data_root = Path(data_root)
    window_repo = window_repo or window_repo_for(length, interval)
    config = data_root_window_config(data_root, window_repo, token)
    counts = {}
    for split in splits:
        if not find_fire_sources(data_root / split):
            print(f"No fires in {data_root / split}; not counted")
            continue
        counts[split] = count_labels(data_root / split, workers=workers)
    config[PIXEL_COUNTS] = {**config.get(PIXEL_COUNTS, {}),
                            **{split: split_counts.record() for split, split_counts in counts.items()}}
    path, uploaded_to = save_window_config(data_root, config, upload, window_repo, token)
    report = PixelCountReport(path, counts, uploaded_to)
    print(report.summary())
    return report


def compute_normalization(data_root=DATA_DIR, split="train", token=None, window_repo=None, length=TS_LENGTH,
                          interval=TS_INTERVAL, upload=False, workers=None):
    """Measure the per-channel input mean and std of a split into window_config.json.

    Windows are read lazily: each thread decompresses one window, folds it
    into running per-channel moments and drops it, so memory stays near one
    window per thread (about 42 MB for 6 days) however large the split. The
    statistics go into the normalization block of <data_root>/window_config.json;
    training standardizes every window with them. Only the training split
    should be measured, so nothing is learned from held-out fires.

    Args:
        data_root (str | Path): the data root with train/, val/ and test/.
        split (str): the split to measure.
        token (str | None): a Hugging Face token (to fetch or upload window_config.json).
        window_repo (str | None): the window repo (default: the one for length/interval).
        length (int): input days per window (picks the default repo).
        interval (int): days between window starts (picks the default repo).
        upload (bool): also write the updated window_config.json to the window repo.
        workers (int | None): reading threads (each holds one window).

    Returns:
        NormalizationReport: the statistics and where they were written.
    """
    data_root = Path(data_root)
    window_repo = window_repo or window_repo_for(length, interval)
    config = data_root_window_config(data_root, window_repo, token)
    print(f"Measuring {data_root / split} one window at a time")
    stats = split_normalization(data_root / split, workers=workers)
    config[NORMALIZATION] = stats.record()
    path, uploaded_to = save_window_config(data_root, config, upload, window_repo, token)
    report = NormalizationReport(path, stats, uploaded_to)
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
    ROI CSVs); archives already on disk are kept. A repo with a zones/ folder
    (zone files for --zones-dir) has those copied to <data_root>/zones too. The repo's window_config.json
    is copied to <data_root>/window_config.json, naming the repo as its dataset
    and keeping statistics computed there earlier that the repo's copy lacks.
    In TESSERA mode, fires without prior-year TESSERA are left out and the
    TESSERA files are downloaded too. Use a separate data root per mode, so the
    two never mix.

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

    # Training reads the data root's copy of window_config.json before anything else.
    if window_config is None:
        print(f"WARNING: {window_repo} has no {WINDOW_CONFIG_FILE}; training needs one in {data_root}")
    else:
        local = keep_statistics(window_config, read_window_config_file(data_root))
        write_window_config_file(data_root, with_dataset(local, window_repo))

    downloaded, present, skipped, zones = [], [], [], []
    excluded_ids = {fire_id_of(name) for name in excluded}
    for name in hub.list_repo_files(window_repo, token):
        # Zone files (zones/p<id>.npz) keep their folder: <data_root>/zones is --zones-dir's usual place.
        if name.split("/")[0] == ZONE_DIR and name.endswith(".npz"):
            fire_id = fire_id_of(name)
            if fire_id in excluded_ids or fire_id not in split_of:
                skipped.append(name)
                continue
            if not (data_root / name).exists():
                hub.download_file(window_repo, name, token, local_dir=data_root)
            zones.append(name)
            continue
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
                            missing, window_config, tuple(zones))
    print(report.summary())
    return report


def upload_data_zip(token, data_root=DATA_DIR, zip_path=None, repo_id=None, zip_name=DATA_ZIP, folders=ZIP_FOLDERS):
    """Zip a data root's folders and upload the zip to a repo.

    Training downloads this one zip instead of one file per fire. Missing
    folders are skipped. The data root's window_config.json (with any
    statistics) goes in at the zip's root. TESSERA mode uses the TESSERA repo and another zip
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
    # window_config.json rides along at the zip's root, so training finds it after unpacking.
    config_file = window_config_path(data_root)
    if not config_file.is_file():
        print(f"WARNING: {config_file} is missing; training will fall back to the official TS-SatFire statistics")
    members = files.zip_folders([Path(data_root) / folder for folder in folders], zip_path,
                                extra_files=[config_file] if config_file.is_file() else ())
    hub.upload_file(zip_path, zip_name, repo_id, token)
    report = ZipUpload(zip_path, tuple(members), repo_id, zip_name)
    print(report.summary())
    return report


def rebuild_data_zip(token, window_repo=None, length=TS_LENGTH, interval=TS_INTERVAL, src_repo=SRC_REPO,
                     work_dir=WORK_DIR, roi_dir=ROI_CACHE_DIR, statistics=True, workers=None):
    """Rebuild data.zip from the window repo's archives, with their statistics, and upload it.

    After new windows are built, data.zip must be remade. This downloads every
    archive into a fresh folder. Once all of them are there, the pixel counts
    and the training split's normalization are measured (count_pixels,
    compute_normalization) into window_config.json, which is uploaded to the
    repo and zipped with the archives, so training gets data and statistics
    that always match. Then everything is uploaded as one zip.

    Args:
        token (str | None): a token with write access to the window repo.
        window_repo (str | None): the window repo (default: the one for length/interval).
        length (int): input days per window.
        interval (int): days between window starts.
        src_repo (str): the raw GeoTIFF repo (for the ROI CSVs).
        work_dir (str | Path): where the download folder and zip go.
        roi_dir (str | Path): the local folder for the ROI CSVs.
        statistics (bool): measure the statistics before zipping (False: zip the repo's
            window_config.json as it is).
        workers (int | None): reading threads for the statistics.

    Returns:
        ZipUpload: the zip, its members and where it went.
    """
    window_repo = window_repo or window_repo_for(length, interval)
    zip_root = Path(work_dir) / f"zip_ts{length}_it{interval}"
    download_windows(token, window_repo=window_repo, src_repo=src_repo, data_root=zip_root, roi_dir=roi_dir)
    if statistics:
        # Every archive is on disk now, so the statistics describe the whole dataset.
        count_pixels(zip_root, token=token, window_repo=window_repo, upload=True, workers=workers)
        compute_normalization(zip_root, token=token, window_repo=window_repo, upload=True, workers=workers)
    return upload_data_zip(token, data_root=zip_root, zip_path=f"{zip_root}.zip", repo_id=window_repo)


def pull_data_zip(data_root, token=None, repo_id=None, zip_name=DATA_ZIP):
    """Download a data zip and unpack it into a data root."""
    data_root = hub.pull_zip(repo_id or window_repo_for(), zip_name, data_root, token)
    print(f"Extracted {zip_name} into {data_root}")
    return Path(data_root)


def stage_training_data(work, token=None, window_repo=None, tessera_repo=TESSERA_REPO,
                        attached_data_root=None, attached_zip=None,
                        attached_tessera_dir=None, attached_tessera_zip=None, src_repo=SRC_REPO):
    """Make the training data ready in a notebook and return where it is.

    The data comes from an attached extracted folder, an attached data.zip, the
    window repo's data.zip, or (a repo without one, such as a daily repo) the
    repo's archives one by one, in that order. In TESSERA mode the TESSERA
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
        src_repo (str): the raw GeoTIFF repo (its ROI CSVs place archives in their splits).

    Returns:
        StagedData: the data root, the training root and archive counts.
    """
    window_repo = window_repo or work.window_repo
    if attached_data_root:
        data_root = files.find_data_root(attached_data_root)
    elif attached_zip or DATA_ZIP in hub.list_repo_files(window_repo, token):
        archive = Path(attached_zip) if attached_zip else Path(
            hub.download_file(window_repo, DATA_ZIP, token, local_dir=work.download_dir)
        )
        data_root = files.extract_data(archive, work.extract_dir)
    else:
        # Daily repos ship no data.zip (it would only duplicate the archives): each
        # archive is downloaded straight into its split folder, so the disk holds
        # the data once and nothing is unpacked.
        download_windows(token, window_repo=window_repo, src_repo=src_repo, data_root=work.extract_dir,
                         roi_dir=work.roi_dir, length=work.ts_length, interval=work.ts_interval)
        data_root = files.find_data_root(work.extract_dir)

    tessera_dir, train_root = None, data_root
    if work.use_tessera:
        view = tessera_master.stage_tessera(data_root, work, token, tessera_repo,
                                            attached_tessera_dir, attached_tessera_zip)
        train_root, tessera_dir = view.root, view.root / "tessera"
    staged = StagedData(data_root, train_root, tessera_dir, files.count_archives(train_root))
    print(staged.summary())
    return staged


# ----- Daily archives ----- #


@dataclass(frozen=True)
class DailyConversionReport:
    """Record what convert_daily did.

    Every converted fire was read back window by window and matched bit for
    bit. summary() is the line the master prints at the end.

    Attributes:
        dest_root (Path): where the daily archives are.
        converted (tuple[tuple[str, int, int, int], ...]): (split/p<id>, windows checked,
            source bytes, daily bytes) per converted fire.
        present (tuple[str, ...]): fires skipped because their daily archive exists.
        uploaded (tuple[str, ...]): files uploaded to the destination repo.
    """

    dest_root: Path
    converted: tuple
    present: tuple
    uploaded: tuple

    def summary(self):
        """Say how many fires were converted and how much smaller they got.

        Sizes count only the fires converted in this run, not the skipped ones.
        The ratio compares the format-2 sources with their daily archives.

        Returns:
            str: one line.
        """
        source = sum(row[2] for row in self.converted)
        daily = sum(row[3] for row in self.converted)
        windows = sum(row[1] for row in self.converted)
        ratio = f" ({source / daily:.1f}x smaller)" if daily else ""
        return (f"Daily archives in {self.dest_root}: {len(self.converted)} fires converted, {windows} windows "
                f"checked bit-identical, {source / 1e9:.2f} GB -> {daily / 1e9:.2f} GB{ratio}; "
                f"{len(self.present)} already there; {len(self.uploaded)} files uploaded")


# Window settings a daily data root does not fix: training cuts its windows at the run's -ts/-it.
DAILY_UNFIXED_SETTINGS = ("ts_length", "ts_interval")


def daily_window_config(config, source, block_days, compresslevel):
    """Give a daily data root's window_config.json: the source's settings and statistics, format version 3.

    A daily archive stores days, not windows, so the file records no window
    length or interval: training cuts windows at its own -ts/-it (any whose days
    an archive holds). The source's ts_length/ts_interval move to the notes. The
    source's pixel counts and normalization were measured on its windows, which
    read back bit-identical; they still describe the data at other -ts/-it, with
    each day weighted by how many of those windows held it.

    Args:
        config (dict): the source's window_config.json.
        source (str): the source repo or data root, for the notes.
        block_days (int): days per block.
        compresslevel (int): Zstandard level.

    Returns:
        dict: the daily data root's window_config.json.
    """
    notes = dict(config.get("notes") or {})
    notes.update({
        "npz_compresslevel": compresslevel,
        "daily_block_days": block_days,
        "converted_from": source,
        "converted_from_windows": {key: config[key] for key in DAILY_UNFIXED_SETTINGS if key in config},
        "daily_code_version": files.code_version(),
        "archive_format_version": "3 = one frame per day; windows are sliced when read (data/archive.py)",
        "windows": "not fixed: training cuts windows at its -ts/-it from the days each archive holds; "
                   "statistics were measured on the source's windows (converted_from_windows)",
    })
    settings = {key: value for key, value in config.items() if key not in DAILY_UNFIXED_SETTINGS}
    return {**settings, "archive_format_version": DAILY_FORMAT_VERSION, "notes": notes}


def daily_sources(token, window_repo, data_root, splits):
    """List the format-2 archives to convert: (split, p<id>, local path or repo file name)."""
    if data_root:
        return [(split, fire_name(path), path) for split in splits for path in find_fire_sources(Path(data_root) / split)]
    return [(name.split("/")[0], Path(name).stem, name) for name in sorted(hub.list_repo_files(window_repo, token))
            if name.split("/")[0] in splits and name.count("/") == 1 and name.endswith(".npz")]


def upload_daily_batch(dest_root, pending, dest_repo, token):
    """Upload a batch of new daily archives, one commit per split.

    The archives are hard-linked into a hidden staging folder, so the disk
    holds no second copy. The folder is removed after the upload.

    Args:
        dest_root (Path): the daily data root.
        pending (list[tuple[str, str]]): (split, file name) of each archive to upload.
        dest_repo (str): the destination repo.
        token (str | None): a token with write access.

    Returns:
        list[str]: the uploaded repo paths.
    """
    uploaded = []
    for split in sorted({split for split, _ in pending}):
        with tempfile.TemporaryDirectory(prefix=".daily_upload_", dir=dest_root) as staging:
            for pending_split, name in pending:
                if pending_split == split:
                    try:
                        os.link(dest_root / split / name, Path(staging) / name)
                    except OSError:
                        shutil.copy2(dest_root / split / name, Path(staging) / name)
            uploaded += [f"{split}/{name}" for name in hub.upload_folder(staging, dest_repo, split, token)]
    return uploaded


def convert_daily(token=None, window_repo=None, data_root=None, dest_root=DAILY_DATA_DIR, dest_repo=None,
                  length=TS_LENGTH, interval=TS_INTERVAL, splits=SPLITS, block_days=DAILY_BLOCK_DAYS,
                  compresslevel=DAILY_COMPRESSLEVEL, batch_size=WINDOW_UPLOAD_BATCH, work_dir=WORK_DIR,
                  overwrite=False):
    """Convert format-2 window archives into daily archives, proving every window reads back bit-identical.

    The source is a data root on disk (data_root) or a window repo, whose
    archives are downloaded one at a time and deleted once converted, so one
    archive is on disk at a time. Daily archives go to dest_root/<split>/; with
    dest_repo they are also uploaded every batch_size fires, with the data
    root's window_config.json at the end. Fires whose daily archive exists are
    skipped (resume) unless overwrite. The source is never changed.

    Args:
        token (str | None): a Hugging Face token (write access for dest_repo).
        window_repo (str | None): the format-2 repo (default: the one for length/interval).
        data_root (str | Path | None): convert this data root's archives instead of a repo.
        dest_root (str | Path): where the daily archives go.
        dest_repo (str | None): also upload them here.
        length (int): input days per window of the source.
        interval (int): days between window starts of the source.
        splits (tuple[str, ...]): the splits to convert.
        block_days (int): days per block.
        compresslevel (int): Zstandard level.
        batch_size (int): fires per upload.
        work_dir (str | Path): where downloads wait while converted.
        overwrite (bool): convert fires whose daily archive exists.

    Returns:
        DailyConversionReport: what was converted, skipped and uploaded.

    Raises:
        ValueError: if the source's window_config.json records other window settings or is already daily.
    """
    source = str(data_root) if data_root else (window_repo or window_repo_for(length, interval))
    config = read_window_config_file(data_root) if data_root else hub.read_window_config(source, token)
    if config is not None:
        if config.get("archive_format_version") == DAILY_FORMAT_VERSION:
            raise ValueError(f"{source} already holds daily archives")
        for key, asked in (("ts_length", length), ("ts_interval", interval)):
            if key in config and int(config[key]) != int(asked):
                raise ValueError(f"{source} records {key} = {config[key]}, not {asked}: pass the source's -ts/-it")
    dest_root = Path(dest_root)
    remote = set(hub.list_repo_files(dest_repo, token)) if dest_repo and not overwrite else set()
    if dest_repo:
        hub.ensure_repo(dest_repo, token)

    converted, present, uploaded, pending = [], [], [], []

    for split, name, where in daily_sources(token, window_repo or source, data_root, splits):
        target = dest_root / split / f"{name}.npz"
        if not overwrite and (target.exists() or f"{split}/{name}.npz" in remote):
            present.append(f"{split}/{name}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="daily_", dir=work_dir) as temp:
            local = Path(where) if data_root else Path(hub.download_file(source, where, token, local_dir=temp))
            source_bytes = sum(f.stat().st_size for f in local.rglob("*")) if local.is_dir() else local.stat().st_size
            windows = convert_to_daily(local, target, length, interval, block_days, compresslevel)
        row = (f"{split}/{name}", windows, source_bytes, target.stat().st_size)
        converted.append(row)
        print(f"{row[0]}: {windows} windows, {row[2] / 1e6:.1f} MB -> {row[3] / 1e6:.1f} MB, bit-identical")
        if dest_repo:
            pending.append((split, target.name))
            if len(pending) >= batch_size:
                uploaded += upload_daily_batch(dest_root, pending, dest_repo, token)
                pending.clear()
    if dest_repo:
        uploaded += upload_daily_batch(dest_root, pending, dest_repo, token)

    if config is not None:
        daily = daily_window_config(config, source, block_days, compresslevel)
        daily = with_dataset(daily, dest_repo) if dest_repo else daily
        write_window_config_file(dest_root, keep_statistics(daily, read_window_config_file(dest_root)))
        if dest_repo:
            hub.upload_file(window_config_path(dest_root), WINDOW_CONFIG_FILE, dest_repo, token)
            uploaded.append(WINDOW_CONFIG_FILE)
    else:
        print(f"WARNING: {source} has no {WINDOW_CONFIG_FILE}; {dest_root} gets none")
    report = DailyConversionReport(dest_root, tuple(converted), tuple(present), tuple(uploaded))
    print(report.summary())
    return report
