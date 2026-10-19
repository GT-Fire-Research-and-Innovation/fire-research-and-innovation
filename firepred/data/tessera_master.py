"""MASTER for TESSERA: every TESSERA step a notebook or the CLI runs.

Level 1 (data, dataset building). Notebooks call only this module and data/windows_master.py.
Each function runs one pipeline step end to end and returns a frozen report
(with .summary()). Progress is printed here and nowhere below. [ours]

    process_tessera     build tessera/p<id>.npz per fire, upload to TESSERA_REPO
    upload_tessera_zip  bundle them (+ excluded_npz.txt) into tessera.zip there
    pull_tessera_zip    unpack tessera.zip beside window data, drop excluded fires
    stage_tessera       training notebooks: extract (or reuse) TESSERA, build the view
    excluded_fires      the window archives TESSERA mode leaves out

TESSERA embeddings come from the year before the fire (data/tessera.py);
fires without one (the 2017 fires) are left out of TESSERA runs and their
matched baselines.
"""

import gc
import shutil
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from firepred.config.dataset import (
    ROI_CACHE_DIR,
    SPLITS,
    SRC_REPO,
    STRIP_ROWS,
    TESSERA_CELLS_PER_PIXEL,
    TESSERA_REPO,
    TESSERA_UPLOAD_BATCH,
    TESSERA_ZIP,
    WORK_DIR,
)
from firepred.data import files, hub, tessera
from firepred.data.fires import read_roi
from firepred.data.tessera import build_fire_tessera, has_prior_year_tessera, missing_tessera, tessera_excluded_npz


# ----- Reports ----- #


@dataclass(frozen=True)
class TesseraBuildReport:
    """Report what process_tessera built, skipped and uploaded.

    The master returns this instead of only printing, so a notebook or test
    can inspect the outcome. Skipped fires carry the reason.

    Attributes:
        repo (str): the TESSERA repo.
        built (tuple[str, ...]): fire ids whose file was built.
        skipped (dict[str, str]): {fire id: why it got no file}.
        uploaded (tuple[str, ...]): file names uploaded.
    """

    repo: str
    built: tuple
    skipped: dict = field(default_factory=dict)
    uploaded: tuple = ()

    def summary(self):
        """Summarize the build: counts and one line per skipped fire.

        The master prints this at the end. Skipped fires are listed with their
        reasons.

        Returns:
            str: the summary, one item per line.
        """
        lines = [f"TESSERA done in {self.repo}: built {len(self.built)}, skipped {len(self.skipped)}"]
        lines += [f"  {fire_id}: {reason}" for fire_id, reason in self.skipped.items()]
        return "\n".join(lines)


@dataclass(frozen=True)
class TesseraZipReport:
    """Report what upload_tessera_zip bundled and uploaded.

    The report also lists kept fires that still have no TESSERA file. Those
    would fail later in training, so the summary warns about them.

    Attributes:
        zip_path (Path): the local tessera.zip.
        embeddings (int): TESSERA files in the zip.
        excluded (tuple[str, ...]): window archives TESSERA runs leave out.
        missing (tuple[str, ...]): kept fires still without a file.
        repo (str): the TESSERA repo the zip went to.
    """

    zip_path: Path
    embeddings: int
    excluded: tuple
    missing: tuple
    repo: str

    def summary(self):
        """Summarize the upload, with a warning when kept fires lack a file.

        The first line gives the counts. A second line warns about fires that
        still need a TESSERA file.

        Returns:
            str: the summary.
        """
        text = (f"Uploaded {self.zip_path.name} to {self.repo}: {self.embeddings} embeddings, "
                f"{len(self.excluded)} excluded window archives")
        if self.missing:
            text += f"\nWARNING: {len(self.missing)} kept fires have no TESSERA file yet: {list(self.missing)}"
        return text


@dataclass(frozen=True)
class TesseraPull:
    """Report what pull_tessera_zip unpacked and removed.

    The removed archives are the fires without prior-year TESSERA. Removing
    them keeps TESSERA and matched no-TESSERA runs on the same fires.

    Attributes:
        tessera_dir (Path): the unpacked tessera/ folder.
        excluded (tuple[str, ...]): archives listed in excluded_npz.txt.
        removed (tuple[str, ...]): archives actually deleted.
    """

    tessera_dir: Path
    excluded: tuple
    removed: tuple

    def summary(self):
        """Summarize the pull in one line.

        The line names the TESSERA folder and how many window archives were
        removed. The master prints it.

        Returns:
            str: the summary.
        """
        return (f"TESSERA in {self.tessera_dir}; removed {len(self.removed)} window archives "
                "of fires without prior-year TESSERA")


# ----- Planning (pure) ----- #


def fires_to_build(roi, splits, only_fires=None):
    """List the fire ids of the chosen splits, each once and in order.

    A fire is listed once even if it appears in two splits. With only_fires the
    list is narrowed to those ids, e.g. for a short timing test.

    Args:
        roi (RoiTable): the fires of every split.
        splits (tuple[str, ...]): the splits to include.
        only_fires (list[str] | None): restrict to these ids.

    Returns:
        list[str]: the fire ids.

    Raises:
        ValueError: if only_fires names a fire outside the splits.
    """
    ids = roi.ids_by_split()
    fire_ids = list(dict.fromkeys(str(fire_id) for split in splits for fire_id in ids[split]))
    if only_fires is None:
        return fire_ids
    wanted = {str(fire_id) for fire_id in only_fires}
    unknown = wanted.difference(fire_ids)
    if unknown:
        raise ValueError(f"Fires not in splits {splits}: {sorted(unknown)}")
    return [fire_id for fire_id in fire_ids if fire_id in wanted]


def skip_reason(fire_id, day_files, years, available_years):
    """Explain why a fire cannot get a TESSERA file, or return "" when it can.

    A fire needs daily images in the source repo, a start date in the ROI CSVs,
    and published embeddings for the year before it started. The first missing
    requirement is the reason.

    Args:
        fire_id (str): the fire id.
        day_files (list[str]): its VIIRS_Day GeoTIFF paths.
        years (dict[str, int]): {fire id: start year}.
        available_years (Iterable[int]): years the TESSERA store publishes.

    Returns:
        str: the reason, or "" when the fire can be built.
    """
    if not day_files:
        return "no VIIRS_Day files in source repo"
    if fire_id not in years:
        return "no start_date in ROI CSVs"
    if not has_prior_year_tessera(years[fire_id]) or years[fire_id] - 1 not in available_years:
        return f"no TESSERA for prior year {years[fire_id] - 1}"
    return ""


@dataclass(frozen=True)
class TesseraJob:
    """Describe one fire that process_tessera will build.

    Only the first and last daily image are downloaded. They share the fire's
    grid, and the grid decides which TESSERA pixels are needed.

    Attributes:
        fire_id (str): the fire id.
        day_files (tuple[str, ...]): the first and last VIIRS_Day repo paths.
        start_year (int): the year the fire started.
    """

    fire_id: str
    day_files: tuple
    start_year: int


@dataclass(frozen=True)
class TesseraPlan:
    """Describe what process_tessera will do before it downloads anything.

    The plan lists the fires to build, the fires it skips (with reasons) and
    the fires already uploaded. Building it touches no file or network, so it
    can be tested on its own.

    Attributes:
        fire_ids (tuple[str, ...]): every fire considered, in order.
        jobs (tuple[TesseraJob, ...]): the fires to build.
        skipped (dict[str, str]): {fire id: why it is skipped}.
        already_built (tuple[str, ...]): fires already in the repo.
    """

    fire_ids: tuple
    jobs: tuple
    skipped: dict
    already_built: tuple = ()

    def summary(self):
        """Summarize the plan in one line.

        The line counts the fires to build, the fires skipped and those already
        uploaded. The master prints it before starting.

        Returns:
            str: the summary.
        """
        return (f"Plan: build {len(self.jobs)} TESSERA files; skip {len(self.skipped)}; "
                f"{len(self.already_built)} already in the repo")


def plan_tessera(roi, splits, locations, existing, available_years, overwrite=False, only_fires=None):
    """Plan which fires get a TESSERA file now, and why the others do not.

    Fires already in the repo are skipped unless overwrite is set; every other
    fire is checked with skip_reason. This is pure: it reads and writes
    nothing, so the plan is the same for the same inputs.

    Args:
        roi (RoiTable): the fires and their start years.
        splits (tuple[str, ...]): the splits to cover.
        locations (dict): the source repo's GeoTIFFs by fire (hub.list_fire_files).
        existing (set[str]): file paths already in the TESSERA repo.
        available_years (Iterable[int]): years the TESSERA store publishes.
        overwrite (bool): rebuild fires already in the repo.
        only_fires (list[str] | None): restrict to these ids.

    Returns:
        TesseraPlan: the jobs, the skipped fires and those already built.
    """
    fire_ids = fires_to_build(roi, splits, only_fires)
    jobs, skipped, already = [], {}, []
    for fire_id in fire_ids:
        if f"tessera/p{fire_id}.npz" in existing and not overwrite:
            already.append(fire_id)
            continue
        day_files = locations.get(fire_id, {}).get("VIIRS_Day", [])
        reason = skip_reason(fire_id, day_files, roi.start_years, available_years)
        if reason:
            skipped[fire_id] = reason
        else:
            jobs.append(TesseraJob(fire_id, tuple(dict.fromkeys((day_files[0], day_files[-1]))), roi.start_years[fire_id]))
    return TesseraPlan(tuple(fire_ids), tuple(jobs), skipped, tuple(already))


# ----- Steps ----- #


def excluded_fires(token=None, src_repo=SRC_REPO, roi_dir=ROI_CACHE_DIR):
    """List the window archives TESSERA mode leaves out, from the source repo's ROI CSVs.

    These are the fires without prior-year embeddings. The CSVs are downloaded
    and read once.

    Args:
        token (str | None): a Hugging Face token.
        src_repo (str): the raw GeoTIFF repo.
        roi_dir (str | Path): the local folder for the CSVs.

    Returns:
        list[str]: sorted "split/p<id>.npz" names.
    """
    return tessera_excluded_npz(read_roi(hub.download_roi_csvs(src_repo, token, roi_dir)))


def build_one(gt, fire_id, day_files, start_year, src_repo, token, staging_dir, strip_rows, temp_root,
              cells_per_pixel=TESSERA_CELLS_PER_PIXEL):
    """Download one fire's two day images and build its TESSERA file.

    The images go into a temporary folder that is deleted afterwards. The file
    is written into staging_dir, ready to upload.

    Args:
        gt (geotessera.GeoTesseraZarr): the TESSERA store client.
        fire_id (str): the fire id.
        day_files (tuple[str, ...]): the first and last VIIRS_Day repo paths.
        start_year (int): the year the fire started.
        src_repo (str): the raw GeoTIFF repo.
        token (str | None): a Hugging Face token.
        staging_dir (str | Path): where finished files wait for upload.
        strip_rows (int): TESSERA rows read per request.
        temp_root (str | Path): where the temporary download folder is made.
        cells_per_pixel (int): TESSERA cells along each side of one VIIRS pixel.

    Returns:
        FireTessera: the written file and its coverage.
    """
    with tempfile.TemporaryDirectory(dir=temp_root) as temp_dir:
        day_paths = {
            remote_name: hub.download_file(src_repo, remote_name, token, cache_dir=temp_dir)
            for remote_name in day_files
        }
        return build_fire_tessera(
            gt, fire_id, day_paths, start_year, Path(staging_dir) / f"p{fire_id}.npz", strip_rows=strip_rows,
            cells_per_pixel=cells_per_pixel,
        )


def process_tessera(
    token,
    src_repo=SRC_REPO,
    tessera_repo=TESSERA_REPO,
    splits=SPLITS,
    batch_size=TESSERA_UPLOAD_BATCH,
    work_dir=WORK_DIR,
    roi_dir=ROI_CACHE_DIR,
    overwrite=False,
    store_url=None,
    strip_rows=STRIP_ROWS,
    only_fires=None,
    cells_per_pixel=TESSERA_CELLS_PER_PIXEL,
):
    """Build a TESSERA file for every fire of the chosen splits and upload them.

    The ROI CSVs are read once, a plan decides which fires to build, then each
    fire's file is built and uploaded in batches. An interrupted run resumes by
    skipping fires already uploaded. A fire that fails is recorded with its
    error and the run goes on.

    Args:
        token (str | None): a token with write access to tessera_repo.
        src_repo (str): the raw GeoTIFF repo.
        tessera_repo (str): where the TESSERA files go (created private if new).
        splits (tuple[str, ...]): the splits to cover.
        batch_size (int): files per upload commit.
        work_dir (str | Path): where temporary and staged files go.
        roi_dir (str | Path): the local folder for the ROI CSVs.
        overwrite (bool): rebuild fires already uploaded.
        store_url (str | None): another TESSERA store (default: geotessera's).
        strip_rows (int): TESSERA rows read per request.
        only_fires (list[str] | None): restrict to these ids (a timing test).
        cells_per_pixel (int): TESSERA cells along each side of one VIIRS pixel:
            1 (default) averages each VIIRS pixel into one vector, 5 keeps 5 x 5
            cells of 75 m. Use a separate tessera_repo per value, because fires
            already in the repo are skipped whatever k they were built with.

    Returns:
        TesseraBuildReport: what was built, skipped and uploaded.
    """
    from geotessera import GeoTesseraZarr

    hub.ensure_repo(tessera_repo, token, private=True)  # a new repo is created private
    roi = read_roi(hub.download_roi_csvs(src_repo, token, roi_dir))
    gt = GeoTesseraZarr() if store_url is None else GeoTesseraZarr(store_url)
    print(f"TESSERA store {gt.url}, years {gt.years}"
          + (f", {cells_per_pixel} x {cells_per_pixel} cells per VIIRS pixel" if cells_per_pixel > 1 else ""))
    plan = plan_tessera(roi, splits, hub.list_fire_files(src_repo, token), set(hub.list_repo_files(tessera_repo, token)),
                        gt.years, overwrite, only_fires)
    print(plan.summary())

    work_dir = Path(work_dir)
    temp_root = work_dir / "temp"
    staging_dir = work_dir / "staging" / "tessera"
    temp_root.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    failures, built, uploaded, staged = {}, [], [], 0
    for number, job in enumerate(plan.jobs, start=1):
        print(f"[TESSERA | {number}/{len(plan.jobs)}] Processing: {job.fire_id}")
        started = time.time()
        try:
            result = build_one(gt, job.fire_id, job.day_files, job.start_year, src_repo, token,
                               staging_dir, strip_rows, temp_root, cells_per_pixel)
            print(f"{job.fire_id}: {result.summary()}; built in {(time.time() - started) / 60:.1f} min")
            built.append(job.fire_id)
            staged += 1
        except Exception as error:  # one bad fire must not stop the run
            traceback.print_exc()
            failures[job.fire_id] = f"{type(error).__name__}: {error}"
        gc.collect()
        if staged >= batch_size:
            uploaded += upload_staged(staging_dir, tessera_repo, token)
            staged = 0
    if staged > 0:
        uploaded += upload_staged(staging_dir, tessera_repo, token)

    reasons = {**plan.skipped, **failures}
    skipped = {fire_id: reasons[fire_id] for fire_id in plan.fire_ids if fire_id in reasons}
    report = TesseraBuildReport(tessera_repo, tuple(built), skipped, tuple(uploaded))
    print(report.summary())
    return report


def upload_staged(staging_dir, tessera_repo, token):
    """Upload the staged TESSERA files, then empty the staging folder.

    All files go to tessera_repo/tessera in one commit. The folder is emptied so
    the next batch starts clean.

    Args:
        staging_dir (str | Path): the folder of finished files.
        tessera_repo (str): the TESSERA repo.
        token (str | None): a token with write access.

    Returns:
        list[str]: the uploaded file names.
    """
    print(f"Uploading staged TESSERA files to {tessera_repo}")
    names = hub.upload_folder(staging_dir, tessera_repo, "tessera", token)
    shutil.rmtree(staging_dir, ignore_errors=True)
    Path(staging_dir).mkdir(parents=True, exist_ok=True)
    return names


def upload_tessera_zip(
    token,
    src_repo=SRC_REPO,
    tessera_repo=TESSERA_REPO,
    work_dir=WORK_DIR,
    roi_dir=ROI_CACHE_DIR,
    zip_name=TESSERA_ZIP,
):
    """Bundle every TESSERA file and the drop list into one zip and upload it.

    The zip holds tessera/p<id>.npz plus excluded_npz.txt, the window archives
    of fires without prior-year TESSERA. Training downloads one zip instead of
    hundreds of files. Kept fires still missing a file are reported.

    Args:
        token (str | None): a token with write access to tessera_repo.
        src_repo (str): the raw GeoTIFF repo (for the ROI CSVs).
        tessera_repo (str): the TESSERA repo.
        work_dir (str | Path): where the bundle is assembled.
        roi_dir (str | Path): the local folder for the ROI CSVs.
        zip_name (str): the zip's name in the repo.

    Returns:
        TesseraZipReport: the zip, its contents and any missing fires.

    Raises:
        RuntimeError: if the repo has no TESSERA files yet.
    """
    bundle_root = Path(work_dir) / "tessera_bundle"
    tessera_dir = bundle_root / "tessera"
    tessera_dir.mkdir(parents=True, exist_ok=True)
    remote_files = [
        name for name in hub.list_repo_files(tessera_repo, token)
        if name.startswith("tessera/") and name.endswith(".npz")
    ]
    if not remote_files:
        raise RuntimeError(f"No tessera/*.npz in {tessera_repo}; run process_tessera first")
    print(f"Downloading {len(remote_files)} TESSERA files from {tessera_repo}")
    for name in remote_files:
        if not (bundle_root / name).exists():
            hub.download_file(tessera_repo, name, token, local_dir=bundle_root)

    # The drop list ships inside the zip, so every training run drops the same fires.
    roi = read_roi(hub.download_roi_csvs(src_repo, token, roi_dir))
    excluded = tessera_excluded_npz(roi)
    tessera.write_excluded_list(tessera_dir, excluded)
    missing = missing_tessera(roi, {Path(name).name for name in remote_files}, excluded)

    zip_path = Path(work_dir) / zip_name
    files.zip_folders([tessera_dir], zip_path)
    hub.upload_file(zip_path, zip_name, tessera_repo, token)
    report = TesseraZipReport(zip_path, len(remote_files), tuple(excluded), tuple(missing), tessera_repo)
    print(report.summary())
    return report


def pull_tessera_zip(data_root, token=None, tessera_repo=TESSERA_REPO, zip_name=TESSERA_ZIP, drop_excluded=True):
    """Download and unpack tessera.zip next to the window data, then drop excluded fires.

    Run this after pulling data.zip into the same data root. With
    drop_excluded the window archives listed in excluded_npz.txt are deleted,
    so the TESSERA run and its matched baseline see the same fires.

    Args:
        data_root (str | Path): the data root to unpack into.
        token (str | None): a Hugging Face token.
        tessera_repo (str): the TESSERA repo.
        zip_name (str): the zip's name in the repo.
        drop_excluded (bool): delete the excluded window archives.

    Returns:
        TesseraPull: the TESSERA folder and what was removed.
    """
    hub.pull_zip(tessera_repo, zip_name, data_root, token)
    tessera_dir = Path(data_root) / "tessera"
    excluded = tessera.read_excluded_list(tessera_dir)
    removed = tessera.drop_excluded_archives(data_root, excluded) if drop_excluded else []
    report = TesseraPull(tessera_dir, tuple(excluded), tuple(removed))
    print(report.summary())
    return report


def stage_tessera(data_root, work, token=None, tessera_repo=TESSERA_REPO, attached_dir=None, attached_zip=None):
    """Prepare TESSERA for a training notebook and build the training view.

    The TESSERA files come from an attached folder, an attached tessera.zip, or
    a download from the repo, in that order. The view links the kept window
    archives plus tessera/, leaving out excluded fires without modifying the
    data.

    Args:
        data_root (str | Path): the extracted window data.
        work (Workspace): the notebook's paths (windows_master.workspace).
        token (str | None): a Hugging Face token.
        tessera_repo (str): the TESSERA repo.
        attached_dir (str | Path | None): an attached folder of TESSERA files.
        attached_zip (str | Path | None): an attached tessera.zip.

    Returns:
        TesseraView: the training root and the archives left out.
    """
    if attached_dir:
        tessera_dir = Path(attached_dir)
    else:
        archive = Path(attached_zip) if attached_zip else Path(
            hub.download_file(tessera_repo, TESSERA_ZIP, token, local_dir=work.download_dir)
        )
        tessera_dir = files.extract_zip(archive, work.tessera_extract_dir) / "tessera"
    print("TESSERA files:", sum(1 for _ in tessera_dir.glob("*.npz")), "in", tessera_dir)
    view = tessera.build_tessera_view(data_root, tessera_dir, work.view_dir)
    print(view.summary())
    return view


def download_tessera_files(data_root, tessera_repo, token):
    """Download every TESSERA file into a data root and list archives still without one.

    Files already present are kept. Every window archive needs a TESSERA file
    of the same name; the ones without are returned so the caller can warn.

    Args:
        data_root (str | Path): the data root (files go to data_root/tessera).
        tessera_repo (str): the TESSERA repo.
        token (str | None): a Hugging Face token.

    Returns:
        list[str]: "split/p<id>.npz" archives that have no TESSERA file.
    """
    data_root = Path(data_root)
    for name in hub.list_repo_files(tessera_repo, token):
        if name.startswith("tessera/") and not (data_root / name).exists():
            hub.download_file(tessera_repo, name, token, local_dir=data_root)
    return [
        f"{split}/{archive.name}"
        for split in SPLITS
        for archive in sorted((data_root / split).glob("*.npz"))
        if not (data_root / "tessera" / archive.name).exists()
    ]
