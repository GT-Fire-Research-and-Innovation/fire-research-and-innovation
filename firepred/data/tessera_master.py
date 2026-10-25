"""MASTER for TESSERA: every TESSERA step a notebook or the CLI runs.

Level 1 (data, dataset building). Notebooks call only this module and data/windows_master.py.
Each function runs one pipeline step end to end and returns a frozen report
(with .summary()). Progress is printed here and nowhere below. [ours]

    process_tessera     build tessera/p<id>.npz per fire, upload to TESSERA_REPO
    convert_tessera_repo  copy a repo's files into a new repo as int8 (or float16), a batch at a time
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
    TESSERA_CONVERT_BATCH,
    TESSERA_QUANTIZATION,
    TESSERA_QUANTIZATIONS,
    TESSERA_REPO,
    TESSERA_UPLOAD_BATCH,
    TESSERA_ZIP,
    WORK_DIR,
)
from firepred.data import files, hub, tessera
from firepred.data.archive import convert_tessera_npz
from firepred.data.files import read_roi
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
        """Summarize the build: counts and one line per skipped fire."""
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
        """Summarize the upload, with a warning when kept fires lack a file."""
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
        """Summarize the pull in one line."""
        return (f"TESSERA in {self.tessera_dir}; removed {len(self.removed)} window archives "
                "of fires without prior-year TESSERA")


@dataclass(frozen=True)
class TesseraConvertReport:
    """Report what convert_tessera_repo converted, skipped and uploaded.

    The sizes cover the files converted in this run, so the summary shows the
    space saved. Failed files carry their error and can be retried by running
    again (files already in the destination are skipped).

    Attributes:
        source_repo (str): the repo the files came from (unchanged).
        dest_repo (str): the repo the converted files went to.
        quantization (str): the storage they were converted to.
        converted (tuple[str, ...]): repo paths converted in this run.
        already (tuple[str, ...]): repo paths already in dest_repo, left alone.
        failed (dict[str, str]): {repo path: error}.
        uploaded (tuple[str, ...]): file names uploaded.
        bytes_before (int): total size of the converted files in source_repo.
        bytes_after (int): their total size after conversion.
    """

    source_repo: str
    dest_repo: str
    quantization: str
    converted: tuple
    already: tuple = ()
    failed: dict = field(default_factory=dict)
    uploaded: tuple = ()
    bytes_before: int = 0
    bytes_after: int = 0

    def summary(self):
        """Summarize the conversion: counts, sizes and one line per failed file."""
        lines = [f"Converted {len(self.converted)} TESSERA files {self.source_repo} -> {self.dest_repo} "
                 f"as {self.quantization}: {self.bytes_before / 1e9:.2f} -> {self.bytes_after / 1e9:.2f} GB; "
                 f"{len(self.already)} already there, {len(self.failed)} failed"]
        lines += [f"  {name}: {reason}" for name, reason in self.failed.items()]
        return "\n".join(lines)


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
        """Summarize the plan in one line."""
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


def plan_conversion(source_files, dest_files, overwrite=False, only_fires=None):
    """Pick the TESSERA files to convert: the source's tessera/p<id>.npz not yet in the destination.

    Only per-fire files are converted; tessera.zip and anything else in the
    source repo are left out. Files already in the destination are skipped
    unless overwrite is set, so a stopped run resumes. This is pure.

    Args:
        source_files (Iterable[str]): file paths in the source repo.
        dest_files (Iterable[str]): file paths in the destination repo.
        overwrite (bool): convert files already in the destination again.
        only_fires (list[str] | None): restrict to these fire ids.

    Returns:
        tuple[tuple[str, ...], tuple[str, ...]]: (repo paths to convert,
        repo paths already in the destination), both sorted.
    """
    wanted = sorted(name for name in source_files if name.startswith("tessera/p") and name.endswith(".npz"))
    if only_fires is not None:
        chosen = {f"tessera/p{fire_id}.npz" for fire_id in only_fires}
        wanted = [name for name in wanted if name in chosen]
    present = set(dest_files)
    todo = tuple(name for name in wanted if overwrite or name not in present)
    already = tuple(name for name in wanted if not overwrite and name in present)
    return todo, already


# ----- Steps ----- #


def excluded_fires(token=None, src_repo=SRC_REPO, roi_dir=ROI_CACHE_DIR):
    """List the window archives TESSERA mode leaves out, from the source repo's ROI CSVs."""
    return tessera_excluded_npz(read_roi(hub.download_roi_csvs(src_repo, token, roi_dir)))


def build_one(gt, fire_id, day_files, start_year, src_repo, token, staging_dir, strip_rows, temp_root,
              cells_per_pixel=TESSERA_CELLS_PER_PIXEL, quantization=TESSERA_QUANTIZATION):
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
        quantization (str): "none" (float16) or "int8" storage of the embedding.

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
            cells_per_pixel=cells_per_pixel, quantization=quantization,
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
    quantization=TESSERA_QUANTIZATION,
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
        quantization (str): "none" (default) stores float16; "int8" stores int8
            codes plus a scale and offset per band (about 2.3x smaller at k = 5,
            RMS error under 0.02 band standard deviations). Both read the same, but
            fires already in the repo keep their storage unless overwrite is set.

    Returns:
        TesseraBuildReport: what was built, skipped and uploaded.
    """
    from geotessera import GeoTesseraZarr

    hub.ensure_repo(tessera_repo, token, private=True)  # a new repo is created private
    roi = read_roi(hub.download_roi_csvs(src_repo, token, roi_dir))
    gt = GeoTesseraZarr() if store_url is None else GeoTesseraZarr(store_url)
    print(f"TESSERA store {gt.url}, years {gt.years}"
          + (f", {cells_per_pixel} x {cells_per_pixel} cells per VIIRS pixel" if cells_per_pixel > 1 else "")
          + (f", stored as {quantization}" if quantization != "none" else ""))
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
                               staging_dir, strip_rows, temp_root, cells_per_pixel, quantization=quantization)
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
    """Upload the staged TESSERA files, then empty the staging folder."""
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


def convert_one(tessera_repo, name, token, staging_dir, temp_root, quantization):
    """Download one TESSERA file, convert it into staging_dir, and delete the download at once.

    The download goes into its own temporary cache folder, removed when the
    file is converted, so no copy stays in the Hugging Face cache. A failed
    conversion leaves nothing in staging_dir, so no broken file is uploaded.

    Args:
        tessera_repo (str): the source repo.
        name (str): the repo path, tessera/p<id>.npz.
        token (str | None): a Hugging Face token.
        staging_dir (str | Path): where converted files wait for upload.
        temp_root (str | Path): where the temporary download folder is made.
        quantization (str): the storage to convert to.

    Returns:
        tuple[int, int]: the file's size in bytes before and after conversion.
    """
    target = Path(staging_dir) / Path(name).name
    with tempfile.TemporaryDirectory(dir=temp_root) as temp_dir:
        source = Path(hub.download_file(tessera_repo, name, token, cache_dir=temp_dir))
        try:
            convert_tessera_npz(source, target, quantization, {"converted_from": tessera_repo})
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return source.stat().st_size, target.stat().st_size


def convert_tessera_repo(
    token,
    dest_repo,
    tessera_repo=TESSERA_REPO,
    quantization="int8",
    batch_size=TESSERA_CONVERT_BATCH,
    work_dir=WORK_DIR,
    overwrite=False,
    only_fires=None,
):
    """Copy every TESSERA file of one repo into a new repo with another storage, a batch at a time.

    Made to re-encode float16 files as int8 without rebuilding them from the
    TESSERA store. Disk use stays small: each file is downloaded into a
    temporary folder, converted into the staging folder and its download
    deleted at once; every batch_size converted files are uploaded in one
    commit and deleted. So at most one download plus one batch of converted
    files is on disk (about 0.4 GB + batch_size x 0.16 GB for int8 at 5 x 5
    cells). The source repo is never changed. Files already in dest_repo are
    skipped, so a stopped run resumes; a file that fails is recorded and the
    run goes on. tessera.zip is not copied: run upload_tessera_zip on dest_repo
    afterwards.

    Args:
        token (str | None): a token that reads tessera_repo and writes dest_repo.
        dest_repo (str): where the converted files go (created private if new); not tessera_repo.
        tessera_repo (str): the repo whose tessera/p<id>.npz files are converted.
        quantization (str): the storage to convert to: "int8" (default) or "none" (float16).
        batch_size (int): converted files per upload commit, the most kept on disk at once.
        work_dir (str | Path): where the downloads and converted files go.
        overwrite (bool): convert files already in dest_repo again.
        only_fires (list[str] | None): restrict to these fire ids (a test run).

    Returns:
        TesseraConvertReport: what was converted, skipped and uploaded, and the sizes.

    Raises:
        ValueError: if dest_repo is missing or is tessera_repo, the quantization
            is unknown, or batch_size is below 1.
    """
    if not dest_repo or dest_repo == tessera_repo:
        raise ValueError(f"Give a destination repo other than the source {tessera_repo}, got {dest_repo!r}")
    if quantization not in TESSERA_QUANTIZATIONS:
        raise ValueError(f"Unknown TESSERA quantization {quantization!r}; choose from {TESSERA_QUANTIZATIONS}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    hub.ensure_repo(dest_repo, token, private=True)  # a new repo is created private
    todo, already = plan_conversion(hub.list_repo_files(tessera_repo, token), hub.list_repo_files(dest_repo, token),
                                    overwrite, only_fires)
    print(f"Converting {len(todo)} TESSERA files from {tessera_repo} to {dest_repo} as {quantization} "
          f"({len(already)} already there), {batch_size} per upload")

    work_dir = Path(work_dir)
    temp_root = work_dir / "temp"
    staging_dir = work_dir / "staging" / "tessera-convert"
    # Files left by a stopped run were never uploaded and may be incomplete: convert them again.
    shutil.rmtree(staging_dir, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    converted, failed, uploaded, staged, before, after = [], {}, [], 0, 0, 0
    for number, name in enumerate(todo, start=1):
        try:
            size_before, size_after = convert_one(tessera_repo, name, token, staging_dir, temp_root, quantization)
            print(f"[TESSERA convert | {number}/{len(todo)}] {name}: "
                  f"{size_before / 1e6:.0f} -> {size_after / 1e6:.0f} MB")
            converted.append(name)
            before, after, staged = before + size_before, after + size_after, staged + 1
        except Exception as error:  # one bad file must not stop the run
            traceback.print_exc()
            failed[name] = f"{type(error).__name__}: {error}"
        gc.collect()
        if staged >= batch_size:
            uploaded += upload_staged(staging_dir, dest_repo, token)
            staged = 0
    if staged > 0:
        uploaded += upload_staged(staging_dir, dest_repo, token)

    report = TesseraConvertReport(tessera_repo, dest_repo, quantization, tuple(converted), already, failed,
                                  tuple(uploaded), before, after)
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
