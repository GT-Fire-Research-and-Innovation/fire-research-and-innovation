"""MASTER for TESSERA: every TESSERA step a notebook or the CLI runs.

Level 1 (prep). Notebooks call only this module and prep/dataset_master.py.
Each function runs one pipeline step end to end and returns a frozen report
(with .summary()). Progress is printed here and nowhere below. [ours]

    process_tessera     build tessera/p<id>.npz per fire, upload to TESSERA_REPO
    upload_tessera_zip  bundle them (+ excluded_npz.txt) into tessera.zip there
    pull_tessera_zip    unpack tessera.zip beside window data, drop excluded fires
    stage_tessera       training notebooks: extract (or reuse) TESSERA, build the view
    excluded_fires      the window archives TESSERA mode leaves out

TESSERA embeddings come from the year before the fire (prep/tessera_grid.py);
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

from firepred.config.data import (
    ROI_CACHE_DIR,
    SPLITS,
    SRC_REPO,
    STRIP_ROWS,
    TESSERA_REPO,
    TESSERA_UPLOAD_BATCH,
    TESSERA_ZIP,
    WORK_DIR,
)
from firepred.prep import repos, staging
from firepred.prep.roi import has_prior_year_tessera, read_roi, tessera_excluded_npz
from firepred.prep.tessera_grid import build_fire_tessera


# ----- Reports ----- #


@dataclass(frozen=True)
class TesseraBuildReport:
    """process_tessera's result: fires built, fires skipped (with reasons), files uploaded."""

    repo: str
    built: tuple
    skipped: dict = field(default_factory=dict)
    uploaded: tuple = ()

    def summary(self):
        """Counts plus one line per skipped fire."""
        lines = [f"TESSERA done in {self.repo}: built {len(self.built)}, skipped {len(self.skipped)}"]
        lines += [f"  {fire_id}: {reason}" for fire_id, reason in self.skipped.items()]
        return "\n".join(lines)


@dataclass(frozen=True)
class TesseraZipReport:
    """upload_tessera_zip's result: the zip, what it holds, and kept fires still missing a file."""

    zip_path: Path
    embeddings: int
    excluded: tuple
    missing: tuple
    repo: str

    def summary(self):
        """One line, plus a warning when kept fires have no TESSERA file yet."""
        text = (f"Uploaded {self.zip_path.name} to {self.repo}: {self.embeddings} embeddings, "
                f"{len(self.excluded)} excluded window archives")
        if self.missing:
            text += f"\nWARNING: {len(self.missing)} kept fires have no TESSERA file yet: {list(self.missing)}"
        return text


@dataclass(frozen=True)
class TesseraPull:
    """pull_tessera_zip's result: the unpacked tessera/ folder and the window archives removed."""

    tessera_dir: Path
    excluded: tuple
    removed: tuple

    def summary(self):
        """One line."""
        return (f"TESSERA in {self.tessera_dir}; removed {len(self.removed)} window archives "
                "of fires without prior-year TESSERA")


# ----- Planning (pure) ----- #


def fires_to_build(roi, splits, only_fires=None):
    """Fire IDs of `splits`, de-duplicated in order, narrowed to `only_fires` if given."""
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
    """Why `fire_id` cannot get a TESSERA file, or "" when it can."""
    if not day_files:
        return "no VIIRS_Day files in source repo"
    if fire_id not in years:
        return "no start_date in ROI CSVs"
    if not has_prior_year_tessera(years[fire_id]) or years[fire_id] - 1 not in available_years:
        return f"no TESSERA for prior year {years[fire_id] - 1}"
    return ""


def missing_tessera(roi, present_names, excluded):
    """Kept window archives ('split/p<id>.npz') whose fire has no TESSERA file in `present_names`."""
    return sorted(
        f"{split}/p{fire_id}.npz"
        for split, fire_ids in roi.ids_by_split().items()
        for fire_id in fire_ids
        if f"{split}/p{fire_id}.npz" not in excluded and f"p{fire_id}.npz" not in present_names
    )


# ----- Steps ----- #


def excluded_fires(token=None, src_repo=SRC_REPO, roi_dir=ROI_CACHE_DIR):
    """The window archives TESSERA mode drops (fires without prior-year TESSERA)."""
    return tessera_excluded_npz(read_roi(repos.download_roi_csvs(src_repo, token, roi_dir)))


def build_one(gt, fire_id, day_files, start_year, src_repo, token, staging_dir, strip_rows, temp_root):
    """Download a fire's first and last VIIRS_Day GeoTIFF, build its TESSERA file; return a FireTessera.

    The two days must share one grid, and that grid decides which TESSERA
    pixels the fire's crop covers. Downloads go to a temporary folder that is
    deleted afterwards.
    """
    with tempfile.TemporaryDirectory(dir=temp_root) as temp_dir:
        day_paths = {
            remote_name: repos.download_file(src_repo, remote_name, token, cache_dir=temp_dir)
            for remote_name in dict.fromkeys((day_files[0], day_files[-1]))
        }
        return build_fire_tessera(
            gt, fire_id, day_paths, start_year, Path(staging_dir) / f"p{fire_id}.npz", strip_rows=strip_rows
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
):
    """Build tessera/p<id>.npz for every fire in `splits` and upload them to `tessera_repo`.

    `only_fires` restricts the run to those fire IDs, e.g. a short timing test.
    Only the first and last VIIRS_Day GeoTIFF of each fire are downloaded;
    embeddings are streamed into RAM. Finished files are staged in
    <work_dir>/staging/tessera and uploaded in batches, so an interrupted run
    resumes by skipping fires already uploaded. A failing fire is recorded in
    the report's `skipped` and the run goes on.
    """
    from geotessera import GeoTesseraZarr

    repos.ensure_repo(tessera_repo, token, private=True)  # a new repo is created private
    roi = read_roi(repos.download_roi_csvs(src_repo, token, roi_dir))
    locations = repos.list_fire_files(src_repo, token)
    existing = set(repos.list_repo_files(tessera_repo, token))
    gt = GeoTesseraZarr() if store_url is None else GeoTesseraZarr(store_url)
    print(f"TESSERA store {gt.url}, years {gt.years}")

    work_dir = Path(work_dir)
    temp_root = work_dir / "temp"
    staging_dir = work_dir / "staging" / "tessera"
    temp_root.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    fire_ids = fires_to_build(roi, splits, only_fires)
    reasons, built, uploaded, staged = {}, [], [], 0
    for fire_number, fire_id in enumerate(fire_ids, start=1):
        if f"tessera/p{fire_id}.npz" in existing and not overwrite:
            print(f"Skipping {fire_id}: already processed")
            continue
        day_files = locations.get(fire_id, {}).get("VIIRS_Day", [])
        reason = skip_reason(fire_id, day_files, roi.start_years, gt.years)
        if reason:
            reasons[fire_id] = reason
            continue

        print(f"[TESSERA | {fire_number}/{len(fire_ids)}] Processing: {fire_id}")
        started = time.time()
        try:
            result = build_one(gt, fire_id, day_files, roi.start_years[fire_id], src_repo, token,
                               staging_dir, strip_rows, temp_root)
            print(f"{fire_id}: {result.summary()}; built in {(time.time() - started) / 60:.1f} min")
            built.append(fire_id)
            staged += 1
        except Exception as error:  # one bad fire must not stop the run
            traceback.print_exc()
            reasons[fire_id] = f"{type(error).__name__}: {error}"
        gc.collect()

        if staged >= batch_size:
            uploaded += upload_staged(staging_dir, tessera_repo, token)
            staged = 0
    if staged > 0:
        uploaded += upload_staged(staging_dir, tessera_repo, token)

    skipped = {fire_id: reasons[fire_id] for fire_id in fire_ids if fire_id in reasons}
    report = TesseraBuildReport(tessera_repo, tuple(built), skipped, tuple(uploaded))
    print(report.summary())
    return report


def upload_staged(staging_dir, tessera_repo, token):
    """Upload the staged TESSERA files to tessera_repo/tessera, then empty the folder; return their names."""
    print(f"Uploading staged TESSERA files to {tessera_repo}")
    names = repos.upload_folder(staging_dir, tessera_repo, "tessera", token)
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
    """Bundle tessera/*.npz from `tessera_repo` into `zip_name` and upload it there.

    The zip holds tessera/p<id>.npz plus tessera/excluded_npz.txt, the window
    archives of fires without prior-year TESSERA. Window data is not copied:
    training pulls data.zip from the window repo and this zip into the same folder.
    """
    bundle_root = Path(work_dir) / "tessera_bundle"
    tessera_dir = bundle_root / "tessera"
    tessera_dir.mkdir(parents=True, exist_ok=True)
    remote_files = [
        name for name in repos.list_repo_files(tessera_repo, token)
        if name.startswith("tessera/") and name.endswith(".npz")
    ]
    if not remote_files:
        raise RuntimeError(f"No tessera/*.npz in {tessera_repo}; run process_tessera first")
    print(f"Downloading {len(remote_files)} TESSERA files from {tessera_repo}")
    for name in remote_files:
        if not (bundle_root / name).exists():
            repos.download_file(tessera_repo, name, token, local_dir=bundle_root)

    # The drop list ships inside the zip, so every training run drops the same fires.
    roi = read_roi(repos.download_roi_csvs(src_repo, token, roi_dir))
    excluded = tessera_excluded_npz(roi)
    staging.write_excluded_list(tessera_dir, excluded)
    missing = missing_tessera(roi, {Path(name).name for name in remote_files}, excluded)

    zip_path = Path(work_dir) / zip_name
    staging.zip_folders([tessera_dir], zip_path)
    repos.upload_file(zip_path, zip_name, tessera_repo, token)
    report = TesseraZipReport(zip_path, len(remote_files), tuple(excluded), tuple(missing), tessera_repo)
    print(report.summary())
    return report


def pull_tessera_zip(data_root, token=None, tessera_repo=TESSERA_REPO, zip_name=TESSERA_ZIP, drop_excluded=True):
    """Unpack tessera.zip into <data_root>/tessera; with `drop_excluded`, delete the excluded window archives.

    Run after pulling data.zip into the same `data_root`. Deleting the
    archives listed in tessera/excluded_npz.txt makes the TESSERA run and the
    matched no-TESSERA run see the same fires.
    """
    repos.pull_zip(tessera_repo, zip_name, data_root, token)
    tessera_dir = Path(data_root) / "tessera"
    excluded = staging.read_excluded_list(tessera_dir)
    removed = staging.drop_excluded_archives(data_root, excluded) if drop_excluded else []
    report = TesseraPull(tessera_dir, tuple(excluded), tuple(removed))
    print(report.summary())
    return report


def stage_tessera(data_root, work, token=None, tessera_repo=TESSERA_REPO, attached_dir=None, attached_zip=None):
    """Training notebooks: find (or download and extract) the TESSERA files and build the training view.

    `work` is a dataset_master.Workspace. Priority: an attached folder, an
    attached tessera.zip, else tessera.zip from `tessera_repo`. Returns a
    staging.TesseraView whose root is the training root (window archives of
    excluded fires left out, tessera/ linked in).
    """
    if attached_dir:
        tessera_dir = Path(attached_dir)
    else:
        archive = Path(attached_zip) if attached_zip else Path(
            repos.download_file(tessera_repo, TESSERA_ZIP, token, local_dir=work.download_dir)
        )
        tessera_dir = staging.extract_zip(archive, work.tessera_extract_dir) / "tessera"
    print("TESSERA files:", sum(1 for _ in tessera_dir.glob("*.npz")), "in", tessera_dir)
    view = staging.build_tessera_view(data_root, tessera_dir, work.view_dir)
    print(view.summary())
    return view
