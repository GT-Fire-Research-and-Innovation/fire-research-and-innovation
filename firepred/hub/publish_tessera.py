"""Build every fire's prior-year TESSERA file and publish them (and tessera.zip) on Hugging Face.

Level 2 (hub). Used by cli/dataset.py (`tessera`, `tessera-zip`) and the
data-grab notebook. The per-fire computation is prep/tessera_grid.py. [ours]
"""

import gc
import os
import shutil
import time
import traceback

from huggingface_hub import HfApi, hf_hub_download, login

from firepred.config.dataset import SPLITS, SRC_REPO, TESSERA_EXCLUDED_LIST, TESSERA_REPO
from firepred.hub.repos import list_fire_files
from firepred.hub.splits import download_roi_csvs, fire_start_years, obtain_ids, tessera_excluded_npz
from firepred.hub.zips import zip_data_from, zip_data_to
from firepred.prep.tessera_grid import build_fire_tessera, has_prior_year_tessera


def process_tessera(
    token,
    src_repo=SRC_REPO,
    tessera_repo=TESSERA_REPO,
    splits=SPLITS,
    batch_size=1,
    work_dir=".",
    roi_dir="hf_roi",
    overwrite=False,
    store_url=None,
    strip_rows=128,
    only_fires=None,
):
    """Build tessera/p<id>.npz for every fire in `splits` and upload them to `tessera_repo`.

    `only_fires` restricts the run to those fire IDs, e.g. a short timing test.

    Only the first and last VIIRS_Day GeoTIFF of each fire are downloaded
    (into <work_dir>/temp, deleted after each fire); embeddings are streamed
    into RAM. Finished files are staged in <work_dir>/staging/tessera and
    uploaded to `tessera_repo` in batches, so an interrupted run resumes by
    skipping fires already uploaded. Returns {fire_id: reason} for skipped
    fires.
    """
    from geotessera import GeoTesseraZarr

    # Connect and make sure the TESSERA repo exists.
    login(token)
    api = HfApi(token=token)
    # A new repo is created private; an existing one keeps its visibility.
    api.create_repo(repo_id=tessera_repo, repo_type="dataset", private=True, exist_ok=True)

    # Inputs: the fire split, each fire's start year (TESSERA uses the year
    # before), the source GeoTIFF list, and what is already uploaded.
    ids_by_split = dict(zip(SPLITS, obtain_ids(src_repo, token, roi_dir)))
    years = fire_start_years(download_roi_csvs(src_repo, token, roi_dir))
    locations = list_fire_files(api, src_repo)
    existing_files = set(api.list_repo_files(tessera_repo, repo_type="dataset"))

    # No cache_dir: the store is read over HTTP straight into memory.
    gt = GeoTesseraZarr() if store_url is None else GeoTesseraZarr(store_url)
    print(f"TESSERA store {gt.url}, years {gt.years}")

    temp_dir = os.path.join(work_dir, "temp")
    staging_dir = os.path.join(work_dir, "staging", "tessera")
    os.makedirs(staging_dir, exist_ok=True)

    # Fire IDs of the requested splits, de-duplicated in order; only_fires
    # narrows them (e.g. a short timing test).
    fire_ids = list(dict.fromkeys(
        str(fire_id) for split in splits for fire_id in ids_by_split[split]
    ))
    if only_fires is not None:
        wanted = {str(fire_id) for fire_id in only_fires}
        unknown = wanted.difference(fire_ids)
        if unknown:
            raise ValueError(f"Fires not in splits {splits}: {sorted(unknown)}")
        fire_ids = [fire_id for fire_id in fire_ids if fire_id in wanted]
    skipped = {}
    staged_count = 0

    def upload_staged():
        print(f"Uploading batch of {staged_count} TESSERA files to Hugging Face")
        api.upload_folder(
            folder_path=staging_dir,
            repo_id=tessera_repo,
            repo_type="dataset",
            path_in_repo="tessera",
        )
        shutil.rmtree(staging_dir, ignore_errors=True)
        os.makedirs(staging_dir, exist_ok=True)

    for fire_idx, fire_id in enumerate(fire_ids):
        # Skip fires already uploaded, and fires that cannot have an embedding (no
        # imagery, no start date, or no TESSERA year before the fire).
        if f"tessera/p{fire_id}.npz" in existing_files and not overwrite:
            print(f"Skipping {fire_id}: already processed")
            continue

        day_files = locations.get(fire_id, {}).get("VIIRS_Day", [])
        if not day_files:
            skipped[fire_id] = "no VIIRS_Day files in source repo"
            continue
        if fire_id not in years:
            skipped[fire_id] = "no start_date in ROI CSVs"
            continue
        if not has_prior_year_tessera(years[fire_id]) or years[fire_id] - 1 not in gt.years:
            skipped[fire_id] = f"no TESSERA for prior year {years[fire_id] - 1}"
            continue

        print(f"[TESSERA | {fire_idx + 1}/{len(fire_ids)}] Processing: {fire_id}")
        started = time.time()
        try:
            os.makedirs(temp_dir, exist_ok=True)
            # The first and last day's GeoTIFF are enough: they must share one grid, and
            # that grid decides which TESSERA pixels the fire's crop covers.
            day_paths = {
                remote_name: hf_hub_download(
                    repo_id=src_repo,
                    filename=remote_name,
                    repo_type="dataset",
                    cache_dir=temp_dir,
                    token=token,
                )
                for remote_name in dict.fromkeys((day_files[0], day_files[-1]))
            }
            build_fire_tessera(
                gt,
                fire_id,
                day_paths,
                years[fire_id],
                os.path.join(staging_dir, f"p{fire_id}.npz"),
                strip_rows=strip_rows,
            )
            staged_count += 1
            print(f"{fire_id}: built in {(time.time() - started) / 60:.1f} min")
        # A failing fire is recorded and skipped instead of stopping the whole run;
        # its temp files are always deleted.
        except Exception as error:
            traceback.print_exc()
            skipped[fire_id] = f"{type(error).__name__}: {error}"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            gc.collect()

        if staged_count >= batch_size:
            upload_staged()
            staged_count = 0

    if staged_count > 0:
        upload_staged()

    print(f"TESSERA done; skipped {len(skipped)} fires")
    for fire_id, reason in skipped.items():
        print(f"  {fire_id}: {reason}")
    return skipped


def upload_tessera_zip(
    token,
    src_repo=SRC_REPO,
    tessera_repo=TESSERA_REPO,
    work_dir=".",
    roi_dir="hf_roi",
    zip_name="tessera.zip",
):
    """Bundle tessera/*.npz from `tessera_repo` into `zip_name` and upload it there.

    The zip holds tessera/p<id>.npz plus tessera/excluded_npz.txt, the window
    archives of fires without prior-year TESSERA. Window data is not copied:
    training pulls data.zip from WINDOW_REPO and this zip into the same folder.
    """
    bundle_root = os.path.join(work_dir, "tessera_bundle")
    tessera_dir = os.path.join(bundle_root, "tessera")
    os.makedirs(tessera_dir, exist_ok=True)

    api = HfApi(token=token)
    remote_files = [
        file_path
        for file_path in api.list_repo_files(tessera_repo, repo_type="dataset")
        if file_path.startswith("tessera/") and file_path.endswith(".npz")
    ]
    if not remote_files:
        raise RuntimeError(f"No tessera/*.npz in {tessera_repo}; run process_tessera first")

    print(f"Downloading {len(remote_files)} TESSERA files from {tessera_repo}")
    for file_path in remote_files:
        if not os.path.exists(os.path.join(bundle_root, file_path)):
            hf_hub_download(
                repo_id=tessera_repo,
                filename=file_path,
                repo_type="dataset",
                local_dir=bundle_root,
                token=token,
            )

    # Ship the list of window archives without TESSERA inside the zip, so every
    # training run drops the same fires.
    excluded = tessera_excluded_npz(token, src_repo, roi_dir)
    with open(os.path.join(tessera_dir, TESSERA_EXCLUDED_LIST), "w") as handle:
        handle.write("\n".join(excluded) + "\n")

    # Every fire that is not dropped needs an embedding before training.
    ids_by_split = dict(zip(SPLITS, obtain_ids(src_repo, token, roi_dir)))
    present = {os.path.basename(file_path) for file_path in remote_files}
    missing = sorted(
        f"{split}/p{fire_id}.npz"
        for split, fire_ids in ids_by_split.items()
        for fire_id in fire_ids
        if f"{split}/p{fire_id}.npz" not in excluded and f"p{fire_id}.npz" not in present
    )
    if missing:
        print(f"WARNING: {len(missing)} kept fires have no TESSERA file yet: {missing}")

    # Zip tessera/ and upload it to the TESSERA repo.
    zip_path = os.path.join(work_dir, zip_name)
    zip_data_from([tessera_dir], zip_path)
    zip_data_to(token, zip_path, tessera_repo, zip_name)
    print(
        f"Uploaded {zip_name} to {tessera_repo}: {len(remote_files)} embeddings, "
        f"{len(excluded)} excluded window archives"
    )
    return missing
