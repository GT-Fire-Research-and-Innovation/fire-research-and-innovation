"""Build the window dataset: stream raw fires from Hugging Face, upload windowed archives.

Level 2 (hub). Used by cli/dataset.py (`process`) and the data-grab notebook.

    SRC_REPO raw GeoTIFFs --download one fire--> prep/windows.build_fire_windows
        --> data/archive.write_fire_archive p<id>.npz --upload in batches--> window repo
"""

import gc
import os
import shutil

from huggingface_hub import HfApi, hf_hub_download, login

from firepred.config.dataset import SPLITS, SRC_REPO, TRAIN_VAL_LABEL_SEL, TS_INTERVAL, TS_LENGTH, window_repo_for
from firepred.data.archive import write_fire_archive
from firepred.hub.repos import ensure_window_config, list_fire_files, window_settings
from firepred.hub.splits import download_roi_csvs, fire_label_sel, obtain_ids
from firepred.prep.windows import build_fire_windows


class DatasetStreamer:
    """Stream GeoTIFFs into one window-addressable NPZ per fire."""

    def __init__(self, src_repo, dest_repo, token, batch_size=10, work_dir="."):
        self.src_repo = src_repo
        self.dest_repo = dest_repo
        self.token = token
        self.batch_size = batch_size
        self.api = HfApi(token=token)
        login(token)

        self.temp_dir = os.path.join(work_dir, "temp")
        os.makedirs(self.temp_dir, exist_ok=True)

        self.staging_dir = os.path.join(work_dir, "staging")
        os.makedirs(self.staging_dir, exist_ok=True)

        self.api.create_repo(
            repo_id=self.dest_repo,
            repo_type="dataset",
            exist_ok=True,
        )

    def get_repo_locations(self):
        """Scan source repository and group GeoTIFFs by fire and mode."""
        return list_fire_files(self.api, self.src_repo)

    def run_split(
        self,
        target_ids,
        split_name,
        length=TS_LENGTH,
        interval=TS_INTERVAL,
        label_sel_by_id=None,
        overwrite=False,
        debug=False,
    ):
        """Process `target_ids` into <split_name>/p<id>.npz in the dest repo.

        Fires already in the repo are skipped unless `overwrite` is set.
        """
        locations = self.get_repo_locations()
        # What the destination repo already holds: finished fires are skipped, so an
        # interrupted run resumes where it stopped.
        existing_files = set(
            self.api.list_repo_files(
                self.dest_repo,
                repo_type="dataset",
            )
        )

        # Built archives wait in a local staging folder and are uploaded in batches
        # (one Hugging Face commit per batch instead of one per fire).
        current_staging_dir = os.path.join(
            self.staging_dir,
            split_name,
        )
        os.makedirs(current_staging_dir, exist_ok=True)
        staged_count = 0

        # Only fires that have GeoTIFFs in the source repo.
        valid_locations = [
            location
            for location in target_ids
            if location in locations
        ]
        print(
            f"Found {len(valid_locations)} valid locations that match."
        )

        for loc_idx, loc_name in enumerate(valid_locations):
            dest_filename = f"{split_name}/p{loc_name}.npz"

            if dest_filename in existing_files and not overwrite:
                print(f"Skipping {loc_name}: already processed")
                continue

            print(
                f"[{split_name.upper()} | {loc_idx + 1}/"
                f"{len(valid_locations)}] Processing: {loc_name}"
            )

            os.makedirs(self.temp_dir, exist_ok=True)
            # Download this fire's GeoTIFFs (every mode) to a temp folder: remote path -> local path.
            local_map = {}
            for paths in locations[loc_name].values():
                for path in paths:
                    local_map[path] = hf_hub_download(
                        repo_id=self.src_repo,
                        filename=path,
                        repo_type="dataset",
                        cache_dir=self.temp_dir,
                        token=self.token,
                    )

            # This fire's label rule: 0 = active fire + NIFC perimeters, 1 = active fire only.
            label_sel = TRAIN_VAL_LABEL_SEL
            if label_sel_by_id is not None:
                label_sel = int(label_sel_by_id.get(loc_name, TRAIN_VAL_LABEL_SEL))

            # Daily GeoTIFFs -> sliding windows: inputs (N, C, T, H, W) and next-day targets (N, H, W).
            loc_x, loc_y = build_fire_windows(
                loc_name,
                local_map,
                length=length,
                interval=interval,
                label_sel=label_sel,
                debug=debug,
            )

            # One archive per fire; each window is its own member, so training can read
            # one window without unpacking the rest.
            if loc_x is not None and len(loc_x) > 0:
                save_path = os.path.join(
                    current_staging_dir,
                    f"p{loc_name}.npz",
                )
                write_fire_archive(
                    save_path,
                    data=loc_x,
                    labels=loc_y,
                    label_sel=label_sel,
                )
                staged_count += 1
            else:
                print(f"{loc_name}: no output saved")

            # Free this fire's windows before the next fire is built.
            del loc_x, loc_y
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            gc.collect()

            # Upload a full batch, then empty the staging folder.
            if staged_count >= self.batch_size:
                print(
                    f"Uploading batch of {staged_count} fires to "
                    "Hugging Face"
                )
                self.api.upload_folder(
                    folder_path=current_staging_dir,
                    repo_id=self.dest_repo,
                    repo_type="dataset",
                    path_in_repo=split_name,
                )
                shutil.rmtree(
                    current_staging_dir,
                    ignore_errors=True,
                )
                os.makedirs(current_staging_dir, exist_ok=True)
                staged_count = 0

        # Upload whatever is left after the last fire.
        if staged_count > 0:
            print(
                f"Uploading final batch of {staged_count} fires to "
                "Hugging Face"
            )
            self.api.upload_folder(
                folder_path=current_staging_dir,
                repo_id=self.dest_repo,
                repo_type="dataset",
                path_in_repo=split_name,
            )
            shutil.rmtree(
                current_staging_dir,
                ignore_errors=True,
            )


def process_data(
    token,
    src_repo=SRC_REPO,
    window_repo=None,
    batch_size=10,
    length=TS_LENGTH,
    interval=TS_INTERVAL,
    splits=SPLITS,
    roi_dir="hf_roi",
    work_dir=".",
    overwrite=False,
    debug=False,
    use_roi_label_sel=True,
    train_val_label_sel=TRAIN_VAL_LABEL_SEL,
):
    """Stream raw fires from `src_repo` into windows of `length` input days.

    A new window starts every `interval` days. Windows go to `window_repo`,
    by default window_repo_for(length, interval), so different settings never
    share a repo. `token` needs write access to `window_repo`.

    Targets follow the TS-SatFire paper: train/val fires use
    `train_val_label_sel` (default TRAIN_VAL_LABEL_SEL = 0, the union of
    accumulated active fire and NIFC perimeters) and test fires use their ROI
    label_sel (the paper's per-fire choice). `use_roi_label_sel=False` uses
    `train_val_label_sel` for test fires too. Each archive records its
    label_sel. Fires already in `window_repo` are skipped unless `overwrite`
    is set.
    """
    if window_repo is None:
        window_repo = window_repo_for(length, interval)
    print(f"Windows: {length} input days every {interval} days -> {window_repo}")

    processor = DatasetStreamer(
        src_repo=src_repo,
        dest_repo=window_repo,
        token=token,
        batch_size=batch_size,
        work_dir=work_dir,
    )
    # Record (or check) the build settings in window_config.json before any fire
    # is built, so a repo with other settings fails at once.
    ensure_window_config(
        processor.api,
        window_repo,
        window_settings(length, interval, train_val_label_sel, use_roi_label_sel),
        overwrite=overwrite,
        token=token,
    )

    # using id metadata from above, split into appropriate folders
    train_ids, val_ids, test_ids = obtain_ids(src_repo, token, roi_dir)
    ids_by_split = {"train": train_ids, "val": val_ids, "test": test_ids}

    # Label rule per fire: the ROI CSV's label_sel where it has one (the paper's
    # per-fire choice for test fires), else train_val_label_sel.
    roi_label_sel = (
        fire_label_sel(download_roi_csvs(src_repo, token, roi_dir))
        if use_roi_label_sel
        else {}
    )
    label_sel_by_id = {
        str(fire_id): int(roi_label_sel.get(str(fire_id), train_val_label_sel))
        for split_name in splits
        for fire_id in ids_by_split[split_name]
    }

    # Build and upload each split in turn.
    for split_name in splits:
        processor.run_split(
            ids_by_split[split_name],
            split_name,
            length=length,
            interval=interval,
            label_sel_by_id=label_sel_by_id,
            overwrite=overwrite,
            debug=debug,
        )
    return processor
