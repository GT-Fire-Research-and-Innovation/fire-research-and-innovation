"""Download windowed archives (and TESSERA files) from Hugging Face into a local data root.

Level 2 (hub). Used by cli/dataset.py (`download`).
"""

import os
import shutil

from huggingface_hub import HfApi, hf_hub_download

from firepred.config.dataset import SPLITS, SRC_REPO, TESSERA_REPO, WINDOW_REPO
from firepred.hub.repos import read_window_config
from firepred.hub.splits import obtain_ids, tessera_excluded_npz


class DatasetLoader:
    def __init__(self, window_repo_id, token, train_ids, val_ids, test_ids, base_output_dir="./data", exclude_files=()):
        self.window_repo_id = window_repo_id
        self.exclude_files = set(exclude_files)  # repo paths, e.g. 'train/p20778153.npz'
        self.token = token
        self.train_ids = set(train_ids)  # Convert to set for faster lookup
        self.val_ids = set(val_ids)      # Convert to set for faster lookup
        self.test_ids = set(test_ids)    # Convert to set for faster lookup
        self.base_output_dir = base_output_dir
        self.api = HfApi(token=token)

        # Local layout: <base_output_dir>/train, val, test.
        self.train_dir = os.path.join(self.base_output_dir, "train")
        self.val_dir = os.path.join(self.base_output_dir, "val")
        self.test_dir = os.path.join(self.base_output_dir, "test")

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)

    def load_and_organize_data(self):
        print(f"Loading data from: {self.window_repo_id}")
        # Walk the window repo and put each fire's archive in the split folder its
        # fire ID belongs to (decided by obtain_ids, not by the repo folder).
        all_processed_files = self.api.list_repo_files(self.window_repo_id, repo_type="dataset")

        for file_path in all_processed_files:
            # Only split folders hold window archives; tessera/p<id>.npz shares
            # their file names and must not be routed into a split folder.
            if file_path.split('/')[0] not in SPLITS:
                continue
            if file_path in self.exclude_files:
                print(f"Skipping {file_path}: excluded")
                if os.path.exists(os.path.join(self.base_output_dir, file_path)):
                    print(f"WARNING: {file_path} is excluded but already exists in {self.base_output_dir}")
                continue
            if file_path.endswith('.npz'):
                # Extract loc_name from filename, e.g., 'train/p20562846.npz' -> '20562846'
                filename_without_extension = os.path.basename(file_path).replace('.npz', '').replace('p', '')
                if filename_without_extension in self.train_ids:
                    target_dir = self.train_dir
                elif filename_without_extension in self.val_ids:
                    target_dir = self.val_dir
                elif filename_without_extension in self.test_ids:
                    target_dir = self.test_dir
                else:
                    print(f"Skipping {file_path}: {filename_without_extension} ID not found in train, val, or test sets.")
                    continue
                # Files already on disk are skipped, so a rerun only fetches what is missing.
                local_file_path = os.path.join(target_dir, os.path.basename(file_path))
                if not os.path.exists(local_file_path):
                    print(f"Downloading {file_path} to {target_dir}")
                    downloaded_path = hf_hub_download(
                        repo_id=self.window_repo_id,
                        filename=file_path,
                        repo_type="dataset",
                        local_dir=self.base_output_dir,
                        token=self.token,
                    )
                    shutil.move(downloaded_path, local_file_path)
                else:
                    print(f"File already exists: {local_file_path}")

        print("Data loading and organization complete.")

    def load_tessera(self, tessera_repo=TESSERA_REPO):
        """Download tessera/* (per-fire embeddings) from `tessera_repo` into <base_output_dir>/tessera."""
        tessera_files = [
            file_path
            for file_path in self.api.list_repo_files(tessera_repo, repo_type="dataset")
            if file_path.startswith("tessera/")
        ]
        print(f"Downloading {len(tessera_files)} TESSERA files")
        for file_path in tessera_files:
            if os.path.exists(os.path.join(self.base_output_dir, file_path)):
                continue
            hf_hub_download(
                repo_id=tessera_repo,
                filename=file_path,
                repo_type="dataset",
                local_dir=self.base_output_dir,
                token=self.token,
            )

        # Every window archive needs a TESSERA file of the same name; list those without one.
        missing = [
            os.path.join(split, file_name)
            for split in SPLITS
            for file_name in sorted(os.listdir(os.path.join(self.base_output_dir, split)))
            if file_name.endswith(".npz")
            and not os.path.exists(os.path.join(self.base_output_dir, "tessera", file_name))
        ]
        if missing:
            print(f"WARNING: {len(missing)} window archives have no TESSERA file: {missing}")


def download_processed_data(
    token,
    window_repo=WINDOW_REPO,
    src_repo=SRC_REPO,
    base_output_dir="./data",
    roi_dir="hf_roi",
    include_tessera=False,
    tessera_repo=TESSERA_REPO,
):
    """Download every windowed NPZ in `window_repo` into <base_output_dir>/{train,val,test}.

    With `include_tessera` (TESSERA mode), fires with no prior-year TESSERA
    (the 2017 fires) are left out and per-fire embeddings are downloaded from
    `tessera_repo` into <base_output_dir>/tessera. The dropped archive names are kept on the
    returned loader as `excluded_npz`. Use a different `base_output_dir` for
    each mode so a full download never mixes with a TESSERA one.
    """
    # Print the repo's build settings, so the log shows which windows were used.
    window_config = read_window_config(HfApi(token=token), window_repo, token)
    print(f"Window repo {window_repo}: {window_config}")
    # Fire IDs per split; TESSERA mode also drops fires with no prior-year embedding.
    train_ids, val_ids, test_ids = obtain_ids(src_repo, token, roi_dir)
    excluded_npz = []
    if include_tessera:
        excluded_npz = tessera_excluded_npz(token, src_repo, roi_dir)
        print(f"TESSERA mode: dropping {len(excluded_npz)} fires without prior-year TESSERA")
    data_loader = DatasetLoader(
        window_repo_id=window_repo,
        token=token,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        base_output_dir=base_output_dir,
        exclude_files=excluded_npz,
    )
    data_loader.excluded_npz = excluded_npz
    data_loader.load_and_organize_data()
    if include_tessera:
        data_loader.load_tessera(tessera_repo)
    return data_loader
