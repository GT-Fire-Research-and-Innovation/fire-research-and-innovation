"""One-file bundles: zip a data root, upload it, pull it back and unpack it.

Level 2 (hub). Used by cli/dataset.py (`upload-zip`, `pull-zip`, `pull-tessera`),
hub/publish_tessera.py and the notebooks.
"""

import os
import zipfile
from pathlib import Path

from huggingface_hub import HfApi

from firepred.config.dataset import TESSERA_EXCLUDED_LIST, TESSERA_REPO, WINDOW_REPO
from firepred.hub.staging import check_members_inside


# zip npz files into a zip, creating output_path file
def zip_data_from(folders, output_path):
    zip_path = Path(output_path)

    # ZIP_STORED: no second compression (the NPZ archives are already compressed).
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zipf:
        for folder in folders:
            folder_path = Path(folder)

            if not folder_path.exists():
                print(f"Folder {folder_path} does not exist. Skipping.")
                continue

            for file_path in folder_path.rglob('**/*'):
                if file_path.is_file():
                    # Names keep their folder (train/p123.npz), so unzipping recreates the layout.
                    archive_name = file_path.relative_to(folder_path.parent)
                    zipf.write(file_path, archive_name)
                    print(f"Added {file_path} to zip file as {archive_name}")
                else:
                    print(f"{file_path} not file")


# send data.zip to repo
def zip_data_to(token, zip_path="./data.zip", repo_id=WINDOW_REPO, zip_name="data.zip"):
    return HfApi().upload_file(
        repo_id=repo_id,
        path_in_repo=zip_name,
        path_or_fileobj=str(zip_path),
        repo_type="dataset",
        token=token,
    )


def upload_data_zip(
    token,
    data_root="./data",
    zip_path="./data.zip",
    repo_id=WINDOW_REPO,
    folders=("test", "train", "val", "esri", "tessera"),
    zip_name="data.zip",
):
    """Zip <data_root>/<folder> for each folder and upload it to `repo_id` as `zip_name`.

    TESSERA mode uses repo_id=TESSERA_REPO, zip_name="data_tessera.zip", so
    the full data.zip in WINDOW_REPO is never replaced by the reduced set.
    """
    zip_data_from([os.path.join(data_root, folder) for folder in folders], zip_path)
    return zip_data_to(token, zip_path, repo_id, zip_name)


# pull from processed data the zip file.
def pull_zip_data_to(extraction_path, token=None, repo_id=WINDOW_REPO, download_dir="./data", zip_name="data.zip"):
    zip_file_path = HfApi().hf_hub_download(
        repo_id=repo_id,
        filename=zip_name,
        repo_type="dataset",
        local_dir=download_dir,
        token=token,
    )

    if os.path.exists(zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Refuse member paths that would land outside the target folder, then unpack.
            check_members_inside(zip_ref, extraction_path)
            zip_ref.extractall(extraction_path)
        print(f'Successfully extracted {zip_file_path} to {extraction_path}')
    else:
        print(f'Error: {zip_file_path} not found. Please ensure the data is downloaded first.')


def pull_tessera_data(
    data_root,
    token=None,
    tessera_repo=TESSERA_REPO,
    zip_name="tessera.zip",
    drop_excluded=True,
):
    """Unpack tessera.zip into <data_root>/tessera and drop excluded windows.

    Run after pulling data.zip into the same `data_root`. With `drop_excluded`,
    the window archives listed in tessera/excluded_npz.txt (fires without
    prior-year TESSERA) are deleted, so both the TESSERA run and the matched
    no-TESSERA run see the same fires.
    """
    pull_zip_data_to(
        data_root,
        token=token,
        repo_id=tessera_repo,
        download_dir=data_root,
        zip_name=zip_name,
    )
    if not drop_excluded:
        return []

    # Delete the listed window archives (fires with no prior-year TESSERA).
    with open(os.path.join(data_root, "tessera", TESSERA_EXCLUDED_LIST)) as handle:
        excluded = [line.strip() for line in handle if line.strip()]
    removed = []
    for name in excluded:
        path = os.path.join(data_root, name)
        if os.path.exists(path):
            os.remove(path)
            removed.append(name)
    print(
        f"TESSERA mode: removed {len(removed)} window archives of fires "
        "without prior-year TESSERA"
    )
    return removed
