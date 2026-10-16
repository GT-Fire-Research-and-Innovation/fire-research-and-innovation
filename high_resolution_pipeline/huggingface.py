"""Upload or download a versioned high-resolution patch dataset on Hugging Face."""

import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


REMOTE_CONFIG_FILE = "high_resolution_config.json"


class HuggingFaceDataset:
    """Small wrapper for authenticated dataset-repository operations."""

    def __init__(self, repo_id: str, token: str | None = None, private: bool = True, ensure_repo: bool = True):
        if not repo_id:
            raise ValueError("A dedicated destination dataset repository is required")
        self.repo_id = repo_id
        self.token = token
        self.api = HfApi(token=token)
        if ensure_repo:
            self.api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=True,
            )

    def list_files(self) -> set[str]:
        """Return all file paths currently stored in the destination dataset."""
        return set(self.api.list_repo_files(self.repo_id, repo_type="dataset"))

    def ensure_configuration(self, configuration: dict) -> None:
        """Record the dataset settings and refuse to mix incompatible uploads."""
        files = self.list_files()
        if REMOTE_CONFIG_FILE in files:
            config_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=REMOTE_CONFIG_FILE,
                repo_type="dataset",
                token=self.token,
            )
            existing = json.loads(Path(config_path).read_text(encoding="utf-8"))
            if existing != configuration:
                raise ValueError(
                    f"Destination repo {self.repo_id} already contains a different "
                    "high-resolution dataset configuration. Use a separate repo."
                )
            return

        existing_data = files.difference({".gitattributes"})
        if existing_data:
            raise RuntimeError(
                f"Destination repo {self.repo_id} already has files but no "
                f"{REMOTE_CONFIG_FILE}; refusing to risk mixing dataset settings."
            )
        self.api.upload_file(
            path_or_fileobj=json.dumps(configuration, indent=2, sort_keys=True).encode("utf-8"),
            path_in_repo=REMOTE_CONFIG_FILE,
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message="Record high-resolution dataset settings",
        )

    def upload_fire(self, local_fire_dir: str | Path, split: str, fire_id: str) -> str:
        """Upload one complete fire directory under `<split>/p<id>/`."""
        local_fire_dir = Path(local_fire_dir)
        if not (local_fire_dir / "manifest.json").is_file():
            raise FileNotFoundError(f"Fire manifest missing from {local_fire_dir}")
        self.api.upload_folder(
            folder_path=str(local_fire_dir),
            repo_id=self.repo_id,
            repo_type="dataset",
            path_in_repo=f"{split}/p{fire_id}",
            commit_message=f"Add high-resolution patches for fire {fire_id}",
        )
        return f"{split}/p{fire_id}"

    def upload_directory(self, local_dir: str | Path, path_in_repo: str = ".") -> str:
        """Upload a directory, useful for a README or dataset-level manifest."""
        self.api.upload_folder(
            folder_path=str(local_dir),
            repo_id=self.repo_id,
            repo_type="dataset",
            path_in_repo=path_in_repo,
        )
        return path_in_repo

    def upload_dataset_card(self, path: str | Path) -> str:
        """Upload a local README as the dataset card at repository root."""
        self.api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo="README.md",
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message="Document high-resolution patch dataset format",
        )
        return "README.md"

    def download(self, local_dir: str | Path, revision: str | None = None) -> Path:
        """Download a repository snapshot to a local training-data directory."""
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        result = snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=str(local_dir),
            token=self.token,
        )
        return Path(result)
