"""Offline stand-in for the parts of `huggingface_hub` the dataset pipeline calls.

Installed as sys.modules["huggingface_hub"] by snapshot.py's hub tier. Each
repo is a folder under ROOT; every API call is logged in CALLS (paths made
relative) so two trees can be compared call by call.
"""

import os
import shutil
from pathlib import Path

ROOT = None   # set by snapshot.py: folder holding one subfolder per repo id
CALLS = []


def _repo(repo_id):
    return Path(ROOT) / repo_id.replace("/", "__")


def _record(name, *args, **kwargs):
    CALLS.append((name, args, tuple(sorted(kwargs.items()))))


def login(token=None, **kwargs):
    _record("login")


def hf_hub_download(repo_id, filename, repo_type=None, cache_dir=None, local_dir=None, token=None, **kwargs):
    """Copy a repo file like the real cache/local_dir layouts do and return its path."""
    _record("hf_hub_download", repo_id, filename, local_dir=local_dir is not None, cache_dir=cache_dir is not None)
    source = _repo(repo_id) / filename
    if not source.is_file():
        raise FileNotFoundError(f"{repo_id}:{filename}")
    if local_dir is not None:
        target = Path(local_dir) / filename
    else:
        base = Path(cache_dir or os.path.join(ROOT, ".cache"))
        target = base / f"datasets--{repo_id.replace('/', '--')}" / "snapshots" / "fake" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return str(target)


class HfApi:
    def __init__(self, token=None, **kwargs):
        pass

    def list_repo_files(self, repo_id, repo_type=None, **kwargs):
        _record("list_repo_files", repo_id)
        root = _repo(repo_id)
        if not root.is_dir():
            return []
        return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())

    def create_repo(self, repo_id, repo_type=None, exist_ok=False, private=False, **kwargs):
        _record("create_repo", repo_id, private=private)
        _repo(repo_id).mkdir(parents=True, exist_ok=True)

    def upload_folder(self, folder_path, repo_id, repo_type=None, path_in_repo=None, **kwargs):
        folder = Path(folder_path)
        names = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
        _record("upload_folder", repo_id, path_in_repo, names)
        for name in names:
            target = _repo(repo_id) / (path_in_repo or "") / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(folder / name, target)

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type=None, token=None, **kwargs):
        _record("upload_file", repo_id, path_in_repo)
        target = _repo(repo_id) / path_in_repo
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(path_or_fileobj, (bytes, bytearray)):
            target.write_bytes(path_or_fileobj)
        else:
            shutil.copyfile(path_or_fileobj, target)

    def hf_hub_download(self, repo_id, filename, repo_type=None, local_dir=None, token=None, **kwargs):
        return hf_hub_download(repo_id, filename, repo_type=repo_type, local_dir=local_dir, token=token)
