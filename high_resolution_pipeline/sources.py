"""Discover fires and download their source rasters from Hugging Face.

ROI CSVs define fire dates, labels, and train/validation/test membership. Raw
GeoTIFFs are downloaded into a per-fire local directory, so the raster reader
can reopen files patch by patch without holding a whole fire in memory.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from satimg_dataset_processor.hf_dataset import VAL_IDS


@dataclass(frozen=True)
class FireRecord:
    fire_id: str
    split: str
    start_date: str
    label_sel: int
    remote_files: dict[str, tuple[str, ...]]

    @property
    def fire_year(self) -> int:
        return int(pd.to_datetime(self.start_date).year)


def _clean_id(value) -> str:
    """Normalize CSV and repository IDs without changing non-numeric IDs."""
    value = str(value).strip()
    try:
        number = float(value)
    except ValueError:
        return value
    return str(int(number)) if number.is_integer() else value


def _mode_files(api: HfApi, source_repo: str) -> dict[str, dict[str, list[str]]]:
    """Return `{fire_id: {folder_mode: [remote GeoTIFF paths]}}`."""
    locations: dict[str, dict[str, list[str]]] = {}
    for remote in api.list_repo_files(source_repo, repo_type="dataset"):
        if not remote.lower().endswith((".tif", ".tiff")):
            continue
        parts = remote.split("/")
        if len(parts) < 3:
            continue
        fire_id, mode = _clean_id(parts[-3]), parts[-2]
        locations.setdefault(fire_id, {}).setdefault(mode, []).append(remote)
    for modes in locations.values():
        for paths in modes.values():
            paths.sort()
    return locations


def _download_roi_tables(api: HfApi, source_repo: str, roi_dir: Path, token: str | None) -> list[Path]:
    """Download ROI CSVs from the source dataset and return local paths."""
    roi_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for remote in api.list_repo_files(source_repo, repo_type="dataset"):
        name = Path(remote).name.lower()
        if not remote.lower().endswith(".csv") or not ("2017" in name or "2018" in name or "2019" in name or "2020" in name or "2021" in name):
            continue
        paths.append(Path(hf_hub_download(
            repo_id=source_repo,
            filename=remote,
            repo_type="dataset",
            local_dir=str(roi_dir),
            token=token,
        )))
    if not paths:
        raise FileNotFoundError(f"No year-specific ROI CSV files found in {source_repo}")
    return paths


def load_fire_records(
    source_repo: str,
    roi_dir: str | Path,
    token: str | None = None,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> list[FireRecord]:
    """Read ROI metadata and source file inventory for selected data splits."""
    api = HfApi(token=token)
    csv_paths = _download_roi_tables(api, source_repo, Path(roi_dir), token)
    locations = _mode_files(api, source_repo)

    by_id: dict[str, dict] = {}
    for path in csv_paths:
        filename = path.name
        if any(year in filename for year in ("2017", "2018", "2019", "2020")):
            year_split = None
        elif "2021" in filename:
            year_split = "test"
        else:
            continue
        frame = pd.read_csv(path, dtype={"Id": str})
        if not {"Id", "start_date"}.issubset(frame.columns):
            continue
        for row in frame.to_dict("records"):
            if pd.isna(row.get("Id")) or pd.isna(row.get("start_date")):
                continue
            fire_id = _clean_id(row["Id"])
            year = int(pd.to_datetime(row["start_date"]).year)
            split = year_split or ("val" if fire_id in {str(v) for v in VAL_IDS} else "train")
            by_id[fire_id] = {
                "split": split,
                "start_date": str(row["start_date"]),
                "label_sel": int(row.get("label_sel", 1)) if pd.notna(row.get("label_sel", 1)) else 1,
                "year": year,
            }

    records = []
    for fire_id, metadata in sorted(by_id.items()):
        if metadata["split"] not in splits:
            continue
        modes = locations.get(fire_id, {})
        if "VIIRS_Day" not in modes:
            continue
        records.append(FireRecord(
            fire_id=fire_id,
            split=metadata["split"],
            start_date=metadata["start_date"],
            label_sel=metadata["label_sel"],
            remote_files={mode: tuple(files) for mode, files in modes.items()},
        ))
    return records


def download_fire_files(
    record: FireRecord,
    source_repo: str,
    fire_dir: str | Path,
    token: str | None = None,
    selected_files: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    """Download one fire's GeoTIFFs and map remote names to local paths."""
    fire_dir = Path(fire_dir)
    fire_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    files = selected_files or tuple(remote for paths in record.remote_files.values() for remote in paths)
    for remote in files:
        local = hf_hub_download(
            repo_id=source_repo,
            filename=remote,
            repo_type="dataset",
            local_dir=str(fire_dir),
            token=token,
        )
        result[remote] = Path(local)
    return result


def match_companion_file(day_remote: str, mode: str, remote_to_local: dict[str, Path]) -> Path | None:
    """Find same-date VIIRS_Night or FirePred file corresponding to a day file."""
    candidate = day_remote.replace("/VIIRS_Day/", f"/{mode}/")
    if candidate in remote_to_local:
        return remote_to_local[candidate]
    basename = Path(day_remote).name
    matches = [path for remote, path in remote_to_local.items() if f"/{mode}/" in remote and Path(remote).name == basename]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous {mode} companion for {day_remote}")
    return matches[0] if matches else None
