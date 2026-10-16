"""Discover fires and download their source rasters from Hugging Face.

ROI CSVs define fire dates, labels, and train/validation/test membership. Raw
GeoTIFFs are downloaded into a per-fire local directory, so the raster reader
can reopen files patch by patch without holding a whole fire in memory.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds

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


def validate_fire_inventory(day_remotes, remote_to_local):
    """Require one same-named FirePred raster for every day frame.

    Night rasters are optional, matching the legacy pipeline, but when present
    they must have an unambiguous same-name companion. This prevents silently
    pairing a frame with another date's auxiliary data.
    """
    if not day_remotes:
        raise ValueError("The fire has no VIIRS_Day frames")
    missing = []
    for day_remote in day_remotes:
        firepred = match_companion_file(day_remote, "FirePred", remote_to_local)
        if firepred is None:
            missing.append((day_remote, "FirePred"))
        night = match_companion_file(day_remote, "VIIRS_Night", remote_to_local)
        if night is None:
            continue
        if night.name != Path(day_remote).name:
            raise ValueError(f"Night frame does not match day-frame filename: {day_remote} -> {night}")
    if missing:
        details = ", ".join(f"{day} ({mode})" for day, mode in missing[:5])
        raise FileNotFoundError(f"Missing same-date FirePred companions: {details}")


def validate_raster_coverage(day_remotes, remote_to_local, grid):
    """Check that each selected date/mode raster has usable metadata and overlaps the grid."""
    left, bottom, right, top = array_bounds(grid.height, grid.width, grid.transform)
    target = (left, bottom, right, top)
    for day_remote in day_remotes:
        candidates = [day_remote]
        for mode in ("VIIRS_Night", "FirePred"):
            companion = match_companion_file(day_remote, mode, remote_to_local)
            if companion is not None:
                candidates.append(next(remote for remote, local in remote_to_local.items() if local == companion))
        for remote in candidates:
            path = remote_to_local[remote]
            with rasterio.open(path) as src:
                if src.crs is None or src.width <= 0 or src.height <= 0:
                    raise ValueError(f"Raster has invalid dimensions or no CRS: {remote}")
                bounds = transform_bounds(src.crs, grid.crs, *src.bounds, densify_pts=21)
                if bounds[2] <= target[0] or bounds[0] >= target[2] or bounds[3] <= target[1] or bounds[1] >= target[3]:
                    raise ValueError(f"Raster does not overlap the shared target grid: {remote}")
