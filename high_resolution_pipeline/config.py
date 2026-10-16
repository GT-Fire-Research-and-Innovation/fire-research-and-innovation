"""Validated settings shared by high-resolution generation modules.

Grid spacing is measured in ground meters per pixel. Patch dimensions and
stride are measured in target-grid pixels. They are independent settings.
The defaults are an initial 10 m experiment; changing them creates a different
dataset format/version and should use a separate output/repository location.
"""

from dataclasses import dataclass
import os
from pathlib import Path

from satimg_dataset_processor.hf_dataset import SRC_REPO


@dataclass(frozen=True)
class PipelineConfig:
    source_repo: str = SRC_REPO
    destination_repo: str | None = None
    token: str | None = None
    output_dir: Path = Path("data_high_resolution")
    work_dir: Path = Path("high_resolution_work")
    roi_dir: Path = Path("high_resolution_work/roi")
    splits: tuple[str, ...] = ("train", "val", "test")
    train_val_label_sel: int = 0
    resolution_m: float = 10.0
    source_crop_offset: int = 128
    source_crop_size: int = 256
    patch_size: int = 256
    patch_stride: int = 256
    time_series_length: int = 6
    window_stride_days: int = 1
    archive_compresslevel: int = 1
    tessera_strip_rows: int = 128
    overwrite: bool = False
    remove_source_files_after_fire: bool = True
    remove_local_fire_after_upload: bool = False

    def __post_init__(self):
        if self.token is None:
            object.__setattr__(self, "token", os.environ.get("HF_TOKEN"))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "work_dir", Path(self.work_dir))
        object.__setattr__(self, "roi_dir", Path(self.roi_dir))
        if not self.source_repo:
            raise ValueError("source_repo must be set")
        if not self.splits or any(s not in {"train", "val", "test"} for s in self.splits):
            raise ValueError("splits must contain train, val, and/or test")
        if self.train_val_label_sel not in (0, 1):
            raise ValueError("train_val_label_sel must be 0 (AF + BA) or 1 (AF only)")
        if self.resolution_m < 10:
            raise ValueError(
                "resolution_m must be at least 10 m: the supported TESSERA "
                "source grid is nominally 10 m, and finer cells would invent samples"
            )
        for name in ("source_crop_size", "patch_size", "patch_stride", "time_series_length", "window_stride_days", "tessera_strip_rows"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.source_crop_offset < 0:
            raise ValueError("source_crop_offset cannot be negative")
        if not 0 <= self.archive_compresslevel <= 9:
            raise ValueError("archive_compresslevel must be between 0 (fastest) and 9 (smallest)")
        if self.patch_stride > self.patch_size:
            raise ValueError("patch_stride cannot exceed patch_size because that would leave uncovered gaps")

    @property
    def has_token(self) -> bool:
        return bool(self.token)

    def to_dict(self) -> dict:
        return {
            "source_repo": self.source_repo,
            "destination_repo": self.destination_repo,
            "resolution_m": self.resolution_m,
            "source_crop_offset_pixels": self.source_crop_offset,
            "source_crop_size_pixels": self.source_crop_size,
            "patch_size_pixels": self.patch_size,
            "patch_stride_pixels": self.patch_stride,
            "time_series_length_days": self.time_series_length,
            "window_stride_days": self.window_stride_days,
            "archive_compresslevel": self.archive_compresslevel,
            "tessera_strip_rows": self.tessera_strip_rows,
            "splits": list(self.splits),
            "train_val_label_sel": self.train_val_label_sel,
        }
