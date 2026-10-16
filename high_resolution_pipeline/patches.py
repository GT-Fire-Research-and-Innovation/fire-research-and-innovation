"""Generate fixed-size patches with geographic transforms and validity masks."""

from dataclasses import dataclass

import numpy as np
from rasterio.transform import Affine

from high_resolution_pipeline.grid import GridSpec, patch_transform


@dataclass(frozen=True)
class PatchSpec:
    row: int
    col: int
    size: int
    crs: object
    transform: Affine
    valid_height: int
    valid_width: int

    @property
    def patch_id(self) -> str:
        return f"r{self.row:06d}_c{self.col:06d}"

    def valid_mask(self) -> np.ndarray:
        mask = np.zeros((self.size, self.size), dtype=bool)
        mask[:self.valid_height, :self.valid_width] = True
        return mask


def _starts(length: int, patch_size: int, stride: int) -> list[int]:
    """Make starts cover the full axis, overlapping the last patch if needed."""
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    final_start = length - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def iter_patch_specs(grid: GridSpec, patch_size: int, stride: int):
    """Yield row-major fixed-size patches that cover the target grid once or more."""
    if patch_size <= 0 or stride <= 0 or stride > patch_size:
        raise ValueError("patch_size must be positive and stride must be in [1, patch_size]")
    for row in _starts(grid.height, patch_size, stride):
        for col in _starts(grid.width, patch_size, stride):
            yield PatchSpec(
                row=row,
                col=col,
                size=patch_size,
                crs=grid.crs,
                transform=patch_transform(grid, row, col),
                valid_height=min(patch_size, grid.height - row),
                valid_width=min(patch_size, grid.width - col),
            )


def apply_valid_mask(array: np.ndarray, spec: PatchSpec, fill_value=0) -> np.ndarray:
    """Zero padded cells beyond the fire crop while retaining fixed shapes."""
    result = np.array(array, copy=True)
    if result.shape[-2:] != (spec.size, spec.size):
        raise ValueError(f"Array shape {result.shape} does not end in patch shape {(spec.size, spec.size)}")
    result[..., spec.valid_height:, :] = fill_value
    result[..., :, spec.valid_width:] = fill_value
    return result
