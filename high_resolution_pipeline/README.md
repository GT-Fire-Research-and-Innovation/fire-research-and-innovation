# High resolution fire data pipeline

This folder contains a second data-preparation pipeline. It creates a
Hugging Face dataset whose VIIRS, auxiliary, label, and TESSERA arrays share a
configurable finer spatial grid and are saved as manageable patches. The
existing 256 × 256 pipeline remains separate.

## Workflow

1. Read fire IDs, dates, and split assignments from the ROI metadata.
   Train/validation use the paper's default target (accumulated active fire
   plus burned area); test fires use their ROI `label_sel` when provided.
2. Fetch the raw source rasters needed for one fire.
3. Use the configured pixel crop from the first VIIRS image to define one
   geographic area, then read every raster's CRS and affine transform and
   reproject it to that same target grid. The default crop (offset 128, size
   256) preserves the area used by the existing pipeline; both settings can
   be changed with `--source-crop-offset` and `--source-crop-size`.
4. Resample each source layer onto that grid using a method appropriate to its
   data type. Resampling coarse values to a finer grid aligns the arrays; it
   does not create new source detail.
5. Stream the aligned grid in chunks, cut every channel and label into the same
   configurable patches, and attach geographic and fire metadata.
6. Package patches into versioned shards and upload completed shards to a
   separate Hugging Face dataset repository.

## Files

- `config.py` — user-selected grid spacing, patch dimensions, overlap, data
  split, and repository/output settings.
- `sources.py` — discovers fires and fetches their source rasters and ROI
  metadata.
- `grid.py` — reads geospatial metadata and defines the common target grid.
- `resample.py` — streams and resamples the different input layers and labels.
- `patches.py` — extracts matching patches from all aligned arrays.
- `package.py` — validates and writes versioned dataset shards and metadata.
- `huggingface.py` — uploads/downloads shards and records the dataset layout.
- `loader.py` — reads one temporal window at a time for PyTorch and keeps optional static TESSERA as one tensor per sample.
- `pipeline.py` — coordinates the per-fire workflow and resumable batches.

The modules implement local patch generation, storage, and Hugging Face
upload/download. No data is downloaded or uploaded merely by importing them.

## Important distinction

Grid spacing (meters per pixel), source crop, and patch dimensions (pixels per patch) are
independent settings. For example, 64 × 64 pixels at 10 m spacing covers 640 m
on each side. The source GeoTIFF transforms, rather than a nominal sensor

## Commands

Run from the repository root. Install the project's environment first and set
`HF_TOKEN` for private or write access.

```powershell
python -m high_resolution_pipeline.pipeline preflight --resolution-m 75 --patch-size 256 --patch-stride 256
python -m high_resolution_pipeline.pipeline process --resolution-m 75 --patch-size 256 --patch-stride 256
python -m high_resolution_pipeline.pipeline upload --repo-id YOUR_ACCOUNT/ts-satfire-75m-patches
python -m high_resolution_pipeline.pipeline run --repo-id YOUR_ACCOUNT/ts-satfire-75m-patches --resolution-m 75 --patch-size 256 --patch-stride 256
python -m high_resolution_pipeline.pipeline download --repo-id YOUR_ACCOUNT/ts-satfire-75m-patches --output-dir ./downloaded_patches
```

The default target spacing is 75 m, dividing a nominal 375 m VIIRS pixel into
5 x 5 target cells. This is not an exact integer grouping of nominal 10 m
TESSERA cells, so TESSERA values are spatially aggregated onto the 75 m grid
and VIIRS rasters are reprojected onto that same grid. A 75 m target cell does
not contain a fixed whole number of TESSERA embeddings.
Defaults follow the paper's prediction setup: 6 input days and a 1-day
window stride. Override these with `--ts-length` and `--window-stride-days`.
Archive members stream directly into the ZIP file to avoid a second full-window
byte buffer. `--archive-compresslevel` controls the speed/size tradeoff (default
1, which favors faster writing and can produce larger files).
Use `--train-val-label-sel 1` if you intentionally want active-fire-only
targets for train/validation; the default `0` combines active-fire and
burned-area masks. The selected rule is saved in each fire manifest, and the
destination repo rejects uploads with mismatched settings.

`preflight` downloads one source GeoTIFF to inspect actual GeoTIFF spacing and
estimate output size. `process` writes locally. `upload` sends completed local
fires. `run` processes and uploads one fire at a time. `download` retrieves a
repository snapshot. Outputs are separated by resolution, patch, and temporal
window settings under `data_high_resolution/`.

The output repository layout is `split/p<fire_id>/manifest.json` plus one
`patches/<patch_id>.npz` per spatial patch. Each NPZ stores TESSERA once and
then adds its temporal examples as individual `data_#####` and `label_#####`
members, which keeps memory bounded and reduces the number of Hugging Face
files. Each temporal sample contains the 27 input channels and next-day label;
the patch contains one shared validity mask. Arrays are FP16 for
inputs/embeddings and uint8 for labels/masks. The `loader.py` module reads this
format; the legacy 256 × 256 loader does not.

For local PyTorch loading, pass a split directory to
`HighResolutionPatchDataset`, such as
`data_high_resolution/<settings>/train`. The returned `data` has 27 raw
channels, `labels` has background/fire class channels, and `valid_mask` marks
cells that should contribute to loss. To include TESSERA, compute
`compute_tessera_stats` from training patch archives only and pass the result
as `tessera_stats`; each sample then includes one static `(129,H,W)` tensor.
The training loop can repeat that tensor over time on the GPU.

Run `preflight` first. At 10 m, a 256 × 256 patch covers 2.56 km per side and
the full fire crop may produce many patches. The size estimate is uncompressed
and actual upload volume depends on compression and the selected resolution.
