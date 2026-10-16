# High resolution fire data pipeline

This folder contains a second data-preparation pipeline. It creates a
Hugging Face dataset whose VIIRS, auxiliary, label, and TESSERA arrays share a
configurable finer spatial grid and are saved as manageable patches. The
existing 256 × 256 pipeline remains separate.

## Workflow

1. Read fire IDs, dates, and split assignments from the ROI metadata.
2. Fetch the raw source rasters needed for one fire.
3. Read every raster's CRS and affine transform and define one common target
   grid for the selected area.
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
- `pipeline.py` — coordinates the per-fire workflow and resumable batches.

The modules implement local patch generation, storage, and Hugging Face
upload/download. No data is downloaded or uploaded merely by importing them.

## Important distinction

Grid spacing (meters per pixel) and patch dimensions (pixels per patch) are
independent settings. For example, 64 × 64 pixels at 10 m spacing covers 640 m
on each side. The source GeoTIFF transforms, rather than a nominal sensor
resolution, must determine exact alignment.

## Commands

Run from the repository root. Install the project's environment first and set
`HF_TOKEN` for private or write access.

```powershell
python -m high_resolution_pipeline.pipeline preflight --resolution-m 10 --patch-size 256 --patch-stride 256
python -m high_resolution_pipeline.pipeline process --resolution-m 10 --patch-size 256 --patch-stride 256
python -m high_resolution_pipeline.pipeline upload --repo-id YOUR_ACCOUNT/ts-satfire-10m-patches
python -m high_resolution_pipeline.pipeline run --repo-id YOUR_ACCOUNT/ts-satfire-10m-patches --resolution-m 10 --patch-size 256 --patch-stride 256
python -m high_resolution_pipeline.pipeline download --repo-id YOUR_ACCOUNT/ts-satfire-10m-patches --output-dir ./downloaded_patches
```

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
inputs/embeddings and uint8 for labels/masks. A separate training loader must
read this new format; the legacy 256 × 256 loader does not.

Run `preflight` first. At 10 m, a 256 × 256 patch covers 2.56 km per side and
the full fire crop may produce many patches. The size estimate is uncompressed
and actual upload volume depends on compression and the selected resolution.
