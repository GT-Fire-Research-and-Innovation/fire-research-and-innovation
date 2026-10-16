"""Command-line workflow for high-resolution, patch-based dataset generation.

Run from the repository root with `python -m high_resolution_pipeline.pipeline
preflight ...`, `process ...`, `upload ...`, or `run ...`. `run` processes and
uploads one fire at a time. It does not launch automatically when imported.
"""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from high_resolution_pipeline.config import PipelineConfig
from high_resolution_pipeline.grid import build_target_grid
from high_resolution_pipeline.huggingface import HuggingFaceDataset
from high_resolution_pipeline.package import (
    PatchArchiveWriter,
    patch_archive_window_count,
    write_dataset_card,
    write_fire_manifest,
)
from high_resolution_pipeline.patches import apply_valid_mask, iter_patch_specs
from high_resolution_pipeline.resample import add_temporal_maxima, read_prediction_features, read_prediction_labels
from high_resolution_pipeline.sources import download_fire_files, load_fire_records, match_companion_file


def _dataset_tag(config: PipelineConfig) -> str:
    return (
        f"{config.resolution_m:g}m_p{config.patch_size}_s{config.patch_stride}_"
        f"t{config.time_series_length}_i{config.window_stride_days}"
    )


def fire_output_dir(config: PipelineConfig, split: str, fire_id: str) -> Path:
    """Return an output path namespaced by target resolution and patch geometry."""
    return config.output_dir / _dataset_tag(config) / split / f"p{fire_id}"


def _tessera_client():
    """Construct the streaming TESSERA source client lazily."""
    try:
        from geotessera import GeoTesseraZarr
    except ImportError as error:
        raise RuntimeError("Install geotessera to build the high-resolution dataset") from error
    return GeoTesseraZarr()


def _read_tessera_patch(client, fire_year: int, spec, strip_rows: int):
    """Stream native TESSERA pixels into one target patch grid."""
    from satimg_dataset_processor.tessera import grid_tessera

    tessera_year = fire_year - 1
    if tessera_year not in client.years:
        raise ValueError(f"Prior-year TESSERA layer {tessera_year} is unavailable")
    embedding, coverage, _, _ = grid_tessera(
        client,
        spec.crs,
        spec.transform,
        tessera_year,
        size=spec.size,
        strip_rows=strip_rows,
    )
    embedding = np.nan_to_num(embedding, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return embedding, coverage, tessera_year


def _fire_manifest_metadata(record, config, day_count):
    return {
        "fire_id": record.fire_id,
        "split": record.split,
        "start_date": record.start_date,
        "fire_year": record.fire_year,
        "label_sel": record.label_sel,
        "day_image_count": day_count,
        "configuration": config.to_dict(),
        "channel_order": [
            "VIIRS_Day_1", "VIIRS_Day_2", "VIIRS_Day_3", "VIIRS_Day_4", "VIIRS_Day_5", "VIIRS_Day_6",
            "VIIRS_Night_1", "VIIRS_Night_2",
            *[f"FirePred_{i}" for i in range(1, 20)],
        ],
        "label_rule": "new pixels in accumulated active-fire mask" if record.label_sel == 1 else "new pixels in accumulated active-fire OR burned-area mask",
    }


def _process_one_patch(record, config, remote_to_local, day_remotes, grid, spec, tessera_client, fire_dir):
    """Stream one patch through time; retain only the temporal input queue."""
    patch_path = fire_dir / "patches" / f"{spec.patch_id}.npz"
    if patch_path.exists() and not config.overwrite:
        return patch_archive_window_count(patch_path)
    valid_mask = spec.valid_mask()
    embedding, coverage, tessera_year = _read_tessera_patch(
        tessera_client, record.fire_year, spec, config.tessera_strip_rows
    )
    coverage = apply_valid_mask(coverage, spec, fill_value=0)
    embedding = apply_valid_mask(embedding, spec, fill_value=0)
    patch_metadata = {
        "fire_id": record.fire_id,
        "split": record.split,
        "fire_year": record.fire_year,
        "tessera_year": tessera_year,
        "patch_id": spec.patch_id,
        "row": spec.row,
        "col": spec.col,
        "resolution_m": config.resolution_m,
        "patch_size_pixels": spec.size,
        "crs": grid.crs.to_string(),
        "transform": list(tuple(spec.transform)[:6]),
        "coverage_threshold_semantics": "fraction of valid source TESSERA pixels",
    }
    queue = deque(maxlen=config.time_series_length)
    running_max = np.zeros((27, spec.size, spec.size), dtype=np.float32)
    cumulative_af = np.zeros((spec.size, spec.size), dtype=bool)
    cumulative_ba = np.zeros_like(cumulative_af)
    previous_final = np.zeros_like(cumulative_af)
    window_count = 0

    with PatchArchiveWriter(patch_path, embedding, coverage, valid_mask, patch_metadata) as archive:
        del embedding, coverage
        for day_index, day_remote in enumerate(day_remotes):
            day_path = remote_to_local[day_remote]
            night_path = match_companion_file(day_remote, "VIIRS_Night", remote_to_local)
            firepred_path = match_companion_file(day_remote, "FirePred", remote_to_local)
            if firepred_path is None:
                raise FileNotFoundError(f"Missing FirePred companion for {day_remote}")

            raw = read_prediction_features(
                day_path, night_path, firepred_path, grid.crs, spec.transform, spec.size
            )
            input_day, running_max = add_temporal_maxima(raw, running_max)
            af_mask, ba_mask = read_prediction_labels(day_path, grid.crs, spec.transform, spec.size)
            cumulative_af |= af_mask
            cumulative_ba |= ba_mask
            final_mask = cumulative_af if record.label_sel == 1 else (cumulative_af | cumulative_ba)

            # At index T, queue contains the T preceding input days and this day is
            # the next-day target. Thereafter emit every configured day stride.
            if day_index >= config.time_series_length and (day_index - config.time_series_length) % config.window_stride_days == 0:
                data = np.stack(tuple(queue), axis=1)  # (C,T,H,W)
                target = final_mask & ~previous_final
                data = apply_valid_mask(data, spec, fill_value=0)
                target = apply_valid_mask(target, spec, fill_value=False)
                window_metadata = {
                    **patch_metadata,
                    "window_index": window_count,
                    "input_day_start_index": day_index - config.time_series_length,
                    "target_day_index": day_index,
                    "input_length_days": config.time_series_length,
                    "window_stride_days": config.window_stride_days,
                    "target_rule": "new cumulative pixels on the target day",
                }
                archive.add_window(data, target, window_metadata)
                window_count += 1

            previous_final = final_mask.copy()
            queue.append(input_day)

    return window_count


def process_fire(record, config: PipelineConfig, tessera_client=None, upload_client=None) -> Path:
    """Build one fire's aligned TESSERA/input/label patches, optionally upload."""
    fire_dir = fire_output_dir(config, record.split, record.fire_id)
    manifest_path = fire_dir / "manifest.json"
    if manifest_path.exists() and not config.overwrite:
        print(f"{record.fire_id}: already has a local manifest; skipping")
        if upload_client is not None:
            upload_client.upload_fire(fire_dir, record.split, record.fire_id)
        return fire_dir

    fire_dir.mkdir(parents=True, exist_ok=True)
    local_source_dir = config.work_dir / "downloads" / f"p{record.fire_id}"
    remote_to_local = download_fire_files(record, config.source_repo, local_source_dir, config.token)
    day_remotes = list(record.remote_files.get("VIIRS_Day", ()))
    if not day_remotes:
        raise ValueError(f"{record.fire_id} has no VIIRS_Day rasters")
    reference_path = remote_to_local[day_remotes[0]]
    grid = build_target_grid(reference_path, resolution_m=config.resolution_m)
    tessera_client = tessera_client or _tessera_client()
    if record.fire_year - 1 not in tessera_client.years:
        raise ValueError(f"No TESSERA for fire {record.fire_id}: prior year {record.fire_year - 1} unavailable")

    patch_specs = list(iter_patch_specs(grid, config.patch_size, config.patch_stride))
    estimated_patches = len(patch_specs)
    print(
        f"{record.fire_id}: source pixel vectors={grid.source_pixel_size[0]:.3f}m x "
        f"{grid.source_pixel_size[1]:.3f}m; target={grid.width}x{grid.height} at "
        f"{config.resolution_m:g}m; patches={estimated_patches}; days={len(day_remotes)}"
    )

    total_windows = 0
    for patch_index, spec in enumerate(patch_specs, start=1):
        print(f"{record.fire_id}: patch {patch_index}/{estimated_patches} {spec.patch_id}")
        total_windows += _process_one_patch(
            record, config, remote_to_local, day_remotes, grid, spec,
            tessera_client, fire_dir,
        )

    write_fire_manifest(
        manifest_path,
        _fire_manifest_metadata(record, config, len(day_remotes)),
        grid.to_dict(),
        patch_count=estimated_patches,
        window_count=total_windows,
    )
    if config.remove_source_files_after_fire:
        shutil.rmtree(local_source_dir, ignore_errors=True)
    if upload_client is not None:
        upload_client.upload_fire(fire_dir, record.split, record.fire_id)
        if config.remove_local_fire_after_upload:
            shutil.rmtree(fire_dir, ignore_errors=True)
    return fire_dir


def _common_arguments(parser):
    parser.add_argument("--source-repo", default="SamuelWu318/ts-satfire")
    parser.add_argument("--repo-id", default=None, help="Dedicated Hugging Face destination dataset repo")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token; defaults to HF_TOKEN")
    parser.add_argument("--output-dir", type=Path, default=Path("data_high_resolution"))
    parser.add_argument("--work-dir", type=Path, default=Path("high_resolution_work"))
    parser.add_argument("--roi-dir", type=Path, default=Path("high_resolution_work/roi"))
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--resolution-m", type=float, default=10.0, help="Target grid pixel spacing in meters; minimum 10")
    parser.add_argument("--patch-size", type=int, default=256, help="Patch side length in target-grid pixels")
    parser.add_argument("--patch-stride", type=int, default=256, help="Patch spacing in target-grid pixels; smaller values overlap")
    parser.add_argument("--ts-length", type=int, default=10, help="Number of daily input frames per example")
    parser.add_argument("--window-stride-days", type=int, default=3)
    parser.add_argument("--tessera-strip-rows", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-source-files", action="store_true", help="Keep each fire's downloaded GeoTIFFs after processing")
    parser.add_argument("--delete-local-after-upload", action="store_true")
    parser.add_argument("--only-fires", nargs="+", help="Limit work to these fire IDs")
    parser.add_argument("--max-fires", type=int, default=None, help="Limit work to the first N matching fires")


def _config_from_args(args) -> PipelineConfig:
    return PipelineConfig(
        source_repo=args.source_repo,
        destination_repo=args.repo_id,
        token=args.token,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        roi_dir=args.roi_dir,
        splits=tuple(args.splits),
        resolution_m=args.resolution_m,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        time_series_length=args.ts_length,
        window_stride_days=args.window_stride_days,
        tessera_strip_rows=args.tessera_strip_rows,
        overwrite=args.overwrite,
        remove_source_files_after_fire=not args.keep_source_files,
        remove_local_fire_after_upload=args.delete_local_after_upload,
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("preflight", "Inspect one fire's real raster metadata and estimate data volume"),
        ("process", "Build patch files locally; does not upload"),
        ("upload", "Upload already-generated fire directories"),
        ("download", "Download a Hugging Face patch dataset snapshot"),
        ("run", "Build and upload one fire at a time"),
    ):
        sub = commands.add_parser(name, help=help_text)
        _common_arguments(sub)
    return parser


def _estimate(config, record, grid):
    patches = len(list(iter_patch_specs(grid, config.patch_size, config.patch_stride)))
    windows_per_patch = max(0, (len(record.remote_files.get("VIIRS_Day", ())) - config.time_series_length - 1) // config.window_stride_days + 1)
    cells = config.patch_size ** 2
    # Raw-size estimate before NPZ compression. FP16 input/TESSERA and uint8 labels.
    tessera_bytes = patches * 128 * cells * 2
    window_bytes = patches * windows_per_patch * (27 * config.time_series_length * cells * 2 + cells)
    return patches, windows_per_patch, tessera_bytes, window_bytes


def _get_records(config):
    return load_fire_records(config.source_repo, config.roi_dir, config.token, config.splits)


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    if args.command == "download":
        if not config.destination_repo:
            raise ValueError("--repo-id is required for download")
        client = HuggingFaceDataset(config.destination_repo, config.token, ensure_repo=False)
        print(client.download(config.output_dir))
        return 0

    if args.command == "upload":
        if not config.destination_repo:
            raise ValueError("--repo-id is required for upload")
        client = HuggingFaceDataset(config.destination_repo, config.token, private=True)
        card_path = write_dataset_card(config.output_dir / _dataset_tag(config), config)
        client.upload_dataset_card(card_path)
        wanted = {str(fire_id) for fire_id in args.only_fires} if args.only_fires else None
        local_fires = []
        for split in config.splits:
            split_dir = config.output_dir / _dataset_tag(config) / split
            if split_dir.is_dir():
                local_fires.extend((split, path) for path in sorted(split_dir.glob("p*")) if path.is_dir())
        local_fires = [
            (split, path) for split, path in local_fires
            if wanted is None or path.name[1:] in wanted
        ]
        if args.max_fires is not None:
            if args.max_fires <= 0:
                raise ValueError("--max-fires must be positive")
            local_fires = local_fires[:args.max_fires]
        for split, fire_dir in local_fires:
            if (fire_dir / "manifest.json").is_file():
                client.upload_fire(fire_dir, split, fire_dir.name[1:])
                if config.remove_local_fire_after_upload:
                    shutil.rmtree(fire_dir, ignore_errors=True)
            else:
                print(f"{fire_dir.name}: no complete local manifest; skipping upload")
        return 0

    records = _get_records(config)
    if args.only_fires:
        wanted = {str(fire_id) for fire_id in args.only_fires}
        records = [record for record in records if record.fire_id in wanted]
        missing = wanted.difference(record.fire_id for record in records)
        if missing:
            raise ValueError(f"Requested fire IDs were not found in the selected splits: {sorted(missing)}")
    if args.max_fires is not None:
        if args.max_fires <= 0:
            raise ValueError("--max-fires must be positive")
        records = records[:args.max_fires]
    if not records:
        raise RuntimeError("No fires with VIIRS_Day imagery found for the requested splits")

    if args.command == "preflight":
        record = records[0]
        local_dir = config.work_dir / "preflight" / f"p{record.fire_id}"
        first_day = record.remote_files["VIIRS_Day"][0]
        files = download_fire_files(
            record, config.source_repo, local_dir, config.token, selected_files=(first_day,)
        )
        grid = build_target_grid(files[first_day], config.resolution_m)
        patches, windows, tessera_bytes, window_bytes = _estimate(config, record, grid)
        print(json.dumps({
            "fire_id": record.fire_id,
            "split": record.split,
            "source_pixel_size_m": grid.source_pixel_size,
            "target_grid_shape": [grid.height, grid.width],
            "resolution_m": config.resolution_m,
            "patch_size_pixels": config.patch_size,
            "patch_stride_pixels": config.patch_stride,
            "patch_count": patches,
            "approx_windows_per_patch": windows,
            "uncompressed_tessera_bytes_for_fire": tessera_bytes,
            "uncompressed_window_bytes_for_fire": window_bytes,
            "estimated_total_gib_before_compression": (tessera_bytes + window_bytes) / (1024 ** 3),
        }, indent=2))
        return 0

    upload_client = None
    if args.command == "run":
        if not config.destination_repo:
            raise ValueError("--repo-id is required for run")
        upload_client = HuggingFaceDataset(config.destination_repo, config.token, private=True)
    dataset_dir = config.output_dir / _dataset_tag(config)
    card_path = write_dataset_card(dataset_dir, config)
    if upload_client is not None:
        upload_client.upload_dataset_card(card_path)
    tessera_client = _tessera_client()
    for index, record in enumerate(records, start=1):
        if record.fire_year - 1 not in tessera_client.years:
            print(f"{record.fire_id}: skipped; no TESSERA layer for prior year {record.fire_year - 1}")
            continue
        print(f"[{index}/{len(records)}] {record.split} fire {record.fire_id}")
        started = time.time()
        process_fire(record, config, tessera_client=tessera_client, upload_client=upload_client)
        print(f"{record.fire_id}: completed in {(time.time() - started) / 60:.1f} minutes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
