"""Record everything the live pipeline computes, for one source tree.

    python tests/equivalence/snapshot.py --tree <repo checkout> --tier quick|full \
        --work <scratch dir> --out <file.pkl>

Run by compare.py once per tree, each in its own process. Symbols are looked
up through CANDIDATES (new location first, then the pre-restructure one), so
the same script snapshots the baseline commit and the restructured tree.
"""

import argparse
import contextlib
import hashlib
import importlib
import io
import os
import pickle
import re
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fake_wandb  # noqa: E402

# The trainer imports wandb at module level; it must see the fake.
sys.modules["wandb"] = fake_wandb

import synth  # noqa: E402

# ----- symbol lookup: new location first, then the baseline location ----- #

CANDIDATES = {
    "config": ["firepred.config.dataset", "satimg_dataset_processor.config"],
    "write_fire_archive": [
        "firepred.data.archive:write_fire_archive",
        "satimg_dataset_processor.satimg_dataset_processor:PredDatasetProcessor._save_windowed_npz",
    ],
    "build_fire_windows": [
        "firepred.prep.windows:build_fire_windows",
        "satimg_dataset_processor.satimg_dataset_processor:PredDatasetProcessor",
    ],
    "binary_mask": [
        "firepred.prep.windows:binary_mask",
        "satimg_dataset_processor.satimg_dataset_processor:PredDatasetProcessor._binary_label",
    ],
    "fire_name": ["firepred.data.archive:fire_name", "satimg_dataset_processor.data_generator_pred_torch:fire_name"],
    "find_fire_sources": [
        "firepred.data.archive:find_fire_sources",
        "satimg_dataset_processor.data_generator_pred_torch:find_fire_sources",
    ],
    "unzip_data_root": [
        "firepred.data.archive:unzip_data_root",
        "satimg_dataset_processor.data_generator_pred_torch:unzip_data_root",
    ],
    "compute_tessera_stats": [
        "firepred.data.tessera:compute_tessera_stats",
        "satimg_dataset_processor.data_generator_pred_torch:compute_tessera_stats",
    ],
    "FireDataset": ["firepred.data.dataset:FireDataset", "satimg_dataset_processor.data_generator_pred_torch:FireDataset"],
    "crop_grid": ["firepred.prep.tessera_grid:crop_grid", "satimg_dataset_processor.tessera:crop_grid"],
    "utm_zones": ["firepred.prep.tessera_grid:utm_zones", "satimg_dataset_processor.tessera:utm_zones"],
    "grid_tessera": ["firepred.prep.tessera_grid:grid_tessera", "satimg_dataset_processor.tessera:grid_tessera"],
    "save_tessera_npz": ["firepred.prep.tessera_grid:save_tessera_npz", "satimg_dataset_processor.tessera:save_tessera_npz"],
    "fire_label_sel": ["firepred.hub.splits:fire_label_sel", "satimg_dataset_processor.hf_dataset:fire_label_sel"],
    "fire_start_years": ["firepred.hub.splits:fire_start_years", "satimg_dataset_processor.tessera:fire_start_years"],
    "zip_data_from": ["firepred.hub.zips:zip_data_from", "satimg_dataset_processor.hf_dataset:zip_data_from"],
    "build_model": ["firepred.models.registry:build_model", "run_spatial_temp_model_pred:build_model"],
    "forward_model": ["firepred.models.registry:forward_model", "run_spatial_temp_model_pred:forward_model"],
    "build_criterion": ["firepred.training.losses:build_criterion", "run_spatial_temp_model_pred:build_criterion"],
    "compute_loss": ["firepred.training.losses:compute_loss", "run_spatial_temp_model_pred:compute_loss"],
    "fire_only_f1": ["firepred.training.metrics:fire_only_f1", "run_spatial_temp_model_pred:fire_only_f1"],
    "background_f1": ["firepred.training.metrics:background_f1", "run_spatial_temp_model_pred:background_f1"],
    "fire_only_iou": ["firepred.training.metrics:fire_only_iou", "run_spatial_temp_model_pred:fire_only_iou"],
    "build_lr_scheduler": ["firepred.training.optim:build_lr_scheduler", "run_spatial_temp_model_pred:build_lr_scheduler"],
    "metric_labels": [
        "firepred.training.tracking:build_wandb_metric_labels",
        "run_spatial_temp_model_pred:build_wandb_metric_labels",
    ],
    "process_data": ["firepred.hub.publish_windows:process_data", "satimg_dataset_processor.hf_dataset:process_data"],
    "download_processed_data": [
        "firepred.hub.download:download_processed_data",
        "satimg_dataset_processor.hf_dataset:download_processed_data",
    ],
    "upload_data_zip": ["firepred.hub.zips:upload_data_zip", "satimg_dataset_processor.hf_dataset:upload_data_zip"],
    "pull_zip_data_to": ["firepred.hub.zips:pull_zip_data_to", "satimg_dataset_processor.hf_dataset:pull_zip_data_to"],
    "process_tessera": ["firepred.hub.publish_tessera:process_tessera", "satimg_dataset_processor.tessera:process_tessera"],
    "upload_tessera_zip": [
        "firepred.hub.publish_tessera:upload_tessera_zip",
        "satimg_dataset_processor.tessera:upload_tessera_zip",
    ],
    "pull_tessera_data": ["firepred.hub.zips:pull_tessera_data", "satimg_dataset_processor.tessera:pull_tessera_data"],
    "train_cli": ["firepred.cli.train", "run_spatial_temp_model_pred"],
    "dataset_cli": ["firepred.cli.dataset", "dataset_gen_pred"],
}


def resolve(name):
    errors = []
    for candidate in CANDIDATES[name]:
        module_name, _, attribute = candidate.partition(":")
        try:
            value = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if not (error.name and module_name.startswith(error.name)):
                raise
            errors.append(f"{candidate}: {error}")
            continue
        try:
            for part in filter(None, attribute.split(".")):
                value = getattr(value, part)
        except AttributeError as error:
            errors.append(f"{candidate}: {error}")
            continue
        return value
    raise LookupError(f"{name}: none of {errors}")


def is_new(name):
    """True when `name` resolves to its first (restructured) candidate."""
    module_name = CANDIDATES[name][0].partition(":")[0]
    try:
        importlib.import_module(module_name)
        return True
    except ModuleNotFoundError:
        return False


# ----- adapters for functions whose signature changed ----- #


def write_fire_archive(path, data, labels, label_sel=None):
    return resolve("write_fire_archive")(path, data, labels, label_sel=label_sel)


def build_fire_windows(loc_name, local_map, length, interval, label_sel):
    target = resolve("build_fire_windows")
    if isinstance(target, type):
        target = target().process_location
    return target(loc_name, local_map, length=length, interval=interval, label_sel=label_sel, debug=False)


def make_model(name, channels, ts_length, heads, width, unetr_version="v1", temporal_downsample=False):
    if is_new("build_model"):
        ModelSpec = importlib.import_module("firepred.models.spec").ModelSpec
        spec = ModelSpec(
            in_channels=channels,
            num_classes=2,
            ts_length=ts_length,
            num_heads=heads,
            feature_size=width,
            options={"unetr_version": unetr_version, "swin_temporal_downsample": temporal_downsample},
        )
        return resolve("build_model")(name, spec)
    return resolve("build_model")(
        model_name=name,
        model_input_channels=channels,
        num_classes=2,
        ts_length=ts_length,
        num_heads=heads,
        hidden_size=width,
        unetr_version=unetr_version,
        swin_temporal_downsample=temporal_downsample,
    )


def forward(model, name, batch):
    import torch

    if is_new("forward_model"):
        return resolve("forward_model")(model, batch)
    return resolve("forward_model")(model, name, batch, torch.device("cpu"))


# ----- conversion to comparable plain values ----- #

HASH_ABOVE = 4096


def plain(value, normalize=None):
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype == object:
            return ("object-array", [plain(item, normalize) for item in array.ravel().tolist()])
        if array.size > HASH_ABOVE:
            digest = hashlib.sha256(array.tobytes()).hexdigest()
            summary = float(array.astype(np.float64).sum()) if array.dtype.kind in "fiub" else None
            return ("array-hash", str(array.dtype), array.shape, digest, summary)
        return ("array", str(array.dtype), array.shape, array.tobytes())
    if isinstance(value, np.generic):
        return ("np", str(value.dtype), value.tobytes())
    if isinstance(value, dict):
        return ("dict", [(plain(k, normalize), plain(v, normalize)) for k, v in value.items()])
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [plain(item, normalize) for item in value])
    if isinstance(value, Path):
        return ("path", normalize(str(value)) if normalize else str(value))
    if isinstance(value, str):
        return normalize(value) if normalize else value
    if isinstance(value, torch.device):
        return ("device", str(value))
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return ("repr", type(value).__name__, repr(value))


def capture_parser(call):
    """Run `call` and return (its result, the ArgumentParser it parsed with)."""
    import argparse as argparse_module

    seen = []
    original = argparse_module.ArgumentParser.parse_args

    def spy(self, *args, **kwargs):
        seen.append(self)
        return original(self, *args, **kwargs)

    argparse_module.ArgumentParser.parse_args = spy
    try:
        result = call()
    finally:
        argparse_module.ArgumentParser.parse_args = original
    return result, seen[0]


def describe_parser(parser, normalize):
    import argparse as argparse_module

    rows = []
    for action in parser._actions:
        if isinstance(action, argparse_module._SubParsersAction):
            for command, subparser in action.choices.items():
                rows.append(("subcommand", command, describe_parser(subparser, normalize)))
            continue
        rows.append(
            (
                tuple(action.option_strings),
                action.dest,
                plain(action.default, normalize),
                action.help,
                plain(action.choices, normalize),
                action.nargs,
                getattr(action.type, "__name__", repr(action.type)),
                plain(action.metavar),
                action.required,
            )
        )
    return rows


# Train flags added after 6ebc222 (see README "Intentional differences").
NEW_TRAIN_OPTIONS = {"model_option", "trust_checkpoint", "lambda_focal", "focal_gamma", "focal_alpha", "device"}
# Dataset flags removed after 6ebc222 with the old-format conversion.
REMOVED_DATASET_OPTIONS = {"processed_repo"}


# ----- quick tier ----- #


def quick(snap, work, tree, normalize):
    import numpy as np
    import torch

    torch.manual_seed(0)
    np.random.seed(0)

    # Config values.
    config = resolve("config")
    for key in (
        "TS_LENGTH", "TS_INTERVAL", "SRC_REPO", "TESSERA_REPO",
        "WINDOW_CONFIG_FILE", "NPZ_COMPRESSLEVEL", "TRAIN_VAL_LABEL_SEL", "TRAIN_VAL_YEARS", "TEST_YEARS", "VAL_IDS",
    ):
        snap[f"config.{key}"] = plain(getattr(config, key))
    snap["config.window_repo_for"] = [config.window_repo_for(ts, it) for ts, it in ((6, 1), (10, 3), (6, 3))]

    # Window archive format: write, then decode every member.
    rng = np.random.default_rng(3)
    data = rng.normal(size=(3, 27, 2, 8, 8)).astype(np.float32)
    labels = (rng.random((3, 8, 8)) < 0.3).astype(np.float32)
    for label_sel in (None, 0):
        path = work / f"archive_{label_sel}.npz"
        count = write_fire_archive(path, data, labels, label_sel=label_sel)
        with np.load(path, allow_pickle=False) as archive:
            snap[f"archive.write.{label_sel}"] = (count, archive.files, [plain(archive[k]) for k in archive.files])

    # Data root reading: sources, unzip cache, FireDataset items.
    root, _ = synth.data_root(work / "data", synth.QUICK_FIRES)
    fire_name = resolve("fire_name")
    find_fire_sources = resolve("find_fire_sources")
    sources = {split: find_fire_sources(root / split) for split in ("train", "val", "test", "missing")}
    snap["archive.sources"] = plain(sources, normalize)
    unzip_root = resolve("unzip_data_root")(root, work / "data-npy", splits=("train", "val", "test"))
    snap["archive.unzip_root"] = plain(unzip_root, normalize)
    snap["archive.unzipped_files"] = sorted(
        str(path.relative_to(work / "data-npy")) for path in (work / "data-npy").rglob("*") if path.is_file()
    )
    snap["archive.unzip_again"] = plain(resolve("unzip_data_root")(root, work / "data-npy"), normalize)
    snap["archive.fire_name"] = [fire_name(path) for path in find_fire_sources(work / "data-npy" / "train")]

    FireDataset = resolve("FireDataset")
    kwargs = dict(ts_length=6, n_channel=27, target_is_single_day=True)

    def items(dataset, seed=None, prefix=""):
        if seed is not None:
            np.random.seed(seed)
        rows = []
        for index in range(len(dataset)):
            sample = dataset[index]
            rows.append({key: plain(value) for key, value in sample.items()})
        return rows

    plain_ds = FireDataset(npz_dir=root / "train", use_augmentations=False, **kwargs)
    snap["dataset.len"] = len(plain_ds)
    snap["dataset.index"] = [(fire_name(path), window) for path, window in plain_ds.index]
    snap["dataset.normalizer"] = plain((plain_ds.normalizer.mean, plain_ds.normalizer.std, sorted(plain_ds.normalizer.dont_normalize_idc)))
    snap["dataset.items.plain"] = items(plain_ds)
    snap["dataset.items.augmented"] = items(FireDataset(npz_dir=root / "train", use_augmentations=True, **kwargs), seed=123)
    snap["dataset.items.folders"] = items(
        FireDataset(npz_dir=work / "data-npy" / "train", use_augmentations=True, **kwargs), seed=123
    )
    snap["dataset.items.val"] = items(FireDataset(npz_dir=root / "val", use_augmentations=False, **kwargs))

    tessera_paths = [root / "tessera" / f"{fire_name(path)}.npz" for path in find_fire_sources(root / "train")]
    stats = resolve("compute_tessera_stats")(tessera_paths, min_coverage=0.5)
    snap["tessera.stats"] = plain(stats)
    snap["tessera.stats.0.7"] = plain(resolve("compute_tessera_stats")(tessera_paths, min_coverage=0.7))
    tessera_ds = FireDataset(
        npz_dir=root / "train", use_augmentations=True, tessera_dir=root / "tessera", tessera_stats=stats, **kwargs
    )
    snap["dataset.items.tessera"] = items(tessera_ds, seed=7)

    # GeoTIFF -> windows.
    local_map = synth.fire_geotiffs(work / "geotiffs")
    binary_mask = resolve("binary_mask")
    snap["windows.binary_mask"] = plain(binary_mask(np.array([np.nan, np.inf, -np.inf, -1.0, 0.0, 0.5, 2.0])))
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        for length, interval, label_sel in ((6, 1, 0), (3, 2, 1), (6, 3, 0), (8, 1, 0)):
            loc_x, loc_y = build_fire_windows("fire", local_map, length, interval, label_sel)
            snap[f"windows.{length}.{interval}.{label_sel}"] = (plain(loc_x), plain(loc_y))
    snap["windows.stdout"] = normalize(sink.getvalue())

    # TESSERA grid (offline part of the TESSERA build).
    import rasterio

    with rasterio.open(next(iter(local_map.values()))) as reader:
        profile = reader.profile
    crs, transform = resolve("crop_grid")(profile)
    snap["tessera.crop_grid"] = (crs.to_string(), tuple(transform))
    snap["tessera.utm_zones"] = [resolve("utm_zones")(*bounds) for bounds in ((-123.5, -122.0), (-126.1, -125.9), (-120.0, -114.0))]
    embedding, coverage, source_pixels, bounds = resolve("grid_tessera")(
        synth.FakeGeoTessera(), crs, transform, 2019, strip_rows=128
    )
    snap["tessera.grid"] = plain((embedding, coverage, source_pixels, bounds))
    resolve("save_tessera_npz")(work / "tessera_out.npz", embedding, coverage, {"fire_id": "x", "tessera_year": np.int16(2019)})
    with np.load(work / "tessera_out.npz", allow_pickle=False) as archive:
        snap["tessera.saved"] = (archive.files, [plain(archive[key]) for key in archive.files])

    # Hugging Face helpers that run offline.
    roi_paths = sorted(str(path) for path in (tree / "roi").glob("*.csv"))
    snap["hub.fire_label_sel"] = plain(resolve("fire_label_sel")(roi_paths))
    snap["hub.fire_start_years"] = plain(resolve("fire_start_years")(roi_paths))
    with contextlib.redirect_stdout(io.StringIO()):
        resolve("zip_data_from")([root / "val", root / "test", root / "absent"], work / "bundle.zip")
    import zipfile

    with zipfile.ZipFile(work / "bundle.zip") as bundle:
        snap["hub.zip"] = [(info.filename, info.compress_type, info.file_size, info.CRC) for info in bundle.infolist()]

    # Models: real configurations on a 64x64 input.
    from torch import nn, optim

    build_criterion = resolve("build_criterion")
    compute_loss = resolve("compute_loss")
    for channels, temporal_downsample in ((43, False), (172, False), (43, True)):
        key = f"model.swinunetr3d.{channels}.{temporal_downsample}"
        torch.manual_seed(42)
        model = nn.DataParallel(make_model("swinunetr3d", channels, 6, 3, 36, temporal_downsample=temporal_downsample))
        snap[key + ".init"] = plain(model.state_dict())
        torch.manual_seed(5)
        x = torch.randn(2, channels, 6, 64, 64)
        y = torch.stack(((torch.rand(2, 64, 64) > 0.1), (torch.rand(2, 64, 64) > 0.9)), dim=1).float()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        outputs = forward(model, "swinunetr3d", x)
        loss, parts = compute_loss(build_criterion("dicece"), outputs, y)
        loss.backward()
        optimizer.step()
        snap[key + ".train_out"] = plain(outputs)
        snap[key + ".loss"] = plain((loss, parts))
        snap[key + ".after_step"] = plain(model.state_dict())
        snap[key + ".optimizer"] = plain(optimizer.state_dict())
        model.eval()
        with torch.no_grad():
            snap[key + ".eval_out"] = plain(forward(model, "swinunetr3d", x))

    for name, size, version in (("unet3d", 64, "v1"), ("attunet", 64, "v1"), ("unetr3d", 256, "v1"), ("unetr3d", 256, "v0")):
        key = f"model.{name}.{version}"
        torch.manual_seed(42)
        try:
            model = nn.DataParallel(make_model(name, 43, 6, 3, 36, unetr_version=version))
        except Exception as error:  # unetr3d fails to build at baseline; keep that behavior visible
            snap[key + ".error"] = (type(error).__name__, str(error))
            continue
        snap[key + ".init"] = plain(model.state_dict())
        torch.manual_seed(5)
        model.eval()
        with torch.no_grad():
            snap[key + ".eval_out"] = plain(forward(model, name, torch.randn(1, 43, 6, size, size)))

    # Losses and metrics on fixed tensors.
    torch.manual_seed(11)
    logits = torch.randn(3, 2, 32, 32) * 3
    target_fire = torch.rand(3, 32, 32) > 0.95
    target_fire[2] = False
    labels_2c = torch.stack((~target_fire, target_fire), dim=1).float()
    for loss_args in (("dice",), ("dicece",), ("dicece", (1.0, 457.1), 0.7, 1.3)):
        criterion = build_criterion(*loss_args)
        snap[f"loss.{loss_args}"] = plain(compute_loss(criterion, logits, labels_2c))
    predictions = (torch.sigmoid(logits) > 0.5).float()
    predictions[2] = 0
    for metric in ("fire_only_f1", "background_f1", "fire_only_iou"):
        snap[f"metric.{metric}"] = plain(resolve(metric)(predictions, labels_2c))

    # Learning-rate schedules.
    for schedule in ("none", "plateau", "cosine"):
        parameter = nn.Parameter(torch.zeros(1))
        optimizer = optim.Adam([parameter], lr=1e-3)
        scheduler = resolve("build_lr_scheduler")(optimizer, schedule, 6, 0.5, 1, 1e-4)
        rates = []
        for val_loss in (1.0, 0.9, 0.95, 0.96, 0.97, 0.5):
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif scheduler is not None:
                scheduler.step()
            rates.append(optimizer.param_groups[0]["lr"])
        snap[f"lr.{schedule}"] = (type(scheduler).__name__, rates)

    snap["wandb.labels.dice"] = plain(resolve("metric_labels")("dice"))
    snap["wandb.labels.dicece"] = plain(resolve("metric_labels")("dicece"))

    # Command-line interfaces: parsed defaults and every option.
    train_cli = resolve("train_cli")
    sys.argv = ["train", "-m", "swinunetr3d", "-mode", "pred", "-b", "8", "-r", "1", "-lr", "0.001",
                "-nh", "3", "-ed", "36", "-ts", "6", "-it", "1"]
    args, parser = capture_parser(train_cli.parse_args)
    for key, value in vars(args).items():
        snap[f"cli.train.default.{key}"] = plain(value, normalize)
    for row in describe_parser(parser, normalize):
        snap[f"cli.train.option.{row[1]}"] = row
    snap["cli.train.option_order"] = [row[1] for row in describe_parser(parser, normalize) if row[1] not in NEW_TRAIN_OPTIONS]
    dataset_cli = resolve("dataset_cli")
    for command in ("process", "tessera", "tessera-zip", "pull-tessera", "download", "upload-zip", "pull-zip"):
        sys.argv = ["dataset", command]
        args, parser = capture_parser(dataset_cli.parse_args)
        kept = {key: value for key, value in vars(args).items() if key not in REMOVED_DATASET_OPTIONS}
        snap[f"cli.dataset.{command}"] = plain(kept, normalize)
    snap["cli.dataset.options"] = describe_parser(parser, normalize)



# ----- hub tier: the Hugging Face dataset pipeline against an offline fake hub ----- #

HUB_FIRES = [  # (year, fire id, CSV label_sel); 20568194 is a validation id, 2017 has no prior-year TESSERA
    ("2017", "20000017", None),
    ("2019", "20000019", None),
    ("2019", "20568194", None),
    ("2021", "US_2021_TEST0001", 1),
]


def install_fake_hub(work):
    """Replace huggingface_hub and geotessera before any pipeline module is imported."""
    import importlib.metadata
    import types

    import fake_hub

    fake_hub.ROOT = str(work / "hub")
    fake_hub.CALLS.clear()
    sys.modules["huggingface_hub"] = fake_hub

    class GeoTesseraZarr(synth.FakeGeoTessera):
        years = list(range(2017, 2025))

        def __init__(self, url=None):
            super().__init__()
            self.url = url or "fake://tessera"

    module = types.ModuleType("geotessera")
    module.GeoTesseraZarr = GeoTesseraZarr
    sys.modules["geotessera"] = module
    real_version = importlib.metadata.version
    importlib.metadata.version = lambda name: "0.0-fake" if name == "geotessera" else real_version(name)
    return fake_hub


def build_source_repo(fake_hub, repo_id):
    """Raw GeoTIFFs and ROI CSVs laid out like SamuelWu318/ts-satfire."""
    root = fake_hub._repo(repo_id)
    rows = {}
    for index, (year, fire_id, label_sel) in enumerate(HUB_FIRES):
        synth.fire_geotiffs(root / year / fire_id, n_days=7, seed=10 + index)
        rows.setdefault(year, []).append((fire_id, label_sel, index))
    for year, fires in rows.items():
        header = "Id,area,lat,lon,start_date,end_date" + (",label_sel" if year == "2021" else "")
        lines = [header] + [
            f"{fire_id},{100 + index},37.5,-122.4,{year}-08-01,{year}-08-20" + (f",{label_sel}" if year == "2021" else "")
            for fire_id, label_sel, index in fires
        ]
        (root / f"us_fire_{year}_out_new.csv").write_text("\n".join(lines) + "\n")


def repo_contents(fake_hub, repo_id, skip_members=()):
    """Every file of a fake repo: arrays decoded for .npz, text for .json/.txt, CRCs for .zip."""
    import zipfile

    import numpy as np

    root = fake_hub._repo(repo_id)
    out = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        name = str(path.relative_to(root))
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                out[name] = [(key, plain(archive[key])) for key in archive.files if key not in skip_members]
        elif path.suffix == ".zip":
            members = []
            with zipfile.ZipFile(path) as bundle:
                for info in bundle.infolist():
                    if info.filename.endswith(".npz"):  # decode, so skip_members applies inside zips too
                        with np.load(io.BytesIO(bundle.read(info)), allow_pickle=False) as archive:
                            members.append((info.filename, [(k, plain(archive[k])) for k in archive.files
                                                            if k not in skip_members]))
                    else:
                        members.append((info.filename, info.file_size, info.CRC))
            out[name] = members
        elif path.suffix in (".json", ".txt", ".csv"):
            out[name] = path.read_text()
        else:
            out[name] = ("bytes", path.stat().st_size)
    return out


def files_under(folder):
    folder = Path(folder)
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()) if folder.exists() else []


def hub(snap, work, tree, normalize):
    import shutil

    # Start from empty fake repos: leftovers would make process_data skip fires.
    for child in work.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    fake_hub = install_fake_hub(work)
    src, windows, tessera = "owner/src", "owner/windows-6-1", "owner/tessera"
    build_source_repo(fake_hub, src)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        resolve("process_data")(
            "token", src_repo=src, window_repo=windows, batch_size=2, length=6, interval=1,
            splits=("train", "val", "test"), roi_dir=str(work / "roi1"), work_dir=str(work / "pw"),
        )
        window_repo_files = repo_contents(fake_hub, windows)
        # window_config.json gained settings after 6ebc222; compared on its own key.
        snap["hub.window_config"] = window_repo_files.pop("window_config.json", None)
        snap["hub.process_data.repo"] = plain(window_repo_files)

        loader = resolve("download_processed_data")(
            "token", window_repo=windows, src_repo=src, base_output_dir=str(work / "dl"), roi_dir=str(work / "roi2"),
        )
        snap["hub.download.files"] = files_under(work / "dl")
        snap["hub.download.excluded"] = plain(getattr(loader, "excluded_npz", None))

        skipped = resolve("process_tessera")(
            "token", src_repo=src, tessera_repo=tessera, batch_size=1, work_dir=str(work / "pt"),
            roi_dir=str(work / "roi3"),
        )
        snap["hub.tessera.skipped"] = plain(skipped)
        missing = resolve("upload_tessera_zip")(
            "token", src_repo=src, tessera_repo=tessera, work_dir=str(work / "tz"), roi_dir=str(work / "roi4"),
        )
        snap["hub.tessera_zip.missing"] = plain(missing)
        # code_version is the git commit of the tree, which differs by design.
        snap["hub.tessera.repo"] = plain(repo_contents(fake_hub, tessera, skip_members=("code_version",)))

        loader = resolve("download_processed_data")(
            "token", window_repo=windows, src_repo=src, base_output_dir=str(work / "dl_t"),
            roi_dir=str(work / "roi5"), include_tessera=True, tessera_repo=tessera,
        )
        snap["hub.download_tessera.files"] = files_under(work / "dl_t")
        snap["hub.download_tessera.excluded"] = plain(loader.excluded_npz)

        resolve("upload_data_zip")(
            "token", data_root=str(work / "dl"), zip_path=str(work / "data.zip"), repo_id=windows,
        )
        resolve("pull_zip_data_to")(str(work / "pulled"), token="token", repo_id=windows, download_dir=str(work / "pulled"))
        removed = resolve("pull_tessera_data")(str(work / "pulled"), token="token", tessera_repo=tessera)
        snap["hub.pull.files"] = files_under(work / "pulled")
        snap["hub.pull_tessera.removed"] = plain(removed)

    # Read-only listings are left out: only downloads and writes must match.
    snap["hub.calls"] = plain([call for call in fake_hub.CALLS if call[0] != "list_repo_files"], normalize)
    # tempfile.mkdtemp names are random by design.
    stdout = re.sub(r"windowed_npz_\w+", "windowed_npz_<random>", normalize(sink.getvalue()))
    # download_processed_data prints the window config, compared as hub.window_config.
    snap["hub.stdout"] = re.sub(r"Window repo (\S+): .*", r"Window repo \1: <window_config>", stdout)


# ----- full tier: end-to-end training runs ----- #

E2E_BASE = ["-m", "swinunetr3d", "-mode", "pred", "-b", "2", "-r", "1", "-lr", "0.001", "-nh", "3",
            "-ed", "12", "-ts", "6", "-it", "1", "-seed", "42", "--epochs", "2"]


def run_main(snap, key, argv, case_dir, normalize):
    import numpy as np
    import torch

    case_dir.mkdir(parents=True, exist_ok=True)
    os.environ["FAKE_WANDB_DIR"] = str(case_dir)
    fake_wandb.CALLS.clear()
    fake_wandb.config.clear()
    fake_wandb.run = None
    train_cli = resolve("train_cli")
    sys.argv = ["train"] + argv
    sink = io.StringIO()
    cwd = os.getcwd()
    os.chdir(case_dir)
    try:
        with contextlib.redirect_stdout(sink):
            train_cli.main()
            fake_wandb.finish()
    finally:
        os.chdir(cwd)
    snap[f"{key}.wandb"] = plain(fake_wandb.CALLS, normalize)
    snap[f"{key}.stdout"] = normalize(sink.getvalue())
    checkpoints = sorted(case_dir.glob("*.pth"))
    snap[f"{key}.checkpoints"] = [path.name for path in checkpoints]
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        snap[f"{key}.checkpoint.{path.name}"] = plain(checkpoint, normalize)
    snap[f"{key}.plots"] = sorted(path.name for path in case_dir.rglob("*.png"))
    snap[f"{key}.rng"] = (float(np.random.random()), float(torch.rand(1)))
    return checkpoints


def full(snap, work, tree, normalize, foreign_checkpoint=None):
    root, csv_path = synth.data_root(work / "e2e", synth.E2E_FIRES)
    # "--unzip-dir auto" was the default at 6ebc222; pinned so both trees take the same path.
    common = ["--data-root", str(root), "--test-csv", str(csv_path), "--test-every", "1", "--unzip-dir", "auto"]
    checkpoints = run_main(
        snap, "e2e.dicece", E2E_BASE + common + ["--loss", "dicece", "--upload-checkpoints", "--plot-dir", "plots"],
        work / "case_dicece", normalize,
    )
    run_main(
        snap, "e2e.dice_tessera_plateau",
        E2E_BASE + common + ["--loss", "dice", "--tessera", "--lr-schedule", "plateau", "--lr-patience", "0",
                             "--plot-dir", "plots"],
        work / "case_tessera", normalize,
    )
    run_main(
        snap, "e2e.workers_cosine",
        E2E_BASE + ["--data-root", str(root), "--test-csv", str(csv_path), "--num-workers", "2",
                    "--unzip-dir", "off", "--lr-schedule", "cosine", "--no-test-after-train"],
        work / "case_workers", normalize,
    )
    for label, checkpoint in (("own", checkpoints[0]), ("foreign", foreign_checkpoint)):
        if checkpoint is None:
            continue
        run_main(
            snap, f"e2e.test_{label}",
            E2E_BASE + ["--data-root", str(root), "--test-csv", str(csv_path), "--loss", "dicece", "-test",
                        "--checkpoint", str(checkpoint), "--plot-dir", "plots", "--unzip-dir", "auto"],
            work / f"case_test_{label}", normalize,
        )
    return checkpoints[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--tier", choices=("quick", "full", "hub"), required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--foreign-checkpoint", default=None)
    args = parser.parse_args()

    tree = Path(args.tree).resolve()
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(tree))
    os.chdir(work)

    import torch

    torch.set_num_threads(args.threads)

    def normalize(text):
        return text.replace(str(work), "<WORK>").replace(str(tree), "<TREE>")

    snap = OrderedDict()
    snap["layout"] = "new" if (tree / "firepred").is_dir() else "baseline"
    if args.tier == "quick":
        quick(snap, work, tree, normalize)
    elif args.tier == "hub":
        hub(snap, work, tree, normalize)
    else:
        checkpoint = full(snap, work, tree, normalize, args.foreign_checkpoint)
        snap["own_checkpoint_path"] = str(checkpoint)
    with open(args.out, "wb") as handle:
        pickle.dump(snap, handle)


if __name__ == "__main__":
    main()
