"""Golden cases: every deterministic output the refactor must keep.

Each @group function computes a dict {key: value} with the package's CURRENT
API. When a refactor moves or renames a function, only the body of the
affected group changes; the stored values in values/<group>.json must not.
So this file's history doubles as the old-name -> new-name migration table.

Groups run one per process (capture.py), because the hub group swaps in fake
huggingface_hub / geotessera modules before the pipeline is imported.
Printed output is not an output: stdout is never recorded.
"""

import contextlib
import io
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
sys.path.insert(0, str(TREE))
sys.path.insert(0, str(TREE / "tests" / "support"))

import synth  # noqa: E402

GROUPS = {}


def group(name):
    """Register `function(ctx) -> dict` as the golden group `name`."""

    def register(function):
        """Store the function under its group name and return it unchanged."""
        GROUPS[name] = function
        return function

    return register


def quiet(call, *args, **kwargs):
    """Run call(*args, **kwargs) with stdout discarded and return its result."""
    with contextlib.redirect_stdout(io.StringIO()):
        return call(*args, **kwargs)


def zip_listing(path):
    """(name, compression, size, CRC) of every member of a zip file."""
    with zipfile.ZipFile(path) as bundle:
        return [(info.filename, info.compress_type, info.file_size, info.CRC) for info in bundle.infolist()]


def files_under(folder):
    """Sorted relative paths of every file below `folder` ([] when missing)."""
    folder = Path(folder)
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()) if folder.exists() else []


def npz_members(path, skip=()):
    """[(member, array)] of an .npz file, minus the `skip` members."""
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        return [(key, archive[key]) for key in archive.files if key not in skip]


def tiny_spec(channels=43, width=12, **options):
    """A ModelSpec for a small but real model configuration."""
    from firepred.models.registry import ModelSpec

    opts = {"unetr_version": "v1", "swin_temporal_downsample": False}
    opts.update(options)
    return ModelSpec(in_channels=channels, num_classes=2, ts_length=6, num_heads=3, feature_size=width, options=opts)


# ----- configuration ----- #


@group("config")
def config_values(ctx):
    """Project constants and derived repo names."""
    from firepred.config import dataset as data, training

    out = {}
    for key in (
        "TS_LENGTH", "TS_INTERVAL", "SRC_REPO", "TESSERA_REPO", "WINDOW_CONFIG_FILE", "NPZ_COMPRESSLEVEL",
        "TRAIN_VAL_LABEL_SEL", "TRAIN_VAL_YEARS", "TEST_YEARS", "VAL_IDS", "CROP_SIZE", "CROP_OFFSET",
    ):
        out[f"dataset.{key}"] = getattr(data, key)
    out["dataset.window_repo_for"] = [data.window_repo_for(ts, it) for ts, it in ((6, 1), (10, 3), (6, 3))]
    for key in (
        "N_STORED_CHANNELS", "DEGREE_CHANNELS", "LANDCOVER_CHANNEL", "NORMALIZATION_MEAN", "NORMALIZATION_STD",
    ):
        out[f"channels.{key}"] = getattr(data, key)
    out["channels.model_input_channels"] = [data.model_input_channels(t) for t in (False, True)]
    for key in sorted(k for k in vars(training) if k.isupper()):
        out[f"training.{key}"] = getattr(training, key)
    return out


# ----- window archives and the data root ----- #


@group("archive")
def archive_values(ctx):
    """Archive format, data-root discovery and the unzip cache."""
    import numpy as np

    from firepred.data.archive import fire_name, find_fire_sources, unzip_data_root, write_fire_archive

    work = ctx.work
    out = {}
    rng = np.random.default_rng(3)
    data = rng.normal(size=(3, 27, 2, 8, 8)).astype(np.float32)
    labels = (rng.random((3, 8, 8)) < 0.3).astype(np.float32)
    for label_sel in (None, 0):
        path = work / f"archive_{label_sel}.npz"
        written = write_fire_archive(path, data, labels, label_sel=label_sel)
        members = npz_members(written)
        out[f"write.{label_sel}"] = (int(dict(members)["num_windows"]), members)
    root, _ = synth.data_root(work / "data", synth.QUICK_FIRES)
    out["sources"] = {split: find_fire_sources(root / split) for split in ("train", "val", "test", "missing")}
    out["unzip_root"] = quiet(unzip_data_root, root, work / "data-npy", splits=("train", "val", "test"))
    out["unzipped_files"] = files_under(work / "data-npy")
    out["unzip_again"] = quiet(unzip_data_root, root, work / "data-npy")
    out["fire_name"] = [fire_name(path) for path in find_fire_sources(work / "data-npy" / "train")]
    return out


# ----- training-time data: FireDataset, transforms, loader helpers ----- #


@group("dataset")
def dataset_values(ctx):
    """FireDataset items (plain, seeded-augmented, unzipped, val, TESSERA) and TESSERA stats."""
    import numpy as np
    import torch

    from firepred.data.archive import fire_name, find_fire_sources, unzip_data_root
    from firepred.data.fire_dataset import FireDataset, batch_inputs, make_dataloader
    from firepred.data.tessera_archive import compute_tessera_stats, load_tessera_maps

    work = ctx.work
    root, _ = synth.data_root(work / "data", synth.QUICK_FIRES)
    quiet(unzip_data_root, root, work / "data-npy", splits=("train",))
    kwargs = dict(ts_length=6, n_channel=27, target_is_single_day=True)

    def items(dataset, seed=None):
        """Every sample of `dataset`, after seeding numpy when `seed` is given."""
        if seed is not None:
            np.random.seed(seed)
        return [dataset[index] for index in range(len(dataset))]

    out = {}
    plain_ds = FireDataset(npz_dir=root / "train", use_augmentations=False, **kwargs)
    out["len"] = len(plain_ds)
    out["index"] = [(fire_name(path), window) for path, window in plain_ds.index]
    out["normalizer"] = (plain_ds.normalizer.mean, plain_ds.normalizer.std, sorted(plain_ds.normalizer.dont_normalize_idc))
    out["items.plain"] = items(plain_ds)
    out["items.augmented"] = items(FireDataset(npz_dir=root / "train", use_augmentations=True, **kwargs), seed=123)
    out["items.folders"] = items(
        FireDataset(npz_dir=work / "data-npy" / "train", use_augmentations=False, **kwargs)
    )
    out["items.val"] = items(FireDataset(npz_dir=root / "val", use_augmentations=False, **kwargs))

    train_maps = load_tessera_maps(root / "tessera", find_fire_sources(root / "train"))
    stats = compute_tessera_stats(train_maps.values(), min_coverage=0.5)
    out["tessera.stats"] = stats
    out["tessera.stats.0.7"] = compute_tessera_stats(train_maps.values(), min_coverage=0.7)
    tessera_ds = FireDataset(
        npz_dir=root / "train", use_augmentations=False, tessera_dir=root / "tessera", tessera_stats=stats, **kwargs
    )
    out["items.tessera"] = items(tessera_ds)
    out["items.tessera.augmented"] = items(
        FireDataset(npz_dir=root / "train", use_augmentations=True, tessera_dir=root / "tessera",
                    tessera_stats=stats, **kwargs),
        seed=7,
    )
    batch = next(iter(make_dataloader(tessera_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)))
    out["batch_inputs.tessera"] = batch_inputs(batch, torch.device("cpu"))
    batch = next(iter(make_dataloader(plain_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)))
    out["batch_inputs.plain"] = batch_inputs(batch, torch.device("cpu"))
    return out


@group("transforms")
def transform_values(ctx):
    """Normalize, preprocess, seeded augment and the land-cover lookup on small tensors."""
    import numpy as np
    import torch

    from firepred.config.dataset import NORMALIZATION_MEAN, NORMALIZATION_STD
    from firepred.data.transforms import Normalize, augment, landcover_one_hot_matrix, preprocess

    rng = np.random.default_rng(4)
    x = rng.normal(20.0, 10.0, size=(27, 3, 16, 16)).astype(np.float32)
    x[[12, 18, 24]] = rng.uniform(0, 360, size=(3, 3, 16, 16)).astype(np.float32)
    x[21] = rng.integers(0, 18, size=(3, 16, 16)).astype(np.float32)
    out = {"landcover": landcover_one_hot_matrix()}
    normalize = Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD, dont_normalize_idc=[21])
    normalized = normalize(x.copy())
    out["normalize"] = normalized
    one_hot = landcover_one_hot_matrix()
    out["preprocess"] = preprocess(torch.from_numpy(normalized.copy()), one_hot)
    y = torch.from_numpy((rng.random((2, 16, 16)) < 0.2).astype(np.float32))
    static = torch.from_numpy(rng.normal(size=(129, 16, 16)).astype(np.float32))
    for seed in (0, 1, 5):
        np.random.seed(seed)
        out[f"augment.{seed}"] = augment(torch.from_numpy(normalized.copy()), y.clone(), static.clone())
    return out


# ----- preprocessing: GeoTIFF -> windows, TESSERA grid, ROI helpers ----- #


@group("windows")
def window_values(ctx):
    """binary_mask and build_fire_windows for several (length, interval, label_sel)."""
    import numpy as np

    from firepred.data.windows import binary_mask, build_fire_windows

    local_map = synth.fire_geotiffs(ctx.work / "geotiffs")
    out = {"binary_mask": binary_mask(np.array([np.nan, np.inf, -np.inf, -1.0, 0.0, 0.5, 2.0]))}
    for length, interval, label_sel in ((6, 1, 0), (3, 2, 1), (6, 3, 0), (8, 1, 0)):
        windows = build_fire_windows("fire", local_map, length=length, interval=interval, label_sel=label_sel)
        # Captured as (inputs, targets), with (None, None) for a fire without windows.
        out[f"{length}.{interval}.{label_sel}"] = (
            (windows.inputs, windows.targets) if windows.count else (None, None)
        )
    return out


@group("tessera")
def tessera_values(ctx):
    """The offline TESSERA steps: crop grid, UTM zones, gridding, the saved npz."""
    import numpy as np
    import rasterio

    from firepred.data.tessera import crop_grid, grid_tessera, utm_zones
    from firepred.data.tessera_archive import save_tessera_npz

    local_map = synth.fire_geotiffs(ctx.work / "geotiffs", n_days=2)
    with rasterio.open(next(iter(local_map.values()))) as reader:
        profile = reader.profile
    crs, transform = crop_grid(profile)
    out = {"crop_grid": (crs.to_string(), tuple(transform))}
    out["utm_zones"] = [utm_zones(*bounds) for bounds in ((-123.5, -122.0), (-126.1, -125.9), (-120.0, -114.0))]
    embedding, coverage, source_pixels, bounds = grid_tessera(synth.FakeGeoTessera(), crs, transform, 2019, strip_rows=128)
    out["grid"] = (embedding, coverage, source_pixels, bounds)
    path = ctx.work / "tessera_out.npz"
    quiet(save_tessera_npz, path, embedding, coverage, {"fire_id": "x", "tessera_year": np.int16(2019)})
    out["saved"] = npz_members(path)
    return out


@group("roi")
def roi_values(ctx):
    """ROI CSV helpers on the real roi/ CSVs and the data-zip listing."""
    from firepred.data.fires import fire_label_sel, fire_start_years
    from firepred.data.files import zip_folders

    roi_paths = sorted(str(path) for path in (TREE / "roi").glob("*.csv"))
    out = {"fire_label_sel": fire_label_sel(roi_paths), "fire_start_years": fire_start_years(roi_paths)}
    root, _ = synth.data_root(ctx.work / "data", synth.QUICK_FIRES)
    zip_folders([root / "val", root / "test", root / "absent"], ctx.work / "bundle.zip")
    out["zip_listing"] = zip_listing(ctx.work / "bundle.zip")
    return out


# ----- the Hugging Face pipeline against an offline fake hub ----- #

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
        """Fake geotessera.GeoTesseraZarr with every year from 2017 available."""

        years = list(range(2017, 2025))

        def __init__(self, url=None):
            """Remember the store URL like the real client."""
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
    """Every file of a fake repo: .npz decoded, zips per member, text files as text."""
    import numpy as np

    root = fake_hub._repo(repo_id)
    out = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        name = str(path.relative_to(root))
        if path.suffix == ".npz":
            out[name] = npz_members(path, skip_members)
        elif path.suffix == ".zip":
            members = []
            with zipfile.ZipFile(path) as bundle:
                for info in bundle.infolist():
                    if info.filename.endswith(".npz"):
                        with np.load(io.BytesIO(bundle.read(info)), allow_pickle=False) as archive:
                            members.append((info.filename, [(k, archive[k]) for k in archive.files
                                                            if k not in skip_members]))
                    else:
                        members.append((info.filename, info.file_size, info.CRC))
            out[name] = members
        elif path.suffix == ".json":
            import json

            settings = json.loads(path.read_text())
            settings.pop("notes", None)  # code version and timestamps, different by design
            out[name] = settings
        elif path.suffix in (".txt", ".csv"):
            out[name] = path.read_text()
        else:
            out[name] = ("bytes", path.stat().st_size)
    return out


@group("hub")
def hub_values(ctx):
    """The whole dataset pipeline: windows, downloads, TESSERA, zips and pulls (files only)."""
    work = ctx.work
    fake_hub = install_fake_hub(work)
    from firepred.data import tessera_master, windows_master

    src, windows, tessera = "owner/src", "owner/windows-6-1", "owner/tessera"
    build_source_repo(fake_hub, src)
    out = {}
    quiet(
        windows_master.process_windows, "token", src_repo=src, window_repo=windows, batch_size=2, length=6,
        interval=1, splits=("train", "val", "test"), roi_dir=str(work / "roi1"), work_dir=str(work / "pw"),
    )
    out["process_data.repo"] = repo_contents(fake_hub, windows)

    report = quiet(
        windows_master.download_windows, "token", window_repo=windows, src_repo=src, data_root=str(work / "dl"),
        roi_dir=str(work / "roi2"),
    )
    out["download.files"] = files_under(work / "dl")
    out["download.excluded"] = list(report.excluded)

    out["tessera.skipped"] = quiet(
        tessera_master.process_tessera, "token", src_repo=src, tessera_repo=tessera, batch_size=1,
        work_dir=str(work / "pt"), roi_dir=str(work / "roi3"),
    ).skipped
    out["tessera_zip.missing"] = list(quiet(
        tessera_master.upload_tessera_zip, "token", src_repo=src, tessera_repo=tessera, work_dir=str(work / "tz"),
        roi_dir=str(work / "roi4"),
    ).missing)
    # code_version is the git commit of the tree, which differs by design.
    out["tessera.repo"] = repo_contents(fake_hub, tessera, skip_members=("code_version",))

    report = quiet(
        windows_master.download_windows, "token", window_repo=windows, src_repo=src, data_root=str(work / "dl_t"),
        roi_dir=str(work / "roi5"), include_tessera=True, tessera_repo=tessera,
    )
    out["download_tessera.files"] = files_under(work / "dl_t")
    out["download_tessera.excluded"] = list(report.excluded)

    quiet(windows_master.upload_data_zip, "token", data_root=str(work / "dl"), zip_path=str(work / "data.zip"),
          repo_id=windows)
    quiet(windows_master.pull_data_zip, str(work / "pulled"), token="token", repo_id=windows)
    out["pull_tessera.removed"] = list(quiet(
        tessera_master.pull_tessera_zip, str(work / "pulled"), token="token", tessera_repo=tessera
    ).removed)
    out["pull.files"] = files_under(work / "pulled")
    return out


# ----- models ----- #


@group("models")
def model_values(ctx):
    """Seeded init, train-mode output, loss, one Adam step and eval output of every model."""
    import torch
    from torch import nn, optim

    from firepred.models.registry import build_model, forward_model
    from firepred.training.config import LossConfig
    from firepred.training.objectives import build_criterion

    out = {}
    cases = [
        ("swinunetr3d", 43, 36, {}, 64),
        ("swinunetr3d", 172, 36, {}, 64),
        ("swinunetr3d", 43, 36, {"swin_temporal_downsample": True}, 64),
        ("swinunetr3d_tpool", 43, 12, {}, 64),
        ("swinunetr3d", 43, 12, {"attn_version": "v2"}, 64),
        ("unet3d", 43, 36, {}, 64),
        ("attunet", 43, 36, {}, 64),
        ("unetr3d", 43, 36, {"unetr_version": "v1"}, 256),
    ]
    for name, channels, width, options, size in cases:
        key = f"{name}.{channels}.{width}.{sorted(options.items())}"
        torch.manual_seed(42)
        try:
            model = nn.DataParallel(build_model(name, tiny_spec(channels, width, **options)))
        except Exception as error:  # a model that cannot be built is recorded, not skipped
            out[key + ".error"] = (type(error).__name__, str(error))
            continue
        out[key + ".init"] = model.state_dict()
        torch.manual_seed(5)
        x = torch.randn(2, channels, 6, size, size)
        y = torch.stack(((torch.rand(2, size, size) > 0.1), (torch.rand(2, size, size) > 0.9)), dim=1).float()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        outputs = forward_model(model, x)
        loss, parts = build_criterion(LossConfig("dicece"))(outputs, y)
        loss.backward()
        optimizer.step()
        out[key + ".train_out"] = outputs
        out[key + ".loss"] = (loss, parts)
        out[key + ".after_step"] = model.state_dict()
        out[key + ".optimizer"] = optimizer.state_dict()
        model.eval()
        with torch.no_grad():
            out[key + ".eval_out"] = forward_model(model, x)
    return out


# ----- losses, metrics, schedules, names ----- #


@group("objectives")
def objective_values(ctx):
    """Losses (with parts), per-window metrics, MONAI validation metrics, LR schedules, chart labels."""
    import torch
    from monai.data import decollate_batch
    from torch import nn, optim

    from firepred.training.config import LossConfig, ScheduleConfig
    from firepred.training.objectives import (
        background_f1, build_criterion, build_post_transform, fire_only_f1, fire_only_iou, validation_dice,
        validation_iou,
    )
    from firepred.training.tracker import chart_labels
    from firepred.training.trainer import build_lr_scheduler, step_lr_scheduler

    torch.manual_seed(11)
    logits = torch.randn(3, 2, 32, 32) * 3
    target_fire = torch.rand(3, 32, 32) > 0.95
    target_fire[2] = False
    labels = torch.stack((~target_fire, target_fire), dim=1).float()
    out = {}
    for loss_args in (
        ("dice",), ("dicece",), ("dicece", (1.0, 457.1), 0.7, 1.3),
        ("dicefocal",), ("dicefocal", (1.0, 446.7836), 1.0, 1.0, 2.0, 1.5, 0.25),
    ):
        criterion = build_criterion(LossConfig(*loss_args))
        out[f"loss.{loss_args}"] = criterion(logits, labels)
        out[f"loss_name.{loss_args}"] = criterion.name
    post = build_post_transform()
    predictions = torch.stack([post(item) for item in decollate_batch(logits)])
    out["post_transform"] = predictions
    predictions[2] = 0
    out["metric.fire_only_f1"] = fire_only_f1(predictions, labels)
    out["metric.background_f1"] = background_f1(predictions, labels)
    out["metric.fire_only_iou"] = fire_only_iou(predictions, labels)
    out["metric.mean_iou"] = validation_iou(predictions, labels)
    out["metric.dice"] = validation_dice(predictions, labels)

    for schedule in ("none", "plateau", "cosine"):
        parameter = nn.Parameter(torch.zeros(1))
        optimizer = optim.Adam([parameter], lr=1e-3)
        scheduler = build_lr_scheduler(optimizer, ScheduleConfig(schedule, 0.5, 1, 1e-4), 6)
        rates = []
        for val_loss in (1.0, 0.9, 0.95, 0.96, 0.97, 0.5):
            step_lr_scheduler(scheduler, val_loss)
            rates.append(optimizer.param_groups[0]["lr"])
        out[f"lr.{schedule}"] = (type(scheduler).__name__, rates)
    for loss in ("dice", "dicece", "dicefocal"):
        out[f"chart_labels.{loss}"] = chart_labels(loss)
    return out


NAME_ARGVS = [
    [],
    ["--loss", "dice"],
    ["--loss", "dicefocal", "--lr-schedule", "cosine"],
    ["--tessera", "--lr-schedule", "plateau"],
    ["-m", "attunet"],
    ["-m", "attunet", "--loss", "dicece", "-test", "--load-epoch", "12"],
    ["-m", "swinunetr3d", "--swin-temporal-downsample", "--model-option", "depth=2", "-ts", "10", "-nh", "4"],
    ["-mode", "other", "-b", "4", "-ed", "48", "--tessera", "--loss", "dice", "-test"],
]
NAME_BASE = ["-m", "swinunetr3d", "-mode", "pred", "-b", "8", "-r", "1", "-lr", "0.001", "-nh", "3", "-ed", "36",
             "-ts", "6", "-it", "1"]


@group("names")
def name_values(ctx):
    """Every derived name and recorded setting for a matrix of command lines (RunConfig)."""
    from firepred.cli.train import config_from_args, parse_args

    out = {}
    for argv in NAME_ARGVS:
        config = config_from_args(parse_args(NAME_BASE + argv))
        out[" ".join(argv) or "<base>"] = {
            "loss": config.loss.name,
            "loss_config": config.loss.record(),
            "lr_config": config.schedule.record(),
            "run_name": config.run_name,
            "run_tag": config.run_tag,
            "checkpoint_template": config.checkpoint_template,
            "test_checkpoint": config.test_checkpoint,
            "artifact_name": config.artifact_name("run123"),
            "model_options": config.model_options,
            "input_channels": config.input_channels,
        }
    return out


# ----- checkpoints ----- #


@group("checkpoints")
def checkpoint_values(ctx):
    """save -> read round trip (plain and with TESSERA stats) and best-of-N bookkeeping."""
    import numpy as np
    import torch
    from torch import nn, optim

    from firepred.training.checkpoints import CheckpointStore, read_checkpoint, save_checkpoint

    torch.manual_seed(3)
    model = nn.DataParallel(nn.Conv3d(4, 2, 1))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model(torch.randn(1, 4, 2, 4, 4)).sum().backward()
    optimizer.step()
    out = {}
    stats = {"mean": np.arange(128, dtype=np.float32), "std": np.ones(128, dtype=np.float32), "min_coverage": 0.5}
    for label, extra in (("plain", {"use_tessera": False}), ("tessera", {"use_tessera": True, "tessera_stats": stats})):
        path = ctx.work / f"{label}.pth"
        save_checkpoint(model, optimizer, 4, 0.25, str(path), extra=extra)
        for trusted in (False, True):
            try:
                out[f"roundtrip.{label}.{trusted}"] = read_checkpoint(str(path), "cpu", trusted=trusted)
            except RuntimeError as error:  # a restricted load may refuse numpy objects
                out[f"roundtrip.{label}.{trusted}.error"] = type(error).__name__
    store = CheckpointStore(ctx.work, "best_{epoch}.pth", {}, keep=2)
    for epoch, loss in enumerate((0.9, 0.7, 0.8, 0.95, 0.6)):
        store.offer(epoch - 1, loss, model, optimizer)  # file names use epoch + 1: best_0 .. best_4
    out["best.path"] = store.best.name
    out["best.kept"] = sorted(p.name for p in ctx.work.glob("best_*.pth"))
    return out


# ----- one epoch of training, validation and test scoring ----- #


@group("training")
def training_values(ctx):
    """One unshuffled, unaugmented training epoch, a validation pass and test scoring on a small SwinUNETR."""
    import torch
    from torch import nn

    from firepred.data.fire_dataset import FireDataset, make_dataloader
    from firepred.models.registry import build_model
    from firepred.training.config import RunConfig
    from firepred.training.evaluator import Evaluator
    from firepred.training.trainer import Trainer

    root, csv_path = synth.data_root(ctx.work / "data", synth.QUICK_FIRES)
    kwargs = dict(ts_length=6, n_channel=27, target_is_single_day=True, use_augmentations=False)
    train_ds = FireDataset(npz_dir=root / "train", **kwargs)
    val_ds = FireDataset(npz_dir=root / "val", **kwargs)
    test_ds = FireDataset(npz_dir=root / "test", **kwargs)
    device = torch.device("cpu")
    config = RunConfig("swinunetr3d", batch_size=2, feature_size=12, epochs=1, num_workers=0, test_csv=csv_path)
    torch.manual_seed(42)
    model = nn.DataParallel(build_model("swinunetr3d", tiny_spec(43, 12)))
    trainer = Trainer(config, model, device)

    def loader(dataset):
        """Unshuffled single-process loader with batch size 2."""
        return make_dataloader(dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)

    out = {}
    out["train_epoch"] = quiet(trainer.train_epoch, loader(train_ds), 0).as_dict()
    out["after_epoch"] = model.state_dict()
    out["validate"] = quiet(trainer.validate, loader(val_ds), 0).as_dict()
    plot_dir = ctx.work / "plots"
    out["evaluate"] = quiet(Evaluator(config, device, test_ds).score, model, plot_dir).as_dict()
    out["plots"] = files_under(plot_dir)
    return out


# ----- command-line interfaces ----- #


def describe_parser(parser):
    """Every option of an ArgumentParser (subcommands expanded) as plain rows."""
    import argparse

    rows = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for command, subparser in action.choices.items():
                rows.append(("subcommand", command, describe_parser(subparser)))
            continue
        rows.append((
            tuple(action.option_strings), action.dest, action.default, action.help, action.choices,
            action.nargs, getattr(action.type, "__name__", repr(action.type)), action.metavar, action.required,
        ))
    return rows


def capture_parser(call):
    """Run `call` and return (its result, the first ArgumentParser it parsed with)."""
    import argparse

    seen = []
    original = argparse.ArgumentParser.parse_args

    def spy(self, *args, **kwargs):
        """Record the parser, then parse as usual."""
        seen.append(self)
        return original(self, *args, **kwargs)

    argparse.ArgumentParser.parse_args = spy
    try:
        result = call()
    finally:
        argparse.ArgumentParser.parse_args = original
    return result, seen[0]


@group("cli")
def cli_values(ctx):
    """Parsed defaults and every option of the train, dataset and check commands."""
    from firepred.cli import check, dataset, train

    out = {}
    args, parser = capture_parser(lambda: train.parse_args(NAME_BASE))
    out["train.defaults"] = vars(args)
    out["train.options"] = describe_parser(parser)
    for command in ("process", "tessera", "tessera-zip", "pull-tessera", "download", "upload-zip", "pull-zip",
                    "rebuild-zip", "stage"):
        args, parser = capture_parser(lambda: dataset.parse_args([command]))
        out[f"dataset.{command}"] = vars(args)
    out["dataset.options"] = describe_parser(parser)
    _, parser = capture_parser(lambda: check.parse_args(["model", "-m", "swinunetr3d"]))
    out["check.options"] = describe_parser(parser)
    return out
