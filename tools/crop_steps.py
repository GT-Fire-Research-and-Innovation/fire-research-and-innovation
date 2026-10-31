"""Show every step from a fire's GeoTIFFs to model input, with the old FirePred crop beside the fixed one.

Used by the Colab notebook notebooks/firepred_crop_steps.ipynb:

    old = crop_steps.load_pipeline("old (model_crop_old)", "/content/code-old")
    fixed = crop_steps.load_pipeline("fixed (model_crop_fix)", "/content/code-fixed")
    local_map = crop_steps.download_fire_days("SamuelWu318/ts-satfire", "20901111", 7, "/content/raw", token)
    trace = crop_steps.trace_fire([old, fixed], "20901111", local_map, label_sel=0, show_day=5,
                                  seed=crop_steps.seed_with_every_change(), work_dir="/content/archives")
    crop_steps.plot_raw_bands(trace.days[fixed.name], trace.config)

Two checkouts of this repository are loaded side by side: one cropping FirePred
by pixel index (model_crop_old) and one resampling it onto the VIIRS grid when
the grids differ (model_crop_fix). Every step is recomputed here one stage at a
time so it can be shown, and each result is checked against the checkout's own
function (read_day_frame, build_fire_windows, prepare_window), so the pictures
show what the pipeline really produces. Nothing here changes the pipeline.
"""

import importlib
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# How each kind of channel is drawn: colour map, and how its range is chosen (see display_range).
COLOR_MAPS = {"value": "viridis", "angle": "twilight", "class": "tab20", "binary": "gray", "sin": "RdBu_r"}
MISSING_COLOR = "#d0d0d0"  # NaN pixels
SPARSE_MASK_PIXELS = 5000  # masks with at most this many positive pixels get a red dot on each
METERS_PER_DEGREE = 111_320.0
DAY_BAND_NAMES = ("VIIRS I1", "VIIRS I2", "VIIRS I3", "VIIRS I4", "VIIRS I5", "VIIRS M11",
                  "active fire (band 6, label)", "NIFC perimeter (band 7, label)")
NIGHT_BAND_NAMES = ("VIIRS I4 night", "VIIRS I5 night")
FIREPRED_MODULES = ("windows", "fire_dataset", "archive", "config")
MODULE_PATHS = {"windows": "firepred.data.windows", "fire_dataset": "firepred.data.fire_dataset",
                "archive": "firepred.data.archive", "config": "firepred.config.dataset"}


# ===== Code and data ===== #


@dataclass(frozen=True)
class Pipeline:
    """Hold the processing modules of one checkout of this repository.

    Two checkouts are loaded into one Python process, each with its own copy of
    the firepred modules. Calling pipeline.windows.read_day_frame therefore runs
    that checkout's code, whatever the other checkout says.

    Attributes:
        name (str): a label for figures, e.g. "old (model_crop_old)".
        root (Path): the checkout folder.
        windows (module): firepred.data.windows of the checkout.
        fire_dataset (module): firepred.data.fire_dataset of the checkout.
        archive (module): firepred.data.archive of the checkout.
        config (module): firepred.config.dataset of the checkout.
    """

    name: str
    root: Path
    windows: object
    fire_dataset: object
    archive: object
    config: object


def load_pipeline(name, repo_dir):
    """Import one checkout's firepred modules without letting them clash with another checkout's.

    Python keeps one module per name, so the firepred modules already imported
    are set aside, the checkout's own are imported, and the old ones are put
    back. The returned modules keep working afterwards: each holds its own
    globals and imports nothing from firepred later.

    Args:
        name (str): a label for figures, e.g. "fixed (model_crop_fix)".
        repo_dir (str | Path): the checkout folder (holding firepred/).

    Returns:
        Pipeline: the checkout's modules.

    Raises:
        ImportError: if a module was imported from somewhere other than repo_dir.
    """
    root = Path(repo_dir).resolve()

    def firepred_names():
        """List the firepred module names currently imported."""
        return [key for key in sys.modules if key == "firepred" or key.startswith("firepred.")]

    saved = {key: sys.modules.pop(key) for key in firepred_names()}
    sys.path.insert(0, str(root))
    try:
        modules = {short: importlib.import_module(MODULE_PATHS[short]) for short in FIREPRED_MODULES}
    finally:
        sys.path.remove(str(root))
        for key in firepred_names():
            sys.modules.pop(key)
        sys.modules.update(saved)
    for short, module in modules.items():
        if root not in Path(module.__file__).resolve().parents:
            raise ImportError(f"{short} came from {module.__file__}, not {root}")
    return Pipeline(name, root, **modules)


def read_token(name="HF_TOKEN"):
    """Read a Hugging Face token from Colab's secrets, else the environment; None if neither has it.

    Colab keeps secrets in google.colab.userdata (the key icon on the left), and
    the notebook must be granted access to each one. Outside Colab the
    environment variable of the same name is used. The source repo is public,
    so None still works, only with lower rate limits.

    Args:
        name (str): the secret's name.

    Returns:
        str | None: the token.
    """
    try:
        from google.colab import userdata

        return userdata.get(name)
    except Exception:  # not on Colab, or the secret is missing or not shared with this notebook
        return os.environ.get(name)


def download_fire_days(repo_id, fire_id, n_days, local_dir, token=None, first_day=0, folder="ts-satfire"):
    """Download n_days consecutive days of one fire's VIIRS_Day, VIIRS_Night and FirePred GeoTIFFs.

    The fire's files are listed under <folder>/<fire_id>/ in the dataset repo.
    The VIIRS_Day dates decide the days, and each day's night and FirePred files
    are the same path with "VIIRS_Day" replaced, as read_day_frame finds them.
    A day without a night file keeps going (the pipeline uses zeros); a day
    without FirePred is an error. Files already downloaded are reused.

    Args:
        repo_id (str): the Hugging Face dataset repo, e.g. "SamuelWu318/ts-satfire".
        fire_id (str): the fire's folder name.
        n_days (int): how many days to fetch (a window needs TS_LENGTH + 1).
        local_dir (str | Path): where the repo's folder layout is recreated.
        token (str | None): the Hugging Face token.
        first_day (int): index of the first VIIRS_Day date to fetch.
        folder (str): the repo folder holding one folder per fire.

    Returns:
        dict[str, str]: {repo path: local path}, the local_map build_fire_windows takes.

    Raises:
        ValueError: if the fire has fewer days than asked for, or a FirePred file is missing.
    """
    from huggingface_hub import HfApi, hf_hub_download

    tree = HfApi(token=token).list_repo_tree(repo_id, path_in_repo=f"{folder}/{fire_id}", recursive=True,
                                             repo_type="dataset")
    remote = {item.path for item in tree if item.path.endswith(".tif")}
    days = sorted(path for path in remote if "/VIIRS_Day/" in path)[first_day:first_day + n_days]
    if len(days) < n_days:
        raise ValueError(f"{fire_id} has {len(days)} VIIRS_Day files from day {first_day}; {n_days} needed")
    wanted = []
    for day in days:
        night, pred = day.replace("VIIRS_Day", "VIIRS_Night"), day.replace("VIIRS_Day", "FirePred")
        if pred not in remote:
            raise ValueError(f"{fire_id}: no FirePred file for {day}")
        wanted += [day, pred] + ([night] if night in remote else [])
    with ThreadPoolExecutor(max_workers=8) as pool:  # many small files: parallel requests are much faster
        local = pool.map(lambda path: hf_hub_download(repo_id, path, repo_type="dataset", token=token,
                                                      local_dir=str(local_dir)), wanted)
        return dict(zip(wanted, local))


def day_date(path):
    """Give a GeoTIFF's date, the part of its file name before the first underscore."""
    return os.path.basename(path).split("_")[0]


# ===== Grids ===== #


def crop_box(transform, offset, size):
    """Give the (left, right, bottom, top) ground box of rows/cols offset to offset + size of a raster."""
    left, top = transform * (offset, offset)
    right, bottom = transform * (offset + size, offset + size)
    return min(left, right), max(left, right), min(bottom, top), max(bottom, top)


def raster_extent(profile):
    """Give a raster's (left, right, bottom, top) ground extent, for imshow."""
    transform = profile["transform"]
    left, right = sorted((transform.c, transform.c + transform.a * profile["width"]))
    bottom, top = sorted((transform.f, transform.f + transform.e * profile["height"]))
    return left, right, bottom, top


def ground_distance_km(a, b):
    """Give the distance in km between two (lon, lat) points close together (flat-earth approximation)."""
    latitude = math.radians((a[1] + b[1]) / 2)
    return math.hypot((a[0] - b[0]) * math.cos(latitude), a[1] - b[1]) * METERS_PER_DEGREE / 1000


def crop_offset_km(day_profile, pred_profile, offset, size):
    """Measure how far FirePred's index crop centre lies from the VIIRS crop centre.

    Both crops take rows/cols offset .. offset + size of their own raster. On a
    shared grid they cover the same ground and the distance is 0; on a coarser
    FirePred grid the same rows/cols reach further from the origin.

    Args:
        day_profile (dict): rasterio profile of the VIIRS_Day file.
        pred_profile (dict): rasterio profile of the FirePred file.
        offset (int): first row/column of the crop.
        size (int): crop size in pixels.

    Returns:
        float: the distance in km.
    """
    centre = offset + size / 2
    return ground_distance_km(day_profile["transform"] * (centre, centre), pred_profile["transform"] * (centre, centre))


def describe_grids(fixed, local_maps):
    """Compare every downloaded day's FirePred grid with its VIIRS_Day grid, using the fixed pipeline's test.

    The fixed pipeline decides per file, on the fly, whether FirePred must be
    resampled (same_grid). This lists that decision with the pixel sizes,
    raster sizes and how far the old index crop lands from the VIIRS crop.

    Args:
        fixed (Pipeline): the fixed pipeline (it has windows.same_grid).
        local_maps (dict[str, dict[str, str]]): {fire id: local_map}.

    Returns:
        list[dict]: one row per fire and day.
    """
    import rasterio

    rows = []
    offset, size = fixed.config.CROP_OFFSET, fixed.config.CROP_SIZE
    for fire_id, local_map in local_maps.items():
        for day_file in sorted(path for remote, path in local_map.items() if "VIIRS_Day" in remote):
            with rasterio.open(day_file) as day, rasterio.open(day_file.replace("VIIRS_Day", "FirePred")) as pred:
                day_profile, pred_profile = day.profile, pred.profile
            same = fixed.windows.same_grid(pred_profile, day_profile)
            rows.append({
                "fire": fire_id, "date": day_date(day_file),
                "VIIRS pixel (m)": round(abs(day_profile["transform"].e) * METERS_PER_DEGREE),
                "FirePred pixel (m)": round(abs(pred_profile["transform"].e) * METERS_PER_DEGREE),
                "VIIRS size": (day_profile["height"], day_profile["width"]),
                "FirePred size": (pred_profile["height"], pred_profile["width"]),
                "same grid": same,
                "fixed pipeline": "index crop" if same else "resample onto VIIRS crop",
                "old crop off by (km)": round(crop_offset_km(day_profile, pred_profile, offset, size), 1),
            })
    return rows


# ===== Tracing the pipeline one step at a time ===== #


@dataclass(frozen=True)
class DayTrace:
    """Hold one day's GeoTIFFs and every stage read_day_frame takes them through.

    The raw arrays are the files as stored. The stack is the 27 cropped
    channels before missing values are filled, so NaN still shows; the image is
    the same after NaN/infinity become 0, which is exactly read_day_frame's
    output (checked when traced).

    Attributes:
        date (str): the day.
        day (np.ndarray): VIIRS_Day bands, (bands, H, W).
        day_profile (dict): its rasterio profile.
        night (np.ndarray | None): VIIRS_Night bands as stored, or None without a file.
        night_profile (dict | None): its rasterio profile.
        pred (np.ndarray): FirePred bands, (19, h, w) on FirePred's own grid.
        pred_profile (dict): its rasterio profile.
        placement (str): how this pipeline put FirePred on the crop.
        stack (np.ndarray): (27, 256, 256) cropped channels, NaN kept.
        image (np.ndarray): (27, 256, 256) cropped channels, NaN/infinity set to 0.
        af (np.ndarray): (256, 256) bool active-fire mask of the day.
        ba (np.ndarray): (256, 256) bool NIFC perimeter mask of the day.
    """

    date: str
    day: np.ndarray
    day_profile: dict
    night: object
    night_profile: object
    pred: np.ndarray
    pred_profile: dict
    placement: str
    stack: np.ndarray
    image: np.ndarray
    af: np.ndarray
    ba: np.ndarray


def check_equal(name, ours, pipeline_value):
    """Raise unless a traced stage equals the pipeline's own output exactly (NaN equal to NaN)."""
    ours = ours.numpy() if hasattr(ours, "numpy") else np.asarray(ours)
    theirs = pipeline_value.numpy() if hasattr(pipeline_value, "numpy") else np.asarray(pipeline_value)
    if ours.shape != theirs.shape or not np.array_equal(ours, theirs, equal_nan=ours.dtype.kind == "f"):
        raise RuntimeError(f"traced {name} differs from the pipeline's own output")
    return True


def trace_day(pipeline, day_file):
    """Read one day the way the pipeline's read_day_frame does, keeping every stage.

    The night file is used as read_day_frame uses it (5-band files keep their
    last two bands, unusable or missing ones become zeros). FirePred is cropped
    by index unless the pipeline has same_grid and it says the grids differ;
    then it is resampled onto the VIIRS crop. The result is checked against
    read_day_frame itself.

    Args:
        pipeline (Pipeline): the checkout whose behaviour to trace.
        day_file (str): the local VIIRS_Day GeoTIFF.

    Returns:
        DayTrace: the raw arrays and each stage.
    """
    windows, config = pipeline.windows, pipeline.config
    offset, size = config.CROP_OFFSET, config.CROP_SIZE
    crop = slice(offset, offset + size)
    day, day_profile = windows.read_tiff(day_file)
    night_file = day_file.replace("VIIRS_Day", "VIIRS_Night")
    night, night_profile = windows.read_tiff(night_file) if os.path.exists(night_file) else (None, None)
    night_used = night[3:] if night is not None and night.shape[0] == 5 else night
    if night_used is None or night_used.shape[0] < 2:
        night_used = np.zeros((2, *day.shape[1:]), dtype=np.float32)
    pred, pred_profile = windows.read_tiff(day_file.replace("VIIRS_Day", "FirePred"))

    day_crop = day[:6, crop, crop]
    if hasattr(windows, "same_grid") and not windows.same_grid(pred_profile, day_profile):
        pred_crop = windows.resample_to_crop(pred, pred_profile, day_profile, offset, day_crop.shape[1:])
        placement = "resampled onto the VIIRS crop (grids differ)"
    else:
        pred_crop = pred[:, crop, crop]
        placement = f"cropped by index, rows/cols {offset}:{offset + size}"
    stack = np.concatenate((day_crop, night_used[:, crop, crop], pred_crop), axis=0)[:, :size, :size]
    image = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)
    af = windows.binary_mask(day[6, crop, crop])
    ba = windows.binary_mask(day[7, crop, crop]) if day.shape[0] >= 8 else np.zeros((size, size), dtype=bool)

    frame = windows.read_day_frame(day_file)
    check_equal(f"{pipeline.name} {day_date(day_file)} image", image, frame.image)
    check_equal(f"{pipeline.name} {day_date(day_file)} active fire", af, frame.af)
    check_equal(f"{pipeline.name} {day_date(day_file)} NIFC", ba, frame.ba)
    return DayTrace(day_date(day_file), day, day_profile, night, night_profile, pred, pred_profile, placement,
                    stack, image, af, ba)


@dataclass(frozen=True)
class WindowTrace:
    """Hold how build_fire_windows turns length + 1 days into the first window and its target.

    Each day's frame is kept as read, then the heat channels (3, 4, 6, 7) are
    replaced by their running maximum. Burned area accumulates day by day under
    both label rules; the target is what the chosen rule adds on the target day.

    Attributes:
        dates (tuple[str, ...]): the length input days, then the target day.
        frames (np.ndarray): (length + 1, 27, 256, 256) each day's image as read.
        stored (np.ndarray): (27, length, 256, 256) the window as stored (running max applied).
        af (np.ndarray): (length + 1, 256, 256) each day's active-fire mask.
        ba (np.ndarray): (length + 1, 256, 256) each day's NIFC mask.
        burned (dict[int, np.ndarray]): {label_sel: (length + 1, 256, 256) burned so far}.
        targets (dict[int, np.ndarray]): {label_sel: (256, 256) newly burned on the target day}.
        label_sel (int): the rule the window's stored target uses.
    """

    dates: tuple
    frames: np.ndarray
    stored: np.ndarray
    af: np.ndarray
    ba: np.ndarray
    burned: dict
    targets: dict
    label_sel: int

    @property
    def target(self):
        """Give the stored target, the one of the window's label rule."""
        return self.targets[self.label_sel]


def trace_window(pipeline, fire_id, local_map, label_sel, length):
    """Build the first window step by step, as build_fire_windows does, and check it against that function.

    Days 0 .. length - 1 are the input and day length is the target. The heat
    channels keep the running maximum over the days so far; burned area
    accumulates and the target is the newly burned pixels. Later windows only
    repeat this from a later start day, so the first one shows every step.

    Args:
        pipeline (Pipeline): the checkout whose behaviour to trace.
        fire_id (str): the fire, for messages.
        local_map (dict[str, str]): {repo path: local path} of the fire's GeoTIFFs.
        label_sel (int): 1 = active fire only, 0 = active fire plus NIFC perimeters.
        length (int): input days per window.

    Returns:
        tuple[WindowTrace, list[DayTrace]]: the window and each day's trace.
    """
    day_files = sorted(path for remote, path in local_map.items() if "VIIRS_Day" in remote)[:length + 1]
    days = [trace_day(pipeline, path) for path in day_files]
    frames = np.stack([day.image for day in days])
    running_max = np.maximum.accumulate(np.maximum(frames, 0), axis=0)  # the pipeline's maximum starts at zeros
    stored = np.concatenate((frames[:, :3], running_max[:, 3:5], frames[:, 5:6], running_max[:, 6:8], frames[:, 8:]),
                            axis=1)[:length].transpose(1, 0, 2, 3)
    af, ba = np.stack([day.af for day in days]), np.stack([day.ba for day in days])
    burned = {1: np.logical_or.accumulate(af, axis=0), 0: np.logical_or.accumulate(af | ba, axis=0)}
    targets = {rule: (mask[length] & ~mask[length - 1]).astype(np.float32) for rule, mask in burned.items()}

    built = pipeline.windows.build_fire_windows(fire_id, local_map, length=length, interval=1, label_sel=label_sel)
    check_equal(f"{pipeline.name} {fire_id} window", stored, built.inputs[0])
    check_equal(f"{pipeline.name} {fire_id} target", targets[label_sel], built.targets[0])
    trace = WindowTrace(tuple(day.date for day in days), frames, stored, af, ba, burned, targets, label_sel)
    return trace, days


@dataclass(frozen=True)
class LoaderTrace:
    """Hold what training does to one stored window: archive round trip, normalize, augment, encode.

    The window is written as a p<id>.npz archive and read back with read_window,
    as training reads it. Then the released order follows: normalize, augment
    (training only), preprocess. Both the training item (augmented) and the
    validation/test item (not augmented) are kept, each checked against
    prepare_window.

    Attributes:
        x (torch.Tensor): (27, T, 256, 256) read back from the archive.
        y (torch.Tensor): (2, 256, 256) target as two channels (background, newly burned).
        normalized (torch.Tensor): (27, T, 256, 256) after normalize.
        augmented (torch.Tensor): (27, T, 256, 256) after augment.
        augmented_y (torch.Tensor): (2, 256, 256) target after augment.
        encoded_train (torch.Tensor): (43, T, 256, 256) model input of a training item.
        encoded_eval (torch.Tensor): (43, T, 256, 256) model input of a validation/test item.
        augmentation (str): which flips and rotation were drawn.
    """

    x: object
    y: object
    normalized: object
    augmented: object
    augmented_y: object
    encoded_train: object
    encoded_eval: object
    augmentation: str


def augment_draws(seed):
    """Give the (horizontal flip, vertical flip, quarter turns) augment draws from a seeded generator.

    augment draws three random numbers in this order. Drawing them from a
    generator with the same seed tells which changes it will make.

    Args:
        seed (int): the generator's seed.

    Returns:
        tuple[bool, bool, int]: the three decisions.
    """
    rng = np.random.default_rng(seed)
    return bool(rng.random() > 0.5), bool(rng.random() > 0.5), int(np.floor(rng.random() * 4))


def seed_with_every_change(start=0):
    """Find the first seed from start whose augmentation flips both ways and rotates, so every change shows."""
    seed = start
    while not (augment_draws(seed)[0] and augment_draws(seed)[1] and augment_draws(seed)[2]):
        seed += 1
    return seed


def trace_loader(pipeline, fire_id, window, label_sel, seed, work_dir):
    """Run one stored window through the training loader's steps, checking each against the pipeline.

    The archive is written with the pipeline's write_fire_archive and read with
    its read_window. normalize uses the paper's statistics (FireDataset's
    default without a window_config.json). augment uses a generator seeded with
    seed, so the drawn flips and rotation can be reported.

    Args:
        pipeline (Pipeline): the checkout whose behaviour to trace.
        fire_id (str): the fire, for the archive name.
        window (WindowTrace): the traced window.
        label_sel (int): the label rule recorded in the archive.
        seed (int): the augmentation seed.
        work_dir (str | Path): where the archive is written.

    Returns:
        LoaderTrace: every stage.
    """
    import torch

    dataset, config = pipeline.fire_dataset, pipeline.config
    folder = Path(work_dir) / pipeline.name.split()[0]
    folder.mkdir(parents=True, exist_ok=True)
    path = pipeline.archive.write_fire_archive(folder / f"p{fire_id}.npz", window.stored[None], window.target[None],
                                               label_sel=label_sel)
    x, y = dataset.read_window(path, 0, config.N_STORED_CHANNELS, window.stored.shape[1], True)
    check_equal(f"{pipeline.name} archive round trip", x, torch.from_numpy(window.stored))

    degrees = list(config.DEGREE_CHANNELS)
    one_hot = dataset.landcover_one_hot_matrix()
    normalized = dataset.normalize(x, config.NORMALIZATION_MEAN, config.NORMALIZATION_STD, skip=dataset.NORMALIZE_SKIP)
    augmented, augmented_y, _ = dataset.augment(normalized.clone(), y.clone(), None, degrees,
                                                rng=np.random.default_rng(seed))
    encoded_train = dataset.preprocess(augmented, one_hot, degrees)
    encoded_eval = dataset.preprocess(normalized, one_hot, degrees)

    train_item = dataset.prepare_window(x, y, one_hot, train=True, rng=np.random.default_rng(seed))
    eval_item = dataset.prepare_window(x, y, one_hot, train=False)
    check_equal(f"{pipeline.name} training item", encoded_train, train_item["data"])
    check_equal(f"{pipeline.name} training labels", augmented_y, train_item["labels"])
    check_equal(f"{pipeline.name} evaluation item", encoded_eval, eval_item["data"])
    hflip, vflip, turns = augment_draws(seed)
    described = f"seed {seed}: horizontal flip {hflip}, vertical flip {vflip}, rotation {turns * 90} degrees"
    return LoaderTrace(x, y, normalized, augmented, augmented_y, encoded_train, encoded_eval, described)


@dataclass(frozen=True)
class FireTrace:
    """Hold one fire's traces for every pipeline, keyed by pipeline name.

    Each pipeline reads the same downloaded files. Comparing the traces shows
    which stages the FirePred fix changes for this fire.

    Attributes:
        fire_id (str): the fire.
        config (module): the dataset settings (the same in both checkouts).
        show_day (int): the input day the per-channel figures show.
        days (dict[str, DayTrace]): {pipeline: trace of the shown day}.
        windows (dict[str, WindowTrace]): {pipeline: first window}.
        loaders (dict[str, LoaderTrace]): {pipeline: loader stages}.
    """

    fire_id: str
    config: object
    show_day: int
    days: dict
    windows: dict
    loaders: dict


def trace_fire(pipelines, fire_id, local_map, label_sel, show_day, seed, work_dir):
    """Trace one fire through every pipeline: the day, the first window and the loader.

    Every pipeline reads the same downloaded files, one after another. Every
    stage is checked against the pipeline's own function while tracing, so a
    figure never shows a stage the pipeline does not produce.

    Args:
        pipelines (list[Pipeline]): the checkouts to compare (old first).
        fire_id (str): the fire.
        local_map (dict[str, str]): {repo path: local path} of its downloaded GeoTIFFs.
        label_sel (int): the label rule for the window's target.
        show_day (int): the input day (0 .. length - 1) to show per channel.
        seed (int): the augmentation seed.
        work_dir (str | Path): where archives are written.

    Returns:
        FireTrace: the traces.
    """
    config = pipelines[-1].config
    days, windows, loaders = {}, {}, {}
    for pipeline in pipelines:
        window, day_traces = trace_window(pipeline, fire_id, local_map, label_sel, config.TS_LENGTH)
        days[pipeline.name] = day_traces[show_day]
        windows[pipeline.name] = window
        loaders[pipeline.name] = trace_loader(pipeline, fire_id, window, label_sel, seed, work_dir)
    return FireTrace(fire_id, config, show_day, days, windows, loaders)


def stage_differences(trace):
    """Say, step by step, which stored channels differ between the first and last pipeline for one fire.

    The FirePred fix should only change FirePred channels (8-26), and only for
    fires whose FirePred grid differs from VIIRS. Every other channel and every
    label must be identical, which this table makes easy to check.

    Args:
        trace (FireTrace): the fire's traces (old pipeline first, fixed last).

    Returns:
        list[dict]: one row per step with the channels that differ ("none" when identical).
    """
    old, fixed = list(trace.days)[0], list(trace.days)[-1]

    def changed(a, b):
        """List the leading-axis indices where two stacks differ, as text."""
        a, b = as_numpy(a), as_numpy(b)
        indices = [str(c) for c in range(a.shape[0]) if not same(a[c], b[c])]
        return ", ".join(indices) if indices else "none"

    day, window, loader = trace.days, trace.windows, trace.loaders
    steps = [
        ("crop (NaN kept)", day[old].stack, day[fixed].stack),
        ("NaN/Inf set to 0 (read_day_frame)", day[old].image, day[fixed].image),
        ("labels: active fire, NIFC", np.stack([day[old].af, day[old].ba]), np.stack([day[fixed].af, day[fixed].ba])),
        ("stored window (build_fire_windows)", window[old].stored, window[fixed].stored),
        ("target", window[old].target[None], window[fixed].target[None]),
        ("normalize", loader[old].normalized, loader[fixed].normalized),
        ("augment (training)", loader[old].augmented, loader[fixed].augmented),
        ("preprocess, training item (43 ch)", loader[old].encoded_train, loader[fixed].encoded_train),
        ("preprocess, eval item (43 ch)", loader[old].encoded_eval, loader[fixed].encoded_eval),
    ]
    return [{"fire": trace.fire_id, "step": step, "channels that differ": changed(a, b)} for step, a, b in steps]


# ===== Channel names and kinds ===== #


def stored_layout(config):
    """Give the 27 stored channels' names and kinds (value, angle, class).

    Angles are drawn on a cyclic colour map and land cover as classes. Every
    other channel is a plain value.

    Args:
        config (module): the dataset settings.

    Returns:
        tuple[list[str], list[str]]: names and kinds.
    """
    kinds = ["angle" if c in config.DEGREE_CHANNELS else "class" if c == config.LANDCOVER_CHANNEL else "value"
             for c in range(config.N_STORED_CHANNELS)]
    return list(config.STORED_CHANNEL_NAMES), kinds


def encoded_layout(config):
    """Give the 43 model-input channels' names and kinds after preprocess.

    Angles become sin(angle). Land cover becomes 17 one-hot planes, plane k for
    class k + 1 (class 0, missing, is all zeros).

    Args:
        config (module): the dataset settings.

    Returns:
        tuple[list[str], list[str]]: names and kinds.
    """
    names, kinds = [], []
    for channel, name in enumerate(config.STORED_CHANNEL_NAMES):
        if channel == config.LANDCOVER_CHANNEL:
            names += [f"land cover = class {k}" for k in range(1, config.N_LANDCOVER_CLASSES + 1)]
            kinds += ["binary"] * config.N_LANDCOVER_CLASSES
        elif channel in config.DEGREE_CHANNELS:
            names.append(f"sin({name})")
            kinds.append("sin")
        else:
            names.append(name)
            kinds.append("value")
    return names, kinds


# ===== Drawing ===== #


def as_numpy(array):
    """Give a numpy view of a numpy array or a torch tensor."""
    return array.numpy() if hasattr(array, "numpy") else np.asarray(array)


def display_range(arrays, kind):
    """Choose one colour range for a channel shown in several arrays, so they compare by eye.

    Values use the 1st to 99th percentile of the finite pixels of all arrays,
    so a few extreme pixels do not wash out the rest. Angles keep their full
    circle: 0..360, or -90..90 for forecast wind direction. Classes, masks and
    sines have fixed ranges.

    Args:
        arrays (list[np.ndarray]): the same channel from every array shown.
        kind (str): "value", "angle", "class", "binary" or "sin".

    Returns:
        tuple[float, float]: (vmin, vmax).
    """
    fixed = {"class": (0.0, 17.0), "binary": (0.0, 1.0), "sin": (-1.0, 1.0)}
    if kind in fixed:
        return fixed[kind]
    values = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    if values.size == 0:
        return 0.0, 1.0
    if kind == "angle":
        return (-90.0, 90.0) if values.min() < 0 and values.max() <= 90.5 else (0.0, 360.0)
    low, high = np.percentile(values, [1, 99])
    return float(low), float(high if high > low else low + 1)


def color_map(kind):
    """Give the colour map of a channel kind, with NaN drawn grey."""
    return matplotlib.colormaps[COLOR_MAPS[kind]].with_extremes(bad=MISSING_COLOR)


def panel_grid(count, columns, size, title):
    """Make a figure with count panels in rows of columns, unused panels hidden.

    Every panel is size inches square, and the figure title sits above all of
    them. Axis ticks are hidden: pixel positions are not what the figures show.

    Args:
        count (int): panels needed.
        columns (int): panels per row.
        size (float): panel side in inches.
        title (str): the figure title.

    Returns:
        tuple[matplotlib.figure.Figure, list]: the figure and its first count axes.
    """
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(columns * size, rows * size * 1.08), squeeze=False)
    axes = list(axes.ravel())
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[count:]:
        ax.axis("off")
    figure.suptitle(title, fontsize=13, y=1.0)
    return figure, axes[:count]


def draw(ax, array, kind, value_range, title, extent=None, changed=False):
    """Draw one channel in a panel with its colour bar; a red title marks a channel that differs from the first version.

    Missing (NaN) pixels are grey, and the share of them is added to the title.
    Sparse masks (fire pixels) also mark every positive pixel with a red dot,
    so single pixels stay visible at panel size.

    Args:
        ax (matplotlib.axes.Axes): the panel.
        array (np.ndarray): (H, W) values.
        kind (str): the channel kind.
        value_range (tuple[float, float]): (vmin, vmax).
        title (str): the panel title.
        extent (tuple | None): (left, right, bottom, top) ground extent, or None for pixels.
        changed (bool): colour the title red.

    Returns:
        matplotlib.image.AxesImage: the drawn image.
    """
    missing = float(np.mean(~np.isfinite(array))) if array.dtype.kind == "f" else 0.0
    image = ax.imshow(array, cmap=color_map(kind), vmin=value_range[0], vmax=value_range[1],
                      interpolation="nearest", extent=extent)
    if kind == "binary" and 0 < np.count_nonzero(array > 0) <= SPARSE_MASK_PIXELS:
        rows, cols = np.nonzero(array > 0)
        ax.scatter(cols, rows, s=3, c="red", linewidths=0)
    label = title + (f"\n{missing:.0%} NaN" if missing else "")
    ax.set_title(label, fontsize=8, color="red" if changed else "black")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
    return image


def same(a, b):
    """Tell whether two arrays are identical, NaN equal to NaN."""
    a, b = as_numpy(a), as_numpy(b)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")


def plot_channels(stacks, names, kinds, title, columns=7, size=2.3):
    """Draw every channel of one or more versions of a stack, one figure per version.

    Each channel uses one colour range across all versions, so they compare by
    eye. A version identical to the first is not drawn again, only reported.
    In the other versions, a channel that differs from the first has a red title.

    Args:
        stacks (dict[str, np.ndarray | torch.Tensor]): {version name: (C, H, W)}.
        names (list[str]): the C channel names.
        kinds (list[str]): the C channel kinds.
        title (str): what step the stack comes from.
        columns (int): panels per row.
        size (float): panel side in inches.

    Returns:
        list[matplotlib.figure.Figure]: the figures drawn.
    """
    stacks = {label: as_numpy(stack) for label, stack in stacks.items()}
    labels = list(stacks)
    first = stacks[labels[0]]
    ranges = [display_range([stack[c] for stack in stacks.values()], kinds[c]) for c in range(len(names))]
    figures = []
    for label in labels:
        stack = stacks[label]
        if label != labels[0] and same(stack, first):
            print(f"{title} | {label}: identical to {labels[0]}, not drawn again")
            continue
        changed = [label != labels[0] and not same(stack[c], first[c]) for c in range(len(names))]
        suffix = f" ({sum(changed)} channels differ from {labels[0]}, red)" if any(changed) else ""
        figure, axes = panel_grid(len(names), columns, size, f"{title} | {label}{suffix}")
        for c, ax in enumerate(axes):
            draw(ax, stack[c], kinds[c], ranges[c], f"{c}: {names[c]}", changed=changed[c])
        figure.tight_layout()
        plt.show()
        figures.append(figure)
    return figures


def plot_raw_bands(day, config, columns=6, size=2.6):
    """Draw every band of one day's three GeoTIFFs on the ground, with the crop boxes.

    Each raster is drawn at its own ground extent (longitude, latitude), so
    rasters on different grids line up. The green box is the ground the VIIRS
    crop covers (rows/cols of the VIIRS grid). On FirePred bands the dashed red
    box is FirePred's own rows/cols at the same indices, which the old pipeline
    read; when the grids match the two boxes coincide.

    Args:
        day (DayTrace): the day (raw arrays are the same in every pipeline).
        config (module): the dataset settings.
        columns (int): panels per row.
        size (float): panel side in inches.

    Returns:
        matplotlib.figure.Figure: the figure.
    """
    offset, size_px = config.CROP_OFFSET, config.CROP_SIZE
    stored_names, stored_kinds = stored_layout(config)
    first = config.FIREPRED_FIRST_CHANNEL if hasattr(config, "FIREPRED_FIRST_CHANNEL") else 8
    panels = [(f"VIIRS_Day {b}: {DAY_BAND_NAMES[b]}", day.day[b], day.day_profile, "value")
              for b in range(day.day.shape[0])]
    if day.night is None:
        panels += [(f"VIIRS_Night: {name}\n(no file: zeros)", None, None, "value") for name in NIGHT_BAND_NAMES]
    else:
        night = day.night[3:] if day.night.shape[0] == 5 else day.night
        panels += [(f"VIIRS_Night {b}: {NIGHT_BAND_NAMES[b] if b < 2 else ''}", night[b], day.night_profile, "value")
                   for b in range(night.shape[0])]
    panels += [(f"FirePred {b}: {stored_names[first + b]}", day.pred[b], day.pred_profile, stored_kinds[first + b])
               for b in range(day.pred.shape[0])]

    viirs_box = crop_box(day.day_profile["transform"], offset, size_px)
    index_box = crop_box(day.pred_profile["transform"], offset, size_px)
    off_km = crop_offset_km(day.day_profile, day.pred_profile, offset, size_px)
    title = (f"{day.date}: every band before cropping. Green: ground of the VIIRS crop (rows/cols {offset}:"
             f"{offset + size_px}). Red dashed: FirePred rows/cols {offset}:{offset + size_px} (old crop), "
             f"centre {off_km:.1f} km away")
    figure, axes = panel_grid(len(panels), columns, size, title)
    latitude = math.radians((viirs_box[2] + viirs_box[3]) / 2)
    for ax, (name, band, profile, kind) in zip(axes, panels):
        if band is None:
            ax.set_title(name, fontsize=8)
            continue
        draw(ax, band.astype(np.float32), kind, display_range([band.astype(np.float32)], kind), name,
             extent=raster_extent(profile))
        boxes = [(viirs_box, "lime", "-")] + ([(index_box, "red", "--")] if profile is day.pred_profile else [])
        for (left, right, bottom, top), color, style in boxes:
            ax.add_patch(Rectangle((left, bottom), right - left, top - bottom, fill=False, edgecolor=color,
                                   linestyle=style, linewidth=1.4))
        ax.set_aspect(1 / math.cos(latitude))
    figure.tight_layout()
    plt.show()
    return figure


def ndvi_agreement(image):
    """Correlate NDVI computed from VIIRS I1/I2 with FirePred's NDVI channel on clear pixels; NaN if too few.

    Both measure the same vegetation, so on matching ground they correlate.
    Bright (I1 >= 40, cloud) and empty pixels are left out.

    Args:
        image (np.ndarray): (27, H, W) a day's cropped channels.

    Returns:
        float: Pearson r.
    """
    i1, i2, ndvi = image[0].astype(float), image[1].astype(float), image[8].astype(float)
    clear = (i1 > 0) & (i2 > 0) & (i1 < 40) & (ndvi != 0)
    if clear.sum() < 500:
        return float("nan")
    return float(np.corrcoef(((i2 - i1) / (i2 + i1))[clear], ndvi[clear])[0, 1])


def plot_alignment(window_traces, size=3.2):
    """Check by eye and by number that FirePred covers the same ground as VIIRS, for every pipeline.

    The first panel is NDVI computed from VIIRS I1/I2 on the shown day; the
    others are FirePred's NDVI channel of each pipeline. On the right ground the
    same fields, valleys and burn scars show in both, and r is high. Below,
    r for every day of the window.

    Args:
        window_traces (dict[str, WindowTrace]): {pipeline: first window}.
        size (float): panel side in inches.

    Returns:
        dict[str, list[float]]: {pipeline: NDVI r per day}.
    """
    labels = list(window_traces)
    frames = window_traces[labels[-1]].frames
    day = frames.shape[0] - 2  # the last input day
    i1, i2 = frames[day, 0].astype(float), frames[day, 1].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        viirs = np.where((i1 > 0) & (i2 > 0) & (i1 < 40), (i2 - i1) / (i2 + i1), np.nan)
    agreement = {label: [ndvi_agreement(frame) for frame in trace.frames] for label, trace in window_traces.items()}
    figure, axes = panel_grid(1 + len(labels), 1 + len(labels), size,
                              f"NDVI check, {window_traces[labels[-1]].dates[day]}: VIIRS vs FirePred")
    draw(axes[0], viirs, "value", display_range([viirs], "value"), "NDVI from VIIRS I1, I2 (clear pixels)")
    ndvi_range = display_range([trace.frames[day, 8] for trace in window_traces.values()], "value")
    for ax, label in zip(axes[1:], labels):
        draw(ax, window_traces[label].frames[day, 8], "value", ndvi_range,
             f"FirePred NDVI, {label}\nr = {agreement[label][day]:.2f}")
    figure.tight_layout()
    plt.show()
    for label, values in agreement.items():
        print(f"{label:28s} NDVI r per day: " + ", ".join("nan" if np.isnan(v) else f"{v:.2f}" for v in values))
    return agreement


def plot_labels(window, size=2.0):
    """Draw how the target is made: each day's masks, the burned area so far under both rules, and the target.

    Rows: the day's active-fire detections, the day's NIFC perimeter, burned so
    far with label_sel 1 (active fire only) and with label_sel 0 (plus NIFC).
    Columns: the input days, then the target day. The last row is each rule's
    target, the pixels burned on the target day but not by the last input day.

    Args:
        window (WindowTrace): the first window.
        size (float): panel side in inches.

    Returns:
        matplotlib.figure.Figure: the figure.
    """
    days = len(window.dates)
    rows = [("active fire (day)", window.af), ("NIFC perimeter (day)", window.ba),
            ("burned so far, label_sel 1", window.burned[1]), ("burned so far, label_sel 0", window.burned[0])]
    figure, axes = plt.subplots(len(rows) + 1, days, figsize=(days * size, (len(rows) + 1) * size * 1.1),
                                squeeze=False)
    for r, (name, masks) in enumerate(rows):
        for d in range(days):
            ax = axes[r, d]
            role = "target day" if d == days - 1 else f"input {d}"
            draw(ax, masks[d].astype(np.float32), "binary", (0, 1),
                 f"{name}\n{window.dates[d]} ({role}): {int(masks[d].sum())} px")
    for d in range(days):
        axes[-1, d].axis("off")
    for k, rule in enumerate((1, 0)):
        ax = axes[-1, days - 1 - k]
        ax.axis("on")
        used = " (stored)" if rule == window.label_sel else ""
        draw(ax, window.targets[rule], "binary", (0, 1),
             f"target, label_sel {rule}{used}\n{int(window.targets[rule].sum())} px newly burned",
             changed=bool(used))
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    figure.suptitle("Labels: masks per day, accumulated burned area, next-day target (red dots: fire pixels)", y=1.0)
    figure.tight_layout()
    plt.show()
    return figure


def plot_running_max(window, config, size=2.0):
    """Draw the heat channels before and after the running maximum, day by day.

    build_fire_windows replaces channels 3, 4, 6 and 7 (I4, I5, day and night)
    with their maximum over the window's days so far, so a pixel that was hot
    earlier still looks hot. Each channel gets two rows: the day's own value,
    then the stored running maximum. The fix does not touch these channels.

    Args:
        window (WindowTrace): the first window.
        config (module): the dataset settings.
        size (float): panel side in inches.

    Returns:
        matplotlib.figure.Figure: the figure.
    """
    channels = (3, 4, 6, 7)
    length = window.stored.shape[1]
    figure, axes = plt.subplots(2 * len(channels), length, figsize=(length * size, 2 * len(channels) * size * 1.08),
                                squeeze=False)
    for k, channel in enumerate(channels):
        value_range = display_range([window.frames[:length, channel], window.stored[channel]], "value")
        name = config.STORED_CHANNEL_NAMES[channel]
        for d in range(length):
            draw(axes[2 * k, d], window.frames[d, channel], "value", value_range, f"{name}, day value\n{window.dates[d]}")
            raised = not np.array_equal(window.frames[d, channel], window.stored[channel, d])
            draw(axes[2 * k + 1, d], window.stored[channel, d], "value", value_range,
                 f"{name}, running max\n{window.dates[d]}", changed=raised)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    figure.suptitle("Heat memory: day value vs stored running maximum (red: raised by an earlier day)", y=1.0)
    figure.tight_layout()
    plt.show()
    return figure


def plot_window_over_time(windows, config, size=1.5):
    """Draw every stored channel on every input day of the window, one figure per pipeline.

    Rows are the 27 channels, columns the input days. A pipeline whose window
    equals the first one is reported, not drawn again. In the others, a red
    title marks a panel that differs from the first pipeline's.

    Args:
        windows (dict[str, WindowTrace]): {pipeline: first window}.
        config (module): the dataset settings.
        size (float): panel side in inches.

    Returns:
        list[matplotlib.figure.Figure]: the figures drawn.
    """
    names, kinds = stored_layout(config)
    labels = list(windows)
    first = windows[labels[0]].stored
    figures = []
    for label in labels:
        stored = windows[label].stored
        if label != labels[0] and same(stored, first):
            print(f"window over time | {label}: identical to {labels[0]}, not drawn again")
            continue
        channels, length = stored.shape[:2]
        figure, axes = plt.subplots(channels, length, figsize=(length * size * 1.25, channels * size * 1.1),
                                    squeeze=False)
        for c in range(channels):
            value_range = display_range([w.stored[c] for w in windows.values()], kinds[c])
            for d in range(length):
                changed = label != labels[0] and not same(stored[c, d], first[c, d])
                draw(axes[c, d], stored[c, d], kinds[c], value_range, f"{c}: {names[c]}\n{windows[label].dates[d]}",
                     changed=changed)
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        figure.suptitle(f"Stored window (27 channels x {length} days) | {label}", y=1.0)
        figure.tight_layout()
        plt.show()
        figures.append(figure)
    return figures


def plot_targets(targets, title, size=2.6):
    """Draw two-channel targets (background, newly burned) of one or more versions side by side.

    read_window turns the stored 0/1 target into two channels: channel 0 is 1
    where nothing new burned, channel 1 where something did. Fire pixels are
    marked with red dots.

    Args:
        targets (dict[str, torch.Tensor | np.ndarray]): {version: (2, H, W)}.
        title (str): what step the targets come from.
        size (float): panel side in inches.

    Returns:
        matplotlib.figure.Figure: the figure.
    """
    figure, axes = panel_grid(2 * len(targets), 2 * len(targets), size, title)
    for k, (label, target) in enumerate(targets.items()):
        target = as_numpy(target).astype(np.float32)
        draw(axes[2 * k], target[0], "binary", (0, 1), f"{label}\nchannel 0: background")
        draw(axes[2 * k + 1], target[1], "binary", (0, 1), f"{label}\nchannel 1: newly burned ({int(target[1].sum())} px)")
    figure.tight_layout()
    plt.show()
    return figure
