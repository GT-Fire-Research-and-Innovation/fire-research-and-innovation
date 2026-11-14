"""Every data constant: input channels, dataset settings, repositories, paths.

Level 0 (config). Used by data/ (normalization, preprocessing, archives; building,
publishing and staging windows), models/ (input width), training/ (channel checks)
and the CLIs. Every function that uses a setting also accepts it as an argument,
so a notebook can override it for one call without editing this file.

Sections: input channels, dataset settings, filesystem paths, preprocessing defaults.

Input channels. A stored window holds 27 channels, in the order of the
TS-SatFire paper's Table 1 (figures/Bands.png), 0-based here. Before the
model sees it:
  * every channel except the angles and land cover is standardized with the
    training split's mean/std from <data-root>/window_config.json (or, with
    --normalization paper, NORMALIZATION_MEAN/STD);
  * the observed wind direction (degrees) becomes degrees / 180 - 1, so -1..1
    covers the whole compass; aspect and forecast wind direction become sin(angle);
  * land cover (integer class 1..17, 0 = missing) becomes 17 one-hot planes.
So the model receives 27 - 1 + 17 = 43 channels. TESSERA mode appends 128
embedding bands plus 1 validity mask: 43 + 129 = 172.
"""

from pathlib import Path

# =============================== Input channels =============================== #

# Stored channel order. Channels 0-5 are VIIRS day bands, 6-7 VIIRS night
# bands, 8-26 the FirePred auxiliary bands (vegetation, weather, terrain,
# fuel). Channels 3, 4, 6, 7 hold the running maximum over the window's days
# (see firepred/data/windows.py).
STORED_CHANNEL_NAMES = (
    "VIIRS I1",                      # 0
    "VIIRS I2",                      # 1
    "VIIRS I3",                      # 2
    "VIIRS I4",                      # 3  running max
    "VIIRS I5",                      # 4  running max
    "VIIRS M11",                     # 5
    "VIIRS I4 night",                # 6  running max
    "VIIRS I5 night",                # 7  running max
    "NDVI",                          # 8
    "EVI",                           # 9
    "total precipitation",           # 10
    "wind speed",                    # 11
    "wind direction",                # 12 degrees
    "min temperature",               # 13
    "max temperature",               # 14
    "energy release component",      # 15
    "specific humidity",             # 16
    "slope",                         # 17
    "aspect",                        # 18 degrees
    "elevation",                     # 19
    "PDSI",                          # 20 Palmer drought severity index
    "land cover",                    # 21 class 1..17
    "total precipitation surface",   # 22
    "forecast wind speed",           # 23
    "forecast wind direction",       # 24 degrees
    "forecast temperature",          # 25
    "forecast specific humidity",    # 26
)
N_STORED_CHANNELS = 27

# Stored channels FIREPRED_FIRST_CHANNEL.. come from the FirePred GeoTIFF (its band 0 onward).
FIREPRED_FIRST_CHANNEL = 8

# Angles in degrees. Must stay a list: x[[12, 18, 24]] selects three
# channels, while x[(12, 18, 24)] would index a single element. [released]
DEGREE_CHANNELS = [12, 18, 24]

# Land-cover class per pixel, replaced by one-hot planes. [released]
LANDCOVER_CHANNEL = 21
N_LANDCOVER_CLASSES = 17

# The observed (gridMET) wind direction: a full 0..360 compass angle. The model
# sees it as degrees / 180 - 1 (data/fire_dataset.preprocess): north (0) is -1,
# east -0.5, south 0, west 0.5, and the values rise toward 1 as the direction
# returns to north, so every direction gets its own value. The released
# sin(direction) gave north and south both 0. Aspect and the forecast wind
# direction (stored as -90..90, half a circle) keep sin(). The archives hold
# degrees either way, so the encoding belongs to the code, not the dataset. [ours]
WIND_DIRECTION_CHANNEL = 12

# Per-channel training statistics of the 27 stored channels, from the authors'
# training data. Used with --normalization paper and for checkpoints that predate
# window_config.json statistics. [released]
NORMALIZATION_MEAN = [
    18.224253,
    26.95519,
    20.09066,
    318.25967,
    308.78717,
    14.165086,
    291.29214,
    288.97382,
    5110.5547,
    2556.2627,
    0.3907487,
    3.4994626,
    216.23518,
    276.5463,
    291.8275,
    70.32086,
    0.0054306216,
    10.120554,
    175.33012,
    1290.8367,
    -1.5219007,
    7.3989105,
    7.584937,
    1.4395763,
    3.306973,
    19.259102,
    0.0057929577,
]
NORMALIZATION_STD = [
    15.438321,
    14.408274,
    10.552524,
    13.1312475,
    12.155249,
    9.652911,
    12.435288,
    8.750125,
    2400.766,
    1206.8983,
    2.37979,
    1.6343528,
    85.730644,
    47.332256,
    50.045837,
    22.48386,
    0.0021515382,
    8.429097,
    104.73222,
    823.01483,
    1.9954495,
    4.1257873,
    26.547232,
    1.2017097,
    48.207355,
    5.4114914,
    0.0017134654,
]

# Model input width without TESSERA: land cover's 1 channel becomes 17.
BASE_MODEL_CHANNELS = N_STORED_CHANNELS - 1 + N_LANDCOVER_CLASSES  # 43

# TESSERA: 128 embedding bands per pixel plus one validity mask. [ours]
TESSERA_BANDS = 128
TESSERA_CHANNELS = TESSERA_BANDS + 1
# TESSERA cells along each side of one 375 m VIIRS pixel. 1 averages every 10 m
# pixel inside the VIIRS pixel into one vector; k > 1 keeps k x k finer cells
# (5 -> 75 m cells), which a learned stem compresses back to the VIIRS grid. [ours]
TESSERA_CELLS_PER_PIXEL = 1


def model_input_channels(use_tessera):
    """Count the channels the model receives for one input day.

    Each stored window has 27 channels; land cover is expanded into 17 one-hot
    planes, giving 43. TESSERA mode adds 128 embedding bands and one validity
    mask, giving 172. Use this number wherever a model's input width is needed.

    Args:
        use_tessera (bool): True when TESSERA embeddings are appended.

    Returns:
        int: 43 without TESSERA, 172 with it.
    """
    return BASE_MODEL_CHANNELS + (TESSERA_CHANNELS if use_tessera else 0)


# ============================== Dataset settings ============================== #

# ----- Spatial crop ----- #

# Every GeoTIFF is cropped to rows/cols CROP_OFFSET:CROP_OFFSET + CROP_SIZE,
# so one 256 x 256 grid (375 m VIIRS pixels, 96 km a side) covers a fire on
# every day. [released]
CROP_OFFSET = 128
CROP_SIZE = 256

# ----- Window generation ----- #

# Input days per window. A fire needs TS_LENGTH + 1 VIIRS_Day images (the
# extra day is the target), so fires with fewer images produce no windows.
# The TS-SatFire paper's prediction benchmark uses 6 input days.
TS_LENGTH = 6

# Days between the first input day of consecutive windows.
TS_INTERVAL = 1

# ----- Hugging Face dataset repositories ----- #

SRC_REPO = "SamuelWu318/ts-satfire"                    # raw GeoTIFFs + ROI CSVs
TESSERA_REPO = "SamuelWu318/ts-tesserafire"            # prior-year TESSERA per fire

# Windowed NPZs live in one repo per (TS_LENGTH, TS_INTERVAL): file names do
# not encode the window length, so different settings must never share a repo.
# Names follow the repos on Hugging Face: ...-processed-6-1, ...-processed-10-3.
WINDOW_REPO_BASE = "SamuelWu318/ts-satfire-processed"


def window_repo_for(ts_length=TS_LENGTH, ts_interval=TS_INTERVAL):
    """Name the Hugging Face repo that holds windows of one length and stride.

    Window archives do not record how long their windows are, so every
    (length, stride) pair gets its own repo. This keeps windows built with
    different settings from ever mixing.

    Args:
        ts_length (int): input days per window.
        ts_interval (int): days between the starts of consecutive windows.

    Returns:
        str: the repo id, e.g. "SamuelWu318/ts-satfire-processed-6-1".
    """
    return f"{WINDOW_REPO_BASE}-{ts_length}-{ts_interval}"


# The default repo: TS_LENGTH-day windows, one starting every TS_INTERVAL days.
WINDOW_REPO = window_repo_for()

# Written at the root of each window repo by process_data; checked before
# new windows are added so one repo never mixes window settings.
WINDOW_CONFIG_FILE = "window_config.json"

# window_config.json names the window repo its windows come from under this key
# (its first key), and training logs it to W&B. It records provenance, not a window
# setting, so two copies are never compared on it. [ours]
WINDOW_CONFIG_DATASET_KEY = "dataset"

# A Hugging Face dataset repo's page.
HF_DATASET_URL = "https://huggingface.co/datasets/{repo}"


def dataset_url(repo):
    """Give the Hugging Face page of a dataset repo (HF_DATASET_URL with the repo id filled in)."""
    return HF_DATASET_URL.format(repo=repo)


def with_dataset(config, repo):
    """Give a copy of a window_config.json dict that names `repo` as its dataset (its first key)."""
    rest = {key: value for key, value in config.items() if key != WINDOW_CONFIG_DATASET_KEY}
    return {WINDOW_CONFIG_DATASET_KEY: repo, **rest}

# ----- Window archives ----- #

# One archive per fire: data_###/label_### members plus num_windows (see
# firepred/data/archive.py). [ours]
ARCHIVE_FORMAT_VERSION = 2
SPLITS = ("train", "val", "test")

# Zstandard level of every member of the window and TESSERA archives
# (data/archive.py; 1 = fastest, 22 = smallest). Only the compressed bytes change:
# the arrays read back identically at every level. Versus zlib level 1 (used
# before), level 6 decompressed about 1.8x faster and packed a quantized float32
# field 16% tighter, for 14% slower writing (synthetic data, not measured on
# real archives).
NPZ_COMPRESSLEVEL = 6

# ----- Daily archives ----- #

# Format version 3 stores each fire's days once instead of its windows: with
# -it 1 every day sat in up to TS_LENGTH windows, and day t of window k is
# bit-identical to day t-1 of window k+1 (the running maxima of channels 3, 4,
# 6, 7 carry over from the fire's first day, never from a window's). Windows are
# sliced from the days when read, for the archive's own window settings or any
# other whose target days it holds. See firepred/data/archive.py. [ours]
DAILY_FORMAT_VERSION = 3
ARCHIVE_FORMAT_VERSIONS = (ARCHIVE_FORMAT_VERSION, DAILY_FORMAT_VERSION)

# Days per compressed block of one channel. Compressing each day alone packs
# about 2x worse (neighbouring days share most pixels); one block per channel
# per fire packs tightest but makes every window decompress the whole fire.
# 6 days matched the whole-fire size within 1% on real fires (30.4 -> 5.5 MB,
# 12 windows) and decoded a window in about 15 ms, as fast as a format-2 window.
DAILY_BLOCK_DAYS = 6

# Daily archives are written once and read many times. Zstandard decodes at the
# same speed at every level, so they use a high one: 19 packed real fires 13%
# tighter than 6, for slower writing only.
DAILY_COMPRESSLEVEL = 19

# The Hugging Face repo of daily archives converted from the 6-1 window repo. One
# repo serves every window length >= 6 and every stride. [ours]
DAILY_REPO = "SamuelWu318/ts-satfire-daily"

# ----- Prediction labels ----- #

# The target is the pixels newly burned on the day after the input window:
# "the difference between the burned area masks of the last observed day and
# the next day" (TS-SatFire paper). label_sel picks the burned-area mask:
#   0: union of accumulated VIIRS active fire (AF) and the NIFC perimeters
#      (BA band). The paper's training label, and test option (a).
#   1: accumulated AF only. Test option (b), chosen per fire by visual
#      inspection "to avoid the high commission errors of NIFC perimeters".
# Train/val fires use TRAIN_VAL_LABEL_SEL; test fires use the label_sel
# column of the 2021 ROI CSV, which records the paper's per-fire choice.
# The released TS-SatFire code used 1 for train/val; set 1 to reproduce it.
TRAIN_VAL_LABEL_SEL = 0

# ----- Train / validation / test split ----- #

TRAIN_VAL_YEARS = ('2017', '2018', '2019', '2020')
TEST_YEARS = ('2021',)
# A test fire the trainer always skips. [released]
EXCLUDED_TEST_FIRES = ("US_2021_NV3700011641620210517",)
VAL_IDS = [
    '20568194', '20701026', '20562846', '20700973', '24462610',
    '24462788', '24462753', '24103571', '21998313', '21751303',
    '22141596', '21999381', '23301962', '22712904', '22713339',
]

# ----- Fire split zones ----- #

# One crop can hold several fires: a new ignition far from the main fire, or one
# fire whose front breaks into two. data/fire_tracks.py follows each fire day by
# day from the window archives' labels, and data/fire_zones.py turns that into
# per-window zone files zones/p<id>.npz. One distance sets every threshold:
# gaps smaller than it are noise to bridge, gaps larger than it separate two
# fires. Smaller = more splits; larger = more cleaning. 2.25 km matches FIRED's
# MTBS-tuned 2.3 km and stays above 2x the 0.8 km VIIRS edge-of-scan pixel. [ours]
FIRE_SEPARATION_KM = 2.25
VIIRS_PIXEL_KM = 0.375
# Spread rates that turn the separation distance into days. Slow (0.2 km/day,
# FIRED 2.3 km / 11 days and FEDS 1 km / 5 days agree) sets how long a quiet
# fire is remembered; fast (1 km/day, FEDS forest buffer, Paugam 2026) sets how
# many recent days count as a fire's front when testing for a split. At 2.25 km
# that is FIRED's pairing: 11 days of memory, so a fire quiet for up to 11 days
# (clouds, a lull) is still the same fire, as FIRED and fire agencies treat it.
# Memory is not tied to the window length: zones may come from up to 5 days
# before a 6-day window's inputs, all of them past days. [ours]
SLOW_SPREAD_KM_PER_DAY = 0.2
FAST_SPREAD_KM_PER_DAY = 1.0
# Days a new fire must grow on (and a split must hold) before it counts.
MIN_GROWTH_DAYS = 2
# A fire holding less than this share of a window's fire pixels gets no zone and
# no training sample there, like a group not yet confirmed as a fire: its new
# pixels are left out of every loss. It is dropped, not merged: a fire tracking
# kept apart lies farther than the separation distance from every other fire,
# so merging would hand its growth to a fire it never touched. Tracking still
# records it. 1% of the 256 x 256 crop (655 px) would drop most whole fires, so
# the share is of the window's fire pixels. [ours]
MIN_FIRE_SHARE = 0.01
# Before the first target day the labels say nothing, so night I4 hotter than
# this marks early fire pixels ("seeds"; night background is about 290 K). They
# count like burned pixels; scattered warm pixels never group into a fire.
SEED_NIGHT_I4_KELVIN = 310.0
NIGHT_I4_CHANNEL = 6
ZONE_FORMAT_VERSION = 2  # 2: settings record min_fire_share, and small fires have no zone
ZONE_DIR = "zones"

# ----- TESSERA embeddings ----- #

# Per-fire files tessera/p<id>.npz (see firepred/data/tessera.py). [ours]
TESSERA_FORMAT_VERSION = 1
# How a TESSERA file stores its embedding (data/archive.py). "none" keeps float16.
# "int8" cuts each band's range in that fire into 254 equal steps, so nothing is
# clipped. Measured on four real 5 x 5 maps: files 2.3x smaller (366 -> 155 MB),
# RMS error 0.014-0.018 band standard deviations, band spreads kept within 0.2%,
# and faster to read. Readers handle both and return float16. [ours]
TESSERA_QUANTIZATIONS = ("none", "int8")
TESSERA_QUANTIZATION = "none"
# First year of published TESSERA embeddings; fires need the year before.
FIRST_TESSERA_YEAR = 2017
# Drop list shipped inside tessera.zip, one window archive path per line.
TESSERA_EXCLUDED_LIST = "excluded_npz.txt"


# =============================== Filesystem paths =============================== #

# firepred/config/dataset.py -> the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
# Written at the root of a code copy by tools/pack_code.py (Kaggle / Colab run a
# packed copy of the local checkout, not a git clone): which code it is. [ours]
CODE_MANIFEST_FILE = "code_version.json"

# Fire metadata (Id, start_date, lat/lon, label_sel) shipped with the repo.
ROI_DIR = REPO_ROOT / "roi"

# Test fires in scoring order: the 2021 fires. [released]
DEFAULT_TEST_CSV = ROI_DIR / "us_fire_2021_out_new.csv"

# Training's data root without --data-root: ./data, where
# `python -m firepred.cli.dataset download` / `pull-zip` put the archives.
# The notebooks always pass --data-root.
DEFAULT_DATA_ROOT = Path("data")


# ============================ Preprocessing defaults ============================ #

# Local folders and file names used by the dataset pipeline (data/), relative
# to the working directory unless a master function is given other paths.
WORK_DIR = "."                 # temp/ and staging/ go here while building
DATA_DIR = "./data"            # a local data root: train/, val/, test/ (+ tessera/)
DAILY_DATA_DIR = "./data-daily"  # convert-daily writes daily archives here
ROI_CACHE_DIR = "hf_roi"       # downloaded ROI CSVs
DATA_ZIP = "data.zip"          # the window data bundle in a window repo
TESSERA_ZIP = "tessera.zip"    # the TESSERA bundle in TESSERA_REPO
# Folders of a data root that go into data.zip (missing ones are skipped).
ZIP_FOLDERS = ("test", "train", "val", "tessera")
# Fires per Hugging Face commit while publishing windows / TESSERA files.
WINDOW_UPLOAD_BATCH = 10
TESSERA_UPLOAD_BATCH = 1
# Converted TESSERA files per commit (tessera_master.convert_tessera_repo), which is
# also the most kept on disk at once: about 10 x 155 MB as int8 at 5 x 5 cells.
TESSERA_CONVERT_BATCH = 10
# Rows per TESSERA strip read at once (memory vs number of requests).
STRIP_ROWS = 128
