"""Every data constant: input channels, dataset settings, repositories, paths.

Level 0 (config). Used by data/ (normalization, preprocessing, archives),
models/ (input width), prep/ and hub/ (building and publishing windows),
training/ (channel checks) and the CLIs. Every function that uses a setting
also accepts it as an argument, so a notebook can override it for one call
without editing this file.

Sections: input channels, dataset settings, filesystem paths.

Input channels. A stored window holds 27 channels, in the order of the
TS-SatFire paper's Table 1 (figures/Bands.png), 0-based here. Before the
model sees it:
  * every channel except the angles and land cover is standardized with
    NORMALIZATION_MEAN/STD;
  * the three angle channels (degrees) become sin(angle);
  * land cover (integer class 1..17, 0 = missing) becomes 17 one-hot planes.
So the model receives 27 - 1 + 17 = 43 channels. TESSERA mode appends 128
embedding bands plus 1 validity mask: 43 + 129 = 172.
"""

from pathlib import Path

# =============================== Input channels =============================== #

# Stored channel order. Channels 0-5 are VIIRS day bands, 6-7 VIIRS night
# bands, 8-26 the FirePred auxiliary bands (vegetation, weather, terrain,
# fuel). Channels 3, 4, 6, 7 hold the running maximum over the window's days
# (see firepred/prep/windows.py).
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

# Angles in degrees. Must stay a list: x[[12, 18, 24]] selects three
# channels, while x[(12, 18, 24)] would index a single element. [released]
DEGREE_CHANNELS = [12, 18, 24]

# Land-cover class per pixel, replaced by one-hot planes. [released]
LANDCOVER_CHANNEL = 21
N_LANDCOVER_CLASSES = 17

# Per-channel training statistics of the 27 stored channels. [released]
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


def model_input_channels(use_tessera):
    """43, or 172 when TESSERA channels are appended."""
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
    """Window repo holding windows of `ts_length` days every `ts_interval` days."""
    return f"{WINDOW_REPO_BASE}-{ts_length}-{ts_interval}"


# The default repo: TS_LENGTH-day windows, one starting every TS_INTERVAL days.
WINDOW_REPO = window_repo_for()

# Written at the root of each window repo by process_data; checked before
# new windows are added so one repo never mixes window settings.
WINDOW_CONFIG_FILE = "window_config.json"

# ----- Window archives ----- #

# One archive per fire: data_###/label_### members plus num_windows (see
# firepred/data/archive.py). [ours]
ARCHIVE_FORMAT_VERSION = 2
SPLITS = ("train", "val", "test")

# zlib level for the window archives (1 = fastest, 9 = smallest). Only the
# compressed bytes change: np.load returns identical arrays at every level.
# On a real 18-window fire, level 1 saved 3.0x faster than NumPy's default
# (level 6) for a 14% larger file.
NPZ_COMPRESSLEVEL = 1

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

# ----- TESSERA embeddings ----- #

# Per-fire files tessera/p<id>.npz (see firepred/prep/tessera_grid.py). [ours]
TESSERA_FORMAT_VERSION = 1
# First year of published TESSERA embeddings; fires need the year before.
FIRST_TESSERA_YEAR = 2017
# Drop list shipped inside tessera.zip, one window archive path per line.
TESSERA_EXCLUDED_LIST = "excluded_npz.txt"


# =============================== Filesystem paths =============================== #

# firepred/config/data.py -> the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Fire metadata (Id, start_date, lat/lon, label_sel) shipped with the repo.
ROI_DIR = REPO_ROOT / "roi"

# Test fires in scoring order: the 2021 fires. [released]
DEFAULT_TEST_CSV = ROI_DIR / "us_fire_2021_out_new.csv"

# Where Colab's data-grab notebook unpacks the window archives.
DEFAULT_DATA_ROOT = Path("/content/data")


# ============================ Preprocessing defaults ============================ #

# Local folders and file names used by the dataset pipeline (prep/), relative
# to the working directory unless a master function is given other paths.
WORK_DIR = "."                 # temp/ and staging/ go here while building
DATA_DIR = "./data"            # a local data root: train/, val/, test/ (+ tessera/)
ROI_CACHE_DIR = "hf_roi"       # downloaded ROI CSVs
DATA_ZIP = "data.zip"          # the window data bundle in a window repo
TESSERA_ZIP = "tessera.zip"    # the TESSERA bundle in TESSERA_REPO
# Folders of a data root that go into data.zip (missing ones are skipped).
ZIP_FOLDERS = ("test", "train", "val", "tessera")
# Fires per Hugging Face commit while publishing windows / TESSERA files.
WINDOW_UPLOAD_BATCH = 10
TESSERA_UPLOAD_BATCH = 1
# Rows per TESSERA strip read at once (memory vs number of requests).
STRIP_ROWS = 128
