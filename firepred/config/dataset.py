"""Dataset settings: windows, crop, labels, splits, archive format, repositories.

Level 0 (config). Used by prep/ (building windows), hub/ (Hugging Face),
data/ (reading archives) and the CLIs. Every function that uses a setting
also accepts it as an argument, so a notebook can override it for one call
without editing this file.
"""

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
