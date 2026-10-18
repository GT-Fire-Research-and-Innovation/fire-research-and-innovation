"""Filesystem locations.

Level 0 (config). Used by the CLIs for their defaults.
"""

from pathlib import Path

# firepred/config/paths.py -> the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Fire metadata (Id, start_date, lat/lon, label_sel) shipped with the repo.
ROI_DIR = REPO_ROOT / "roi"

# Test fires in scoring order: the 2021 fires. [released]
DEFAULT_TEST_CSV = ROI_DIR / "us_fire_2021_out_new.csv"

# Where Colab's data-grab notebook unpacks the window archives.
DEFAULT_DATA_ROOT = Path("/content/data")
