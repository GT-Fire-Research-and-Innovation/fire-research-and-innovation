#!/usr/bin/env bash
# One-time setup on a PACE ICE login node. Open a terminal there with
#   ssh <gt-username>@login-ice.pace.gatech.edu        (GT VPN when off campus)
# or in a browser: ICE OnDemand > Clusters > ICE Shell Access. Then, from the repository root:
#   bash scripts/ice/setup.sh
# It builds the Python environment in ~/scratch, then runs the Hugging Face and W&B logins
# (paste your tokens into the terminal prompts only, never into a chat or a file in the repo).
# Safe to run again: an existing environment is reused and only updated.
#
# Python (3.9 or newer):
#   default              a venv from ICE's own python3 (Python 3.9 on ICE's RHEL 9)
#   PYTHON=/path/python  a venv from another interpreter
#   CONDA_PYTHON=3.12    a conda environment with that Python (from ICE's mamba or anaconda3 module)
# Libraries:
#   STACK=paper (default)  torch 2.3.1 + MONAI 1.3.2 (CUDA 12.1 wheels), the paper's versions and the
#                          ones the golden tests were captured with; numpy stays below 2 for this torch
#   STACK=latest           the newest torch and MONAI for this Python (CUDA 12.6 wheels; torch 2.8 on
#                          Python 3.9). Golden model and training values then differ slightly, since
#                          torch changes floating-point results between versions.
#   TORCH_INDEX=URL        where torch comes from instead (e.g. https://download.pytorch.org/whl/cu118)
# The whole test suite: sbatch scripts/ice/test.sbatch (a CPU job; the login node only runs a quick check).
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/ice/env.sh
STACK="${STACK:-paper}"

if [ ! -x "$FIREPRED_ENV/bin/python" ]; then
  mkdir -p "$(dirname "$FIREPRED_ENV")"
  if [ -n "${CONDA_PYTHON:-}" ]; then
    module load mamba 2>/dev/null || module load anaconda3 2>/dev/null || true
    CONDA=$(command -v mamba || command -v conda || true)
    [ -n "$CONDA" ] || { echo "No mamba or conda module found: run 'module avail' and load one, then rerun"; exit 1; }
    "$CONDA" create -y -p "$FIREPRED_ENV" -c conda-forge python="$CONDA_PYTHON" pip
    module purge 2>/dev/null || true
  else
    PYTHON="${PYTHON:-python3}"
    "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 9))' \
      || { echo "!! $PYTHON is $("$PYTHON" -V 2>&1); firepred needs Python 3.9 or newer (set PYTHON=...)"; exit 1; }
    "$PYTHON" -m venv "$FIREPRED_ENV"
  fi
fi
export PATH="$FIREPRED_ENV/bin:$PATH"
PY="$FIREPRED_ENV/bin/python"
echo "Python: $("$PY" -V 2>&1) in $FIREPRED_ENV"
"$PY" -m pip install --upgrade pip

# The GPU wheels run on ICE's H100/H200/A100/V100 nodes; the login node has no GPU, which is fine for installing.
case "$STACK" in
  paper)
    "$PY" -m pip install --upgrade --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}" \
      torch==2.3.1 torchvision==0.18.1
    PINS=("numpy<2" monai==1.3.2) ;;
  latest)
    "$PY" -m pip install --upgrade --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}" torch torchvision
    PINS=(numpy monai) ;;
  *) echo "!! STACK must be paper or latest, not '$STACK'"; exit 1 ;;
esac
# backports.zstd reads the Zstandard archives before Python 3.14; its last release for Python 3.9 is 1.3.0.
"$PY" -m pip install --upgrade "${PINS[@]}" scipy pandas matplotlib tqdm huggingface_hub rasterio pyproj \
  einops wandb pytest 'backports.zstd; python_version<"3.14"'
"$PY" -c "import sys, torch, monai, numpy, scipy; print('python', sys.version.split()[0], '| torch', torch.__version__,
'| monai', monai.__version__, '| numpy', numpy.__version__, '| scipy', scipy.__version__)"
"$PY" -m pytest -q tests/test_daily.py   # quick CPU check of the code on ICE

# The processed window repos are gated: log in once (older huggingface_hub has only huggingface-cli).
if command -v hf >/dev/null; then hf auth whoami >/dev/null 2>&1 || hf auth login
else huggingface-cli whoami >/dev/null 2>&1 || huggingface-cli login; fi
wandb login                                          # entity gt-fri, project AFBAPred
mkdir -p logs
echo "Ready. Next: bash scripts/ice/partitions.sh (where you can run), sbatch scripts/ice/test.sbatch (all tests),"
echo "       then bash scripts/ice/plan.sh  (see README, 'GT PACE ICE', and scripts/ice/QUICKSTART.md)"
