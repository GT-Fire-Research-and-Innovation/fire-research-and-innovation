#!/usr/bin/env bash
# One-time setup on a PACE ICE login node. Open a terminal there with
#   ssh <gt-username>@login-ice.pace.gatech.edu        (GT VPN when off campus)
# or in a browser: ICE OnDemand > Clusters > ICE Shell Access. Then, from the repository root:
#   bash scripts/ice/setup.sh
# It builds the Python environment in ~/scratch, then runs the Hugging Face and W&B logins
# (paste your tokens into the terminal prompts only, never into a chat or a file in the repo).
# Safe to run again: an existing environment is reused and only updated.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/ice/env.sh

# ICE provides conda through modules; whichever exists is used only to create the environment.
module load mamba 2>/dev/null || module load anaconda3 2>/dev/null || true
CONDA=$(command -v mamba || command -v conda || true)
[ -n "$CONDA" ] || { echo "No mamba or conda module found: run 'module avail' and load one, then rerun"; exit 1; }
if [ ! -x "$FIREPRED_ENV/bin/python" ]; then
  "$CONDA" create -y -p "$FIREPRED_ENV" -c conda-forge \
    python=3.12 numpy scipy pandas matplotlib tqdm huggingface_hub rasterio pip
fi
module purge 2>/dev/null || true
export PATH="$FIREPRED_ENV/bin:$PATH"

# CUDA 12.4 wheels run on ICE's H100/H200/A100 nodes; the login node has no GPU, which is fine for installing.
pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install --upgrade monai einops wandb pytest
python -c "import torch, monai, scipy; print('torch', torch.__version__, '| monai', monai.__version__, '| scipy', scipy.__version__)"
python -m pytest -q tests/test_daily.py   # quick CPU check of the code on ICE

# The processed window repos are gated: log in once (older huggingface_hub has only huggingface-cli).
if command -v hf >/dev/null; then hf auth whoami >/dev/null 2>&1 || hf auth login
else huggingface-cli whoami >/dev/null 2>&1 || huggingface-cli login; fi
wandb login                                          # entity gt-fri, project AFBAPred
mkdir -p logs
echo "Ready. Next: bash scripts/ice/plan.sh  (see README, 'GT PACE ICE')"
