#!/usr/bin/env bash
# One-time setup on a PACE ICE login node. Open a terminal there with
#   ssh <gt-username>@login-ice.pace.gatech.edu        (GT VPN when off campus)
# or in a browser: ICE OnDemand > Clusters > ICE Shell Access. Then, from the repository root:
#   bash scripts/ice/setup.sh
# It builds the Python environment in ~/scratch, then runs the Hugging Face and W&B logins
# (paste your tokens into the terminal prompts only, never into a chat or a file in the repo).
# Safe to run again: an existing environment is reused and only updated. Rerun it whenever a library
# is missing (ModuleNotFoundError): it checks every import and installs what is missing.
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
# Optional tests: sbatch scripts/ice/test.sbatch (a CPU job; setup only checks library imports).
set -euo pipefail
cd "$(dirname "$0")/../.."
FIREPRED_SETUP=1                       # env.sh: the environment may not exist yet
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

# torch comes from PyTorch's own index. Its GPU wheels run on ICE's NVIDIA nodes; the login node has no
# GPU, which is fine for installing.
case "$STACK" in
  paper)  TORCH=(torch==2.3.1 torchvision==0.18.1); PINS=("numpy<2" monai==1.3.2)
          TORCH_URL="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}" ;;
  latest) TORCH=(torch torchvision); PINS=(numpy monai)
          TORCH_URL="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}" ;;
  *) echo "!! STACK must be paper or latest, not '$STACK'"; exit 1 ;;
esac
# Everything else firepred imports. pip picks the newest release for this Python; on ICE's Python 3.9
# that is numpy 1.26, scipy 1.13, pandas 2.3, matplotlib 3.9, rasterio 1.4, pyproj 3.6.1 and
# backports.zstd 1.3.0. backports.zstd reads the Zstandard archives before Python 3.14.
PACKAGES=(scipy pandas matplotlib tqdm huggingface_hub rasterio pyproj einops wandb pytest)
FORCE=(huggingface_hub)                # always reinstalled from scratch (see below)
if "$PY" -c 'import sys; sys.exit(sys.version_info >= (3, 14))'; then
  PACKAGES+=(backports.zstd); FORCE+=(backports.zstd)
fi
MODULES=(torch torchvision monai numpy "${PACKAGES[@]}")   # import names, which are also the pip names

# One package that fails must not leave the others uninstalled: a failed install is only reported here,
# and the check below installs whatever is still missing.
"$PY" -m pip install --upgrade --index-url "$TORCH_URL" "${TORCH[@]}" || echo "!! torch did not install"
"$PY" -m pip install --upgrade "${PINS[@]}" "${PACKAGES[@]}" || echo "!! some packages did not install"
# pip skips a package whose record is still there even when its files are gone ("Requirement already
# satisfied", then ModuleNotFoundError), so these two are always reinstalled. Both are small.
"$PY" -m pip install --force-reinstall --no-deps "${FORCE[@]}" || echo "!! ${FORCE[*]} did not reinstall"

# missing MODULE...: the modules this Python cannot import, one per line (each reason goes to stderr).
missing() {
  "$PY" - "$@" <<'EOF'
import importlib
import sys
for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
    except Exception as error:
        print(name)
        print(f"   cannot import {name}: {type(error).__name__}: {error}", file=sys.stderr)
EOF
}
for name in $(missing "${MODULES[@]}"); do
  echo "Installing $name again on its own"
  case "$name" in
    torch|torchvision) "$PY" -m pip install --force-reinstall --no-deps --index-url "$TORCH_URL" "${TORCH[@]}" || true
                       "$PY" -m pip install --index-url "$TORCH_URL" "${TORCH[@]}" || true ;;
    *) "$PY" -m pip install --force-reinstall --no-deps "$name" || true      # files gone, record left
       "$PY" -m pip install "${PINS[@]}" "$name" || true ;;                  # and its dependencies
  esac
done
MISSING=$(missing "${MODULES[@]}")
if [ -n "$MISSING" ]; then
  echo "!! Still missing in $FIREPRED_ENV:" $MISSING
  echo "   The pip errors above say why. Fix that, then rerun bash scripts/ice/setup.sh (it keeps what is installed)."
  exit 1
fi
# firepred's own command lines must start too, which also shows the code runs on this Python.
"$PY" -m firepred.cli.dataset --help >/dev/null && "$PY" -m firepred.cli.train --help >/dev/null \
  || { echo "!! firepred does not start in $FIREPRED_ENV (error above)"; exit 1; }
"$PY" - "${MODULES[@]}" <<'EOF'
import sys
from importlib import metadata
versions = [f"{name} {metadata.version(name.replace('.', '_'))}" for name in sys.argv[1:]]   # 3.9 needs "_"
print("All imports work. python", sys.version.split()[0], "|", " | ".join(versions))
EOF

# The processed window repos are gated: log in once (older huggingface_hub has only huggingface-cli).
if command -v hf >/dev/null; then hf auth whoami >/dev/null 2>&1 || hf auth login
else huggingface-cli whoami >/dev/null 2>&1 || huggingface-cli login; fi
wandb login                                          # entity gt-fri, project AFBAPred
mkdir -p logs
echo "Ready. Next: bash scripts/ice/partitions.sh (where you can run), then NAME=swin_daily bash scripts/ice/plan.sh"
echo "Optional tests: sbatch scripts/ice/test.sbatch  (see scripts/ice/README.md)"
