# Shared settings for the PACE ICE scripts. Sourced by every script here, never run on its own.
# Override any of these by exporting it before sbatch / bash (sbatch passes your environment on).
#
#   FIREPRED_ENV  the Python environment setup.sh builds        (~/scratch/envs/firepred)
#   FIRE_WORK     data, run states and W&B runs                 (~/scratch/firepred)
#   DATA_ROOT     the daily data root training reads             ($FIRE_WORK/data-daily)
#
# Everything lives in ~/scratch: the home directory is small (check yours with pace-quota).

FIREPRED_ENV="${FIREPRED_ENV:-$HOME/scratch/envs/firepred}"
FIRE_WORK="${FIRE_WORK:-$HOME/scratch/firepred}"
DATA_ROOT="${DATA_ROOT:-$FIRE_WORK/data-daily}"
RUNS="$FIRE_WORK/runs"                                   # <name>.state (--resume) and <name>.done

module purge 2>/dev/null || true
export PATH="$FIREPRED_ENV/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/scratch/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/scratch/.cache/pip}"
export WANDB_DIR="${WANDB_DIR:-$FIRE_WORK}"              # runs and checkpoints: $WANDB_DIR/wandb/<run>/files
export PYTHONUNBUFFERED=1                                # log lines appear as they happen
mkdir -p "$FIRE_WORK" "$RUNS"
