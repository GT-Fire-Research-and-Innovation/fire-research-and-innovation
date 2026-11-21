#!/usr/bin/env bash
# Queue a whole run on PACE ICE from the login node, in order:
#   1. the daily data root, if $DATA_ROOT has none yet (data.sbatch)
#   2. optional, SMOKE=1: one training step on a GPU with the real data (check data --train-step)
#   3. CHAIN training jobs of one run, each continuing the last through --resume (train.sbatch)
# Training waits for successful data preparation and, when enabled, the smoke test.
#   NAME=swin_daily bash scripts/ice/plan.sh
#   NAME=swin_daily SMOKE=1 bash scripts/ice/plan.sh   # optional GPU check before training
#   NAME=swin_long CHAIN=4 EXTRA="--epochs 100" bash scripts/ice/plan.sh
#   NAME=swin_daily PARTITION=ice-gpu,coe-gpu bash scripts/ice/plan.sh   # first free GPU of the two
#   NAME=swin_h100 PARTITION=ice-gpu GPU_TYPE=h100 bash scripts/ice/plan.sh
#   NAME=swin_daily TEST=10 bash scripts/ice/plan.sh   # test F1/IoU every 10 epochs; TEST=0: only at the end
# Everything train.sbatch reads (MODEL, BATCH, LR, TS, IT, TEST, EXTRA, ...) passes through.
# TEST (default 5) also scores the test fires every TEST epochs with the current weights, for W&B and the log.
# Where jobs run: PARTITION, CPU_PARTITION, QOS, CPU_QOS, ACCOUNT, GPU_TYPE (scripts/ice/slurm.sh); every listed
# partition is checked with sbatch --test-only first, and one that refuses the jobs is left out.
# Which partitions you can use: bash scripts/ice/partitions.sh   Progress: bash scripts/ice/status.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/ice/env.sh
source scripts/ice/slurm.sh
: "${NAME:?set NAME, e.g. NAME=swin_daily}"
CHAIN="${CHAIN:-3}"
TS="${TS:-6}"
TEST="${TEST:-5}"
# Checked here too, so a bad value stops on the login node instead of after a wait in the queue.
[[ "$TEST" =~ ^[0-9]+$ ]] || { echo "!! TEST must be a whole number of epochs (0 = test only at the end), got '$TEST'"; exit 1; }
mkdir -p logs
export NAME TS TEST DATA_ROOT FIRE_WORK FIREPRED_ENV

# A partition that refuses the jobs stops here, on the login node, instead of failing later in the queue.
GPU_PARTS=$(usable_partitions "$PARTITION" gpu)
[ -n "$GPU_PARTS" ] || { echo "!! no GPU partition of '$PARTITION' accepts jobs under QOS ${QOS:-default}." \
  "See bash scripts/ice/partitions.sh, then set PARTITION=... QOS=..."; exit 1; }
echo "GPU jobs: -p $GPU_PARTS ${GPU_OPTS[*]}"

DEP=()
if [ ! -f "$DATA_ROOT/window_config.json" ]; then
  CPU_PARTS=$(usable_partitions "$CPU_PARTITION" cpu)
  [ -n "$CPU_PARTS" ] || { echo "!! no CPU partition of '$CPU_PARTITION' accepts jobs under QOS ${CPU_QOS:-default}." \
    "See bash scripts/ice/partitions.sh, then set CPU_PARTITION=... CPU_QOS=..."; exit 1; }
  D=$(sbatch --parsable -p "$CPU_PARTS" ${CPU_OPTS[@]+"${CPU_OPTS[@]}"} scripts/ice/data.sbatch)
  DEP=(--dependency=afterok:$D)
  echo "data: $D on $CPU_PARTS -> $DATA_ROOT"
fi

if [ "${SMOKE:-0}" = 1 ]; then
  S=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} -J fp-smoke -p "$GPU_PARTS" "${GPU_OPTS[@]}" \
        -N 1 -c 8 --mem=64G -t 0:45:00 -o logs/%x-%j.out \
        --wrap "source scripts/ice/env.sh && echo \"code: \$(git log --oneline -1)  python: \$(python -V 2>&1)\" && \
                python -m firepred.cli.check data -m ${MODEL:-swinunetr3d} --data-root $DATA_ROOT -ts $TS \
                --train-step")
  echo "smoke test: $S"
  DEP=(--dependency=afterok:$S)
fi

IDS=()
for k in $(seq 1 "$CHAIN"); do
  J=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} -J "fp-$NAME" \
        -p "$GPU_PARTS" "${GPU_OPTS[@]}" scripts/ice/train.sbatch)
  IDS+=("$J"); DEP=(--dependency=afterany:$J)  # later parts start when the previous one ends, however it ended
done
echo "training $NAME: ${IDS[*]}  (state: $RUNS/$NAME.state)"
if [ "$TEST" = 0 ]; then echo "test F1/IoU: at the end only (TEST=0)"
else echo "test F1/IoU: every $TEST epochs and at the end (TEST=$TEST; TEST=0 for the end only)"; fi
echo "Watch: bash scripts/ice/status.sh   Cancel all: scancel -u \$USER"
