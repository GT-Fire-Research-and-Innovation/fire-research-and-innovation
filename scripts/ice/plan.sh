#!/usr/bin/env bash
# Queue a whole run on PACE ICE from the login node, in order:
#   1. the daily data root, if $DATA_ROOT has none yet (data.sbatch)
#   2. a smoke test: one training step on a GPU with the real data (check data --train-step)
#   3. CHAIN training jobs of one run, each continuing the last through --resume (train.sbatch)
# If the smoke test fails Slurm cancels the training jobs (DependencyNeverSatisfied).
#   NAME=swin_daily bash scripts/ice/plan.sh
#   NAME=swin_long CHAIN=4 EXTRA="--epochs 100" bash scripts/ice/plan.sh
#   NAME=swin_daily PARTITION=coc-gpu,ice-gpu,pace-gpu bash scripts/ice/plan.sh   # first free GPU of the three
#   NAME=swin_h100 PARTITION=ice-gpu GPU_TYPE=h100 bash scripts/ice/plan.sh
# Everything train.sbatch reads (MODEL, BATCH, LR, TS, IT, EXTRA, ...) passes through.
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
mkdir -p logs
export NAME TS DATA_ROOT FIRE_WORK FIREPRED_ENV

# A partition that refuses the jobs stops here, on the login node, instead of failing later in the queue.
GPU_PARTS=$(usable_partitions "$PARTITION" gpu)
[ -n "$GPU_PARTS" ] || { echo "!! no GPU partition of '$PARTITION' accepts jobs under QOS $QOS." \
  "See bash scripts/ice/partitions.sh, then set PARTITION=... QOS=..."; exit 1; }
echo "GPU jobs: -p $GPU_PARTS ${GPU_OPTS[*]}"

DEP=()
if [ ! -f "$DATA_ROOT/window_config.json" ]; then
  CPU_PARTS=$(usable_partitions "$CPU_PARTITION" cpu)
  [ -n "$CPU_PARTS" ] || { echo "!! no CPU partition of '$CPU_PARTITION' accepts jobs under QOS $CPU_QOS." \
    "See bash scripts/ice/partitions.sh, then set CPU_PARTITION=... CPU_QOS=..."; exit 1; }
  D=$(sbatch --parsable -p "$CPU_PARTS" "${CPU_OPTS[@]}" scripts/ice/data.sbatch)
  DEP=(--dependency=afterok:$D)
  echo "data: $D on $CPU_PARTS -> $DATA_ROOT"
fi

S=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} -J fp-smoke -p "$GPU_PARTS" "${GPU_OPTS[@]}" \
      -N 1 -c 8 --mem=64G -t 0:45:00 -o logs/%x-%j.out \
      --wrap "source scripts/ice/env.sh && echo \"code: \$(git log --oneline -1)  python: \$(python -V 2>&1)\" && \
              python -m firepred.cli.check data -m ${MODEL:-swinunetr3d} --data-root $DATA_ROOT -ts $TS \
              --train-step")
echo "smoke test: $S"

PREV=$S
KIND=afterok
IDS=()
for k in $(seq 1 "$CHAIN"); do
  J=$(sbatch --parsable --dependency=$KIND:$PREV -J "fp-$NAME" \
        -p "$GPU_PARTS" "${GPU_OPTS[@]}" scripts/ice/train.sbatch)
  IDS+=("$J"); PREV=$J; KIND=afterany     # later parts start when the previous one ends, however it ended
done
echo "training $NAME: ${IDS[*]}  (state: $RUNS/$NAME.state)"
echo "Watch: bash scripts/ice/status.sh   Cancel all: scancel -u \$USER"
