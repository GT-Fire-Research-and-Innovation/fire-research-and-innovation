#!/usr/bin/env bash
# Queue a whole run on PACE ICE from the login node, in order:
#   1. the daily data root and its zone files, if $DATA_ROOT has none yet (data.sbatch)
#   2. a smoke test: one training step on a GPU with the real data (check data --train-step)
#   3. CHAIN training jobs of one run, each continuing the last through --resume (train.sbatch)
# If the smoke test fails Slurm cancels the training jobs (DependencyNeverSatisfied).
#   NAME=swin_daily bash scripts/ice/plan.sh
#   NAME=swin_zones ZONES=1 CHAIN=4 bash scripts/ice/plan.sh
# Everything train.sbatch reads (MODEL, BATCH, LR, TS, IT, ZONES, EXTRA, ...) passes through;
# PARTITION / CPU_PARTITION / QOS replace coe-gpu / ice-cpu / coe-ice for every job.
# Progress: bash scripts/ice/status.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/ice/env.sh
: "${NAME:?set NAME, e.g. NAME=swin_daily}"
CHAIN="${CHAIN:-3}"
TS="${TS:-6}"
mkdir -p logs
export NAME TS DATA_ROOT FIRE_WORK FIREPRED_ENV

DEP=()
if [ ! -f "$DATA_ROOT/window_config.json" ]; then
  D=$(sbatch --parsable -p "${CPU_PARTITION:-ice-cpu}" --qos="${QOS:-coe-ice}" scripts/ice/data.sbatch)
  DEP=(--dependency=afterok:$D)
  echo "data: $D -> $DATA_ROOT"
fi

ZONES_ARGS=""
[ -n "${ZONES:-}" ] && ZONES_ARGS="--zones-dir ${ZONES_ROOT:-$DATA_ROOT/zones}"
S=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} -J fp-smoke -p "${PARTITION:-coe-gpu}" --qos="${QOS:-coe-ice}" --gres=gpu:1 \
      -N 1 -c 8 --mem=64G -t 0:45:00 -o logs/%x-%j.out \
      --wrap "source scripts/ice/env.sh && echo \"code: \$(git log --oneline -1)\" && \
              python -m firepred.cli.check data -m ${MODEL:-swinunetr3d} --data-root $DATA_ROOT -ts $TS \
              --train-step $ZONES_ARGS")
echo "smoke test: $S"

PREV=$S
KIND=afterok
IDS=()
for k in $(seq 1 "$CHAIN"); do
  J=$(sbatch --parsable --dependency=$KIND:$PREV -J "fp-$NAME" \
        -p "${PARTITION:-coe-gpu}" --qos="${QOS:-coe-ice}" scripts/ice/train.sbatch)
  IDS+=("$J"); PREV=$J; KIND=afterany     # later parts start when the previous one ends, however it ended
done
echo "training $NAME: ${IDS[*]}  (state: $RUNS/$NAME.state)"
echo "Watch: bash scripts/ice/status.sh   Cancel all: scancel -u \$USER"
