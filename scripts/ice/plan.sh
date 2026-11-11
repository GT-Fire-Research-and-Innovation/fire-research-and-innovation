#!/usr/bin/env bash
# Queue a whole run on PACE ICE from the login node, in order:
#   1. the daily data root, if $DATA_ROOT has none yet (data.sbatch)
#   2. a smoke test: one training step on a GPU with the real data (check data --train-step)
#   3. CHAIN training jobs of one run, each continuing the last through --resume (train.sbatch)
# If the smoke test fails Slurm cancels the training jobs (DependencyNeverSatisfied).
#   NAME=swin_daily bash scripts/ice/plan.sh
#   NAME=swin_long CHAIN=4 EXTRA="--epochs 100" bash scripts/ice/plan.sh
# Everything train.sbatch reads (MODEL, BATCH, LR, TS, IT, EXTRA, ...) passes through;
# PARTITION / CPU_PARTITION / QOS replace coc-gpu / ice-cpu / coc-ice for every job; they are checked
# against sinfo and your Slurm associations before anything is submitted.
# Progress: bash scripts/ice/status.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/ice/env.sh
: "${NAME:?set NAME, e.g. NAME=swin_daily}"
CHAIN="${CHAIN:-3}"
TS="${TS:-6}"
mkdir -p logs
export NAME TS DATA_ROOT FIRE_WORK FIREPRED_ENV
PARTITION="${PARTITION:-coc-gpu}"
CPU_PARTITION="${CPU_PARTITION:-ice-cpu}"
QOS="${QOS:-coc-ice}"

# A partition that does not exist stops here, on the login node, instead of failing later in the queue.
# A QOS missing from your associations is only reported: some come with the partition instead.
if command -v sinfo >/dev/null; then
  for part in "$PARTITION" "$CPU_PARTITION"; do
    sinfo -h -p "$part" -o %P 2>/dev/null | grep -q . || {
      echo "!! partition $part not found. Yours: sinfo -s. Then set PARTITION=... / CPU_PARTITION=..."; exit 1; }
  done
fi
if command -v sacctmgr >/dev/null && ! sacctmgr -nP show assoc user="$USER" format=qos 2>/dev/null \
     | tr ',' '\n' | grep -qx "$QOS"; then
  echo "note: QOS $QOS is not listed for you (sacctmgr show assoc user=\$USER format=account,partition,qos%40);" \
       "submitting anyway, set QOS=... if jobs are rejected"
fi

DEP=()
if [ ! -f "$DATA_ROOT/window_config.json" ]; then
  D=$(sbatch --parsable -p "$CPU_PARTITION" --qos="$QOS" scripts/ice/data.sbatch)
  DEP=(--dependency=afterok:$D)
  echo "data: $D -> $DATA_ROOT"
fi

S=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} -J fp-smoke -p "$PARTITION" --qos="$QOS" --gres=gpu:1 \
      -N 1 -c 8 --mem=64G -t 0:45:00 -o logs/%x-%j.out \
      --wrap "source scripts/ice/env.sh && echo \"code: \$(git log --oneline -1)\" && \
              python -m firepred.cli.check data -m ${MODEL:-swinunetr3d} --data-root $DATA_ROOT -ts $TS \
              --train-step")
echo "smoke test: $S"

PREV=$S
KIND=afterok
IDS=()
for k in $(seq 1 "$CHAIN"); do
  J=$(sbatch --parsable --dependency=$KIND:$PREV -J "fp-$NAME" \
        -p "$PARTITION" --qos="$QOS" scripts/ice/train.sbatch)
  IDS+=("$J"); PREV=$J; KIND=afterany     # later parts start when the previous one ends, however it ended
done
echo "training $NAME: ${IDS[*]}  (state: $RUNS/$NAME.state)"
echo "Watch: bash scripts/ice/status.sh   Cancel all: scancel -u \$USER"
