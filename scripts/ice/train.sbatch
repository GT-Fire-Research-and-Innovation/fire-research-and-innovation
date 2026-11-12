#!/bin/bash
# GT PACE ICE: one training run on one GPU, by default of the coc-gpu partition (coc-ice QOS).
# The run state ($RUNS/$NAME.state, --resume) is rewritten every epoch, so after the 16 h limit the
# same command continues where it stopped; plan.sh chains such jobs. A finished run leaves
# $RUNS/$NAME.done and later jobs of the chain exit at once.
#   NAME=swin_daily sbatch scripts/ice/train.sbatch
#   NAME=swin_zones ZONES=1 sbatch scripts/ice/train.sbatch               # one training sample per fire
#   NAME=swin_it2 IT=2 EXTRA="--epochs 50" sbatch scripts/ice/train.sbatch
#   MODEL, BATCH, LR, TS, IT, ZONES_ROOT, EXTRA (any other train flags) also apply.
#   Elsewhere: sbatch -p coc-gpu,ice-gpu,pace-gpu --qos=<qos> [--gres=gpu:h100:1] scripts/ice/train.sbatch
#   (a comma list starts on the first partition with a free GPU; bash scripts/ice/partitions.sh shows yours).
#SBATCH -J fp-train
#SBATCH -p coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH -N 1 --cpus-per-task=8 --mem=64G
#SBATCH -t 16:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/ice/env.sh
: "${NAME:?set NAME, e.g. NAME=swin_daily}"
if [ -f "$RUNS/$NAME.done" ]; then echo "$NAME already finished ($RUNS/$NAME.done)"; exit 0; fi
echo "code: $(git log --oneline -1)  run: $NAME  state: $RUNS/$NAME.state"
echo "partition: ${SLURM_JOB_PARTITION:-?}  node: $(hostname)  python: $(python -V 2>&1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "!! no GPU visible"
ZONES_ARGS=()
[ -n "${ZONES:-}" ] && ZONES_ARGS=(--zones-dir "${ZONES_ROOT:-$DATA_ROOT/zones}")
python -m firepred.cli.train -m "${MODEL:-swinunetr3d}" -mode pred -b "${BATCH:-8}" -lr "${LR:-0.001}" \
       -nh 3 -ed 36 -ts "${TS:-6}" -it "${IT:-1}" --data-root "$DATA_ROOT" --num-workers 6 \
       --resume "$RUNS/$NAME.state" ${ZONES_ARGS[@]+"${ZONES_ARGS[@]}"} ${EXTRA:-}
touch "$RUNS/$NAME.done"
echo "$NAME finished"
