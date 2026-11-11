#!/bin/bash
# GT PACE ICE: one training run on one GPU of the College of Computing's coc-gpu partition (coc-ice QOS).
# The run state ($RUNS/$NAME.state, --resume) is rewritten every epoch, so after the 16 h limit the
# same command continues where it stopped; plan.sh chains such jobs. A finished run leaves
# $RUNS/$NAME.done and later jobs of the chain exit at once.
#   NAME=swin_daily sbatch scripts/ice/train.sbatch
#   NAME=swin_it2 IT=2 EXTRA="--epochs 50" sbatch scripts/ice/train.sbatch
#   MODEL, BATCH, LR, EXTRA (any other train flags) also apply. Daily archives here give the windows they
#   were converted with (6 days, 1 day apart), so TS=6 IT=1 (the defaults) must match window_config.json.
#   Other access? sbatch -p ice-gpu --qos=<yours> scripts/ice/train.sbatch  (sacctmgr show assoc user=$USER)
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "!! no GPU visible"
python -m firepred.cli.train -m "${MODEL:-swinunetr3d}" -mode pred -b "${BATCH:-8}" -lr "${LR:-0.001}" \
       -nh 3 -ed 36 -ts "${TS:-6}" -it "${IT:-1}" --data-root "$DATA_ROOT" --num-workers 6 \
       --resume "$RUNS/$NAME.state" ${EXTRA:-}
touch "$RUNS/$NAME.done"
echo "$NAME finished"
