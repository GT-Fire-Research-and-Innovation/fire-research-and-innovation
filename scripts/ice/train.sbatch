#!/bin/bash
# GT PACE ICE: one training run on one GPU, by default of the ice-gpu partition (your Slurm default QOS).
# The run state ($RUNS/$NAME.state, --resume) is rewritten every epoch, so after the 16 h limit the
# same command continues where it stopped; plan.sh chains such jobs. A finished run leaves
# $RUNS/$NAME.done and later jobs of the chain exit at once.
#   NAME=swin_daily sbatch scripts/ice/train.sbatch
#   NAME=swin_it2 IT=2 EXTRA="--epochs 50" sbatch scripts/ice/train.sbatch
#   MODEL, BATCH, LR, EXTRA (any other train flags) also apply. Daily archives here give the windows they
#   were converted with (6 days, 1 day apart), so TS=6 IT=1 (the defaults) must match window_config.json.
#   Elsewhere: sbatch -p ice-gpu,coe-gpu [--qos=<qos>] [--gres=gpu:h100:1] scripts/ice/train.sbatch
#   (a comma list starts on the first partition with a free GPU; bash scripts/ice/partitions.sh shows yours).
#SBATCH -J fp-train
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1 --cpus-per-task=8 --mem=64G
#SBATCH -t 16:00:00
#SBATCH -o logs/%x-%j.out
# Direct sbatch submissions use your Slurm default QOS; pass --qos=<name> for a named QOS.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/ice/env.sh
: "${NAME:?set NAME, e.g. NAME=swin_daily}"
if [ -f "$RUNS/$NAME.done" ]; then echo "$NAME already finished ($RUNS/$NAME.done)"; exit 0; fi
echo "code: $(git log --oneline -1)  run: $NAME  state: $RUNS/$NAME.state"
echo "partition: ${SLURM_JOB_PARTITION:-?}  node: $(hostname)  python: $(python -V 2>&1)"
# Check the GPU before training: torch must see one (else it would train on the CPU, many times slower)
# and have kernels for it (a newer model such as a Blackwell RTX Pro 6000 or B200 fails at its first
# operation with "no kernel image is available").
if ! python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print(f"!! torch {torch.__version__} sees no GPU on this node")
    sys.exit(1)
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) on {name}, compute capability {major}.{minor}; "
      f"built for {' '.join(torch.cuda.get_arch_list())}")
try:
    torch.cat((torch.ones(2, device="cuda"), torch.zeros(2, device="cuda"))).sum().item()
except RuntimeError as error:
    print(f"!! this torch build cannot run on {name}: {str(error).splitlines()[0]}")
    sys.exit(1)
PY
then
  echo "   Stopping before training. Pick a GPU this torch supports, e.g. GPU_TYPE=h100"
  echo "   (or a100, h200, l40s) with plan.sh, or --gres=gpu:h100:1 with sbatch directly."
  exit 1
fi
python -m firepred.cli.train -m "${MODEL:-swinunetr3d}" -mode pred -b "${BATCH:-8}" -lr "${LR:-0.001}" \
       -nh 3 -ed 36 -ts "${TS:-6}" -it "${IT:-1}" --data-root "$DATA_ROOT" --num-workers 6 \
       --resume "$RUNS/$NAME.state" ${EXTRA:-}
touch "$RUNS/$NAME.done"
echo "$NAME finished"
