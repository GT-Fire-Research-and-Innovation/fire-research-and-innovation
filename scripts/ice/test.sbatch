#!/bin/bash
# GT PACE ICE: run the whole test suite on a CPU node, in the environment setup.sh built.
#   sbatch scripts/ice/test.sbatch                    # every test except the slow ones (as pytest runs by default)
#   SLOW=1 sbatch scripts/ice/test.sbatch             # also the slow SwinUNETR-3D end-to-end runs
#   sbatch -p <cpu partition> --qos=<qos> scripts/ice/test.sbatch    # elsewhere (bash scripts/ice/partitions.sh)
#   PYTEST_ARGS="tests/test_golden.py -x" sbatch scripts/ice/test.sbatch
#SBATCH -J fp-test
#SBATCH -p ice-cpu
#SBATCH --qos=coc-ice
#SBATCH -N 1 --cpus-per-task=8 --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/ice/env.sh
echo "code: $(git log --oneline -1)  node: $(hostname)"
python -c "import sys, torch, monai, numpy; print('python', sys.version.split()[0], '| torch', torch.__version__,
'| monai', monai.__version__, '| numpy', numpy.__version__)"
MARK=()
[ -n "${SLOW:-}" ] && MARK=(-m "slow or not slow")
python -m pytest -q ${MARK[@]+"${MARK[@]}"} ${PYTEST_ARGS:-}
