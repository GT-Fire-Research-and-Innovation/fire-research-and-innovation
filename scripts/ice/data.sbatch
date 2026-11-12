#!/bin/bash
# GT PACE ICE: build the daily data root and its zone files on a CPU node (compute nodes have internet).
#   sbatch scripts/ice/data.sbatch                                      # convert the processed 6-1 windows
#   DAILY_REPO=SamuelWu318/ts-satfire-daily sbatch scripts/ice/data.sbatch   # or download daily archives
#   DEST_REPO=SamuelWu318/ts-satfire-daily sbatch scripts/ice/data.sbatch    # convert and also upload them
#   SCALE_KM=3 ZONES_ROOT=... TS=6 also apply. Rerunning skips fires already converted and zone files built.
#   Elsewhere: sbatch -p <cpu partition> --qos=<qos> scripts/ice/data.sbatch (plan.sh passes CPU_PARTITION, CPU_QOS).
#SBATCH -J fp-data
#SBATCH -p ice-cpu
#SBATCH -N 1 --cpus-per-task=8 --mem=64G
#SBATCH -t 16:00:00
#SBATCH -o logs/%x-%j.out
# Direct sbatch submissions use your Slurm default QOS; pass --qos=<name> for a named QOS.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/ice/env.sh
echo "code: $(git log --oneline -1)"
TS="${TS:-6}"
SRC_REPO="${SRC_REPO:-SamuelWu318/ts-satfire-processed-6-1}"
if [ -n "${DAILY_REPO:-}" ]; then
  python -m firepred.cli.dataset --window-repo "$DAILY_REPO" --work-dir "$FIRE_WORK/tmp" \
         download -ts 6 -it 1 --data-root "$DATA_ROOT"
else
  # One fire on disk at a time; the daily root keeps the source's statistics (window_config.json).
  python -m firepred.cli.dataset --window-repo "$SRC_REPO" --work-dir "$FIRE_WORK/tmp" \
         convert-daily -ts 6 -it 1 --dest-root "$DATA_ROOT" ${DEST_REPO:+--dest-repo "$DEST_REPO"}
fi
# Zone files for -ts $TS (one folder per setting: rebuild with -ts N to train at -ts N).
python -m firepred.cli.dataset zones --data-root "$DATA_ROOT" -ts "$TS" -it 1 \
       --zones-root "${ZONES_ROOT:-$DATA_ROOT/zones}" ${SCALE_KM:+--scale-km "$SCALE_KM"}
echo "Data ready: $DATA_ROOT"
