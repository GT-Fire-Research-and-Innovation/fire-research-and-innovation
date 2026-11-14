#!/usr/bin/env bash
# What can you run on PACE ICE? Lists partitions with their GPUs, free nodes and time limit, and checks
# whether Slurm would accept a job from you on each (sbatch --test-only: nothing is submitted).
#   bash scripts/ice/partitions.sh                              # every partition
#   bash scripts/ice/partitions.sh ice-gpu coe-gpu ice-cpu      # only these
#   QOS=coe-ice ACCOUNT=... GPU_TYPE=h100 bash scripts/ice/partitions.sh coe-gpu
# A partition is tried with QOS (when set), then with your default QOS, then with each other QOS of your
# Slurm associations; the QOS column names the first one accepted, to pass as QOS=... to plan.sh.
# Then: PARTITION=<gpu partitions, comma-separated> QOS=... CPU_PARTITION=... CPU_QOS=... bash scripts/ice/plan.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/ice/slurm.sh
command -v sinfo >/dev/null || { echo "No Slurm here: run this on an ICE login node (login-ice.pace.gatech.edu)"; exit 1; }
USER_QOS=$(sacctmgr -nP show assoc user="$USER" format=qos 2>/dev/null | tr ',' '\n' | grep -v '^$' | sort -u | tr '\n' ' ')

echo "=== your Slurm associations (account | partition | QOS; an empty partition means any)"
sacctmgr -nP show assoc user="$USER" format=account,partition,qos 2>/dev/null | sed 's/^/  /' \
  || echo "  (sacctmgr unavailable)"
echo
if [ $# -gt 0 ]; then PARTS=("$@"); else
  PARTS=($(sinfo -h -o %R | sort -u)); fi
printf '%-14s %-6s %-12s %-14s %-11s %s\n' PARTITION ACCESS QOS "NODES A/I/O/T" TIMELIMIT GPUS
for part in "${PARTS[@]}"; do
  info=$(sinfo -h -p "$part" -o '%F|%l' 2>/dev/null | head -1)
  if [ -z "$info" ]; then printf '%-14s %s\n' "$part" "not found (sinfo -s lists the partitions)"; continue; fi
  gpus=$(sinfo -h -p "$part" -o '%G' 2>/dev/null | tr ',' '\n' | grep '^gpu' | sed 's/(.*//' | sort -u | tr '\n' ' ')
  kind=cpu; [ -n "$gpus" ] && kind=gpu
  access=no; used="-"; reason=""
  for qos in ${QOS:+"$QOS"} "" $USER_QOS; do
    if slurm_accepts "$part" "$qos" "$kind"; then access=yes; used=${qos:-default}; break; fi
    [ -z "$reason" ] && reason=$SLURM_REASON
  done
  printf '%-14s %-6s %-12s %-14s %-11s %s\n' "$part" "$access" "$used" "${info%%|*}" "${info##*|}" "${gpus:--}"
  [ "$access" = no ] && echo "    ($reason)"
done
echo
echo "NODES A/I/O/T: allocated / idle / other (down, drained) / total nodes. GPUS: model:count per node."
echo "QOS=default (or CPU_QOS=default) means omit --qos and use your Slurm default, not a QOS named default."
echo "GPU_TYPE=<model> (e.g. h100) asks for one model; PARTITION=a,b,c takes the first free one of several."
echo "Busy now: squeue -p <partition> -t PENDING | wc -l   Your jobs: squeue -u \$USER"
