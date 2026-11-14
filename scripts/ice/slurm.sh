# Slurm settings and checks shared by the PACE ICE scripts. Sourced by plan.sh, partitions.sh and
# submit-time helpers, never run on its own. Override any of these by exporting it first:
#
#   PARTITION      GPU partition(s) for training and the smoke test        (ice-gpu)
#                  one name, or a comma list such as ice-gpu,coe-gpu: Slurm then starts
#                  each job on whichever listed partition can run it first
#   CPU_PARTITION  CPU partition(s) for the data and test jobs              (ice-cpu)
#   QOS            quality of service of the GPU jobs, e.g. coc-ice         (your Slurm default)
#   CPU_QOS        quality of service of the CPU jobs                       ($QOS)
#   ACCOUNT        Slurm account (sbatch -A); empty uses your default        (empty)
#   GPU_TYPE       one GPU model, e.g. h100, h200, a100 (bash scripts/ice/partitions.sh lists them);
#                  empty takes any GPU                                       (empty)
#
# QOS=default (or QOS='') uses your Slurm default without passing --qos.
# CPU_QOS supports the same values; when unset it follows QOS.
# Which of these you may use: bash scripts/ice/partitions.sh

PARTITION="${PARTITION:-ice-gpu}"
CPU_PARTITION="${CPU_PARTITION:-ice-cpu}"
QOS="${QOS:-}"
CPU_QOS="${CPU_QOS-$QOS}"
[ "$QOS" = default ] && QOS=""
[ "$CPU_QOS" = default ] && CPU_QOS=""
ACCOUNT="${ACCOUNT:-}"
GPU_TYPE="${GPU_TYPE:-}"
GRES="gpu:${GPU_TYPE:+$GPU_TYPE:}1"

# sbatch options the GPU and the CPU jobs get besides their partition.
GPU_OPTS=(--gres="$GRES")
CPU_OPTS=()
[ -n "$QOS" ] && GPU_OPTS+=(--qos="$QOS")
[ -n "$CPU_QOS" ] && CPU_OPTS+=(--qos="$CPU_QOS")
if [ -n "$ACCOUNT" ]; then GPU_OPTS+=(-A "$ACCOUNT"); CPU_OPTS+=(-A "$ACCOUNT"); fi

# slurm_accepts PART QOS KIND: does Slurm accept a 10-minute job of KIND (gpu or cpu) on PART under QOS
# (empty QOS: your default)? sbatch --test-only checks partition, account, QOS and GPU type without
# submitting anything. Returns 0 if accepted; SLURM_REASON then holds the estimated start, else the error.
slurm_accepts() {
  local part=$1 qos=$2 kind=$3 args=(-p "$part" -t 0:10:00 -N 1 -c 1 --mem=1G)
  [ -n "$qos" ] && [ "$qos" != default ] && args+=(--qos="$qos")
  [ -n "$ACCOUNT" ] && args+=(-A "$ACCOUNT")
  [ "$kind" = gpu ] && args+=(--gres="$GRES")
  local out status=0
  out=$(sbatch --test-only "${args[@]}" --wrap true 2>&1) || status=$?
  SLURM_REASON=$(printf '%s\n' "$out" | grep -v '^\s*$' | tail -1 | sed -e 's/^sbatch: error: //' -e 's/^sbatch: //')
  return $status
}

# usable_partitions LIST KIND: the partitions of the comma list LIST that accept KIND jobs (gpu: under
# $QOS, cpu: under $CPU_QOS), as a comma list (empty if none). Each refused one is reported on stderr
# with Slurm's reason. Without sbatch on the PATH (not on ICE) the list is returned unchecked.
usable_partitions() {
  local list=$1 kind=$2 part keep="" qos=$QOS
  [ "$kind" = cpu ] && qos=$CPU_QOS
  if ! command -v sbatch >/dev/null; then echo "$list"; return 0; fi
  for part in ${list//,/ }; do
    if slurm_accepts "$part" "$qos" "$kind"; then
      keep="${keep:+$keep,}$part"
    else
      echo "!! $part refuses $kind jobs under QOS ${qos:-default}: $SLURM_REASON" >&2
    fi
  done
  echo "$keep"
}
