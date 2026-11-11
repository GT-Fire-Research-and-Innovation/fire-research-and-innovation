#!/usr/bin/env bash
# One look at everything on PACE ICE:   bash scripts/ice/status.sh
cd "$(dirname "$0")/../.."
source scripts/ice/env.sh >/dev/null 2>&1
echo "=== jobs"
squeue -u "$USER" -o "%.10i %.20j %.9T %.10M %.12l %.24R"
echo
echo "=== runs ($RUNS)"
for state in "$RUNS"/*.state; do
  [ -e "$state" ] || { echo "(none yet)"; break; }
  name=$(basename "$state" .state)
  done_mark=""; [ -f "$RUNS/$name.done" ] && done_mark="  finished"
  echo "$name: state saved $(date -r "$state" '+%m-%d %H:%M')$done_mark"
done
echo
echo "=== latest logs"
for log in $(ls -t logs/*.out 2>/dev/null | head -6); do
  last=$(grep -h "^Epoch [0-9]*:" "$log" | tail -1)
  err=$(grep -m1 -h "Traceback\|Error\|CANCELLED\|DUE TO TIME LIMIT" "$log")
  echo "$log: ${last:-$(tail -1 "$log" | cut -c1-120)}"
  [ -n "$err" ] && echo "    !! $err"
done
