#!/usr/bin/env bash
# Local acceptance of the measured prefill headroom (reorg-headroom, 82207c8) on the owner's
# RTX 5080 (WSL), the machine whose numbers count:
#   1. checkpoint1.sh ck4 -- Nemotron checkpoint re-pass (the growable-KV path is checkpointed):
#      ck4-whole, ck4-mirror-1m, ck4-mirror, ck4-whole-close, needles/recall vs the reference.
#   2. R-S12a: Ornith NVFP4 whole model, ratio 1.00, 8K/32K/80K/128K, two passes, twice
#      (a, b: the bimodality needs more than one draw), then the RAM-saver arm.
# Waits for a quiet host first: no rustc/cargo, MemAvailable >= 23 GiB [agent practice:
# the owner's other campaigns share this host; 2026-09-24 a contended host stalled all arms].
# Run as a systemd transient unit (--setenv=PATH), never from an agent shell.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
HERE="$PWD/tasks/exclusive-expert-ram"
OUT="$HERE/results/ornith-headroom-local"
mkdir -p "$OUT"
quiet() {
  until ! pgrep -x rustc >/dev/null && ! pgrep -x cargo >/dev/null \
        && [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -ge 23 ]; do sleep 60; done
  echo "host quiet at $(date -Is): $(grep MemAvailable /proc/meminfo)"
}
quiet
"$HERE/checkpoint1.sh" ck4 || echo "checkpoint ck4 exit $?"
export FT_VENV=/home/lucas/ai/FreeToken/.venv FT_RATIO=1.00
export FT_MODEL="$HOME/ai/models/Ornith-1.5-35B-A3B-NVFP4" FT_NAME=ornith
export FT_EXTRA="--num-tokens 262144 --max-seq-len-override 262144 --served-model-name ornith"
export FT_SIZES="8000 32000 80000 128000"
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock
for arm in "ornith-hr-whole-a 0" "ornith-hr-whole-b 0" "ornith-hr-saver -1"; do
  set -- $arm
  quiet
  flock 9
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$used" = 0 ] || { echo "GPU holds $used MiB, skipping $1"; flock -u 9; continue; }
  nvidia-smi dmon -s puct -d 1 -o T > "$OUT/$1-dmon.txt" 2>&1 & smi=$!
  FT_ROWS=$2 "$HERE/measure.sh" "$1" || echo "$1 measure exit $?"
  kill $smi 2>/dev/null
  journalctl --user -u ft-measure-$1 -o cat --no-pager | sed 's/\x1b\[[0-9;]*m//g' > "$OUT/$1-journal.txt" || true
  mv "$HERE"/results/$1-* "$OUT/" 2>/dev/null || true
  flock -u 9
  sleep 30
done
echo "headroom-local done"
