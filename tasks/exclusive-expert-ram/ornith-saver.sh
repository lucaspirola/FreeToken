#!/usr/bin/env bash
# Ornith with the RAM saver ON, first run after the cross-shard pool fix (dd1e56e): the pool used
# to refuse Ornith ("requires each expert's tensors in one shard"). Two arms on this commit, whole
# model in RAM first as the reference, then the auto pool; same probes (8K-250K, pass 2 of record)
# and the same needles/recall, compared field for field (correctness is the output, not counters).
# Run as a systemd transient unit, never from an agent shell.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
HERE="$PWD/tasks/exclusive-expert-ram"
OUT="$HERE/results/ornith-saver"
mkdir -p "$OUT"
export FT_VENV=/home/lucas/ai/FreeToken/.venv FT_RATIO=1.00
export FT_MODEL="$HOME/ai/models/Ornith-1.5-35B-A3B-NVFP4" FT_NAME=ornith
export FT_EXTRA="--num-tokens 262144 --max-seq-len-override 262144 --served-model-name ornith"
export FT_SIZES="8000 32000 80000 128000 250000" FT_POST_TIMEOUT=7200
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock
flock 9
preflight() {
  [ -z "$(git status --porcelain -- python)" ] || { echo "python/ dirty"; exit 1; }
  systemctl is-active --quiet freetoken-serve && { echo "freetoken-serve is up"; exit 1; }
  systemctl --user list-units --state=active --no-legend 'ft-measure-*' | grep -q . && { echo "ft-measure unit up"; exit 1; }
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$used" = 0 ] || { echo "GPU holds $used MiB"; exit 1; }
  avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  [ "$avail" -ge 22 ] || { echo "MemAvailable $avail GiB"; exit 1; }
  echo "preflight ok: GPU 0 MiB, MemAvailable $avail GiB, code $(git log --oneline -1)"
}
post() { echo "NEEDLES_THINK_MAX_TOKENS=65536 NEEDLES_OUT=$OUT/$1-needles.json python3 $HERE/needles.py 21000 120000; python3 $HERE/recall.py 21000 120000 240000"; }
arm() {  # arm NAME ROWS
  preflight
  echo "=== arm $1 (FT_ROWS=$2) ==="
  FT_ROWS=$2 FT_POST="$(post "$1")" "$HERE/measure.sh" "$1" || echo "$1 measure exit $?"
  journalctl --user -u "ft-measure-$1" -o cat --no-pager | sed 's/\x1b\[[0-9;]*m//g' > "$OUT/$1-journal.txt" || true
  mv "$HERE"/results/$1-* "$OUT/" 2>/dev/null || true
  sleep 30
}
[ -z "${ARMS:-}" ] || [[ " $ARMS " == *" whole "* ]] && arm ornith-sv-whole 0
[ -z "${ARMS:-}" ] || [[ " $ARMS " == *" mirror "* ]] && arm ornith-sv-mirror -1
PY=/home/lucas/ai/FreeToken/.venv/bin/python
"$PY" "$HERE/compare_needles.py" "$OUT" ornith-sv-whole ornith-sv-mirror > "$OUT/needles-compare.txt" 2>&1 || true
tail -3 "$OUT/needles-compare.txt"
echo "saver done"
