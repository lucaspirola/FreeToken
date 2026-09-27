#!/usr/bin/env bash
# One server arm on :1920 from worktree WT with replay.py as measure.sh's FT_POST, under the GPU
# host lock (run it as a systemd user unit, never from an agent shell).
#
#   arm.sh WT ARM OUTDIR [ROWS (0 = whole, -1 = mirror; default 0)] [extra FT_ENVS words]
#
# Waits for an idle GPU (<= 32 MiB: the owner's Windows apps can hold ~17 MiB under WSL) and
# MemAvailable >= 23 GiB. Leaves OUTDIR/ARM-replay.json, OUTDIR/ARM-journal.txt and the arm's
# measure.sh results in OUTDIR/ARM/.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WT=$1; ARM=$2; OUT=$3; ROWS=${4:-0}; ENVS="${5:-}"
H=$WT/tasks/exclusive-expert-ram
mkdir -p "$OUT/$ARM"
gpu() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock
flock 9
until [ "$(gpu)" -le 32 ]; do sleep 5; done
until [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -ge 23 ]; do sleep 20; done
echo "start $(date -Is) gpu=$(gpu) MiB $(cut -d' ' -f1-3 /proc/loadavg) $(grep MemAvailable /proc/meminfo | tr -s ' ')"
FT_VENV=/home/lucas/ai/FreeToken/.venv FT_RATIO=1.00 FT_ROWS=$ROWS FT_RESERVE=256 FT_SIZES=8000 \
  FT_POST_TIMEOUT=1200 FT_ENVS="$ENVS" \
  FT_POST="REPLAY_OUT=$OUT/$ARM-replay.json python3 $HERE/replay.py" "$H/measure.sh" "$ARM" || echo "measure exit $?"
journalctl --user -u "ft-measure-$ARM" -o cat --no-pager | sed 's/\x1b\[[0-9;]*m//g' > "$OUT/$ARM-journal.txt" || true
mv "$H"/results/"$ARM"-* "$OUT/$ARM/" 2>/dev/null || true
git -C "$WT" checkout -- tasks/exclusive-expert-ram/results/sweep.tsv 2>/dev/null || true
until [ "$(gpu)" -le 32 ]; do sleep 5; done
echo "done $(date -Is)"
