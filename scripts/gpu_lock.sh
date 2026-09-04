#!/usr/bin/env bash
# Exclusive slot for anything that loads the model or touches the GPU.
# Usage: scripts/gpu_lock.sh <command...>
#  - flock serializes jobs (FREETOKEN_GPU_LOCK, wait FREETOKEN_GPU_LOCK_WAIT s, default 2h)
#  - the wrapped job is capped at FREETOKEN_GPU_LOCK_MAX_HOLD s (default 4h) so a dead or
#    unattended holder cannot block the queue forever
#  - the job and its children get oom_score_adj=1000 so the kernel OOM killer takes THEM,
#    not the terminal/Claude session (2026-09-04: an OOM sweep killed the whole tmux scope)
#  - on exit every FreeToken worker in the job's process group is killed (spawn workers hold
#    ~18 GB of shmem expert banks and outlive `pkill -f "ft serve"`)
#  - refuses to start when MemAvailable is below FREETOKEN_GPU_LOCK_MIN_AVAIL_GB (default 22)
set -euo pipefail
LOCK="${FREETOKEN_GPU_LOCK:-${XDG_RUNTIME_DIR:-/tmp}/freetoken-gpu.lock}"
WAIT="${FREETOKEN_GPU_LOCK_WAIT:-7200}"
MAXHOLD="${FREETOKEN_GPU_LOCK_MAX_HOLD:-14400}"
MINAVAIL="${FREETOKEN_GPU_LOCK_MIN_AVAIL_GB:-22}"
exec 9>"$LOCK"
flock -w "$WAIT" 9 || { echo "gpu_lock: timed out after ${WAIT}s waiting for $LOCK" >&2; exit 75; }
avail_gb=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
if [ "$avail_gb" -lt "$MINAVAIL" ]; then
  echo "gpu_lock: only ${avail_gb} GiB host RAM available (< ${MINAVAIL}); refusing to start" >&2
  exit 76
fi
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true
set -m
cleanup() {
  pkill -9 -g $$ -f "/home/lucas/ai/FreeToken/.venv/bin/python" 2>/dev/null || true
  pkill -9 -g $$ 2>/dev/null || true
}
trap cleanup EXIT INT TERM
timeout --signal=TERM --kill-after=60 "$MAXHOLD" "$@"
