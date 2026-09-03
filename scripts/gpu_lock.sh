#!/usr/bin/env bash
# Exclusive GPU slot for measurements. Usage: scripts/gpu_lock.sh <command...>
# Blocks until the lock is free (flock), then runs the command. Any agent that
# launches a server, a benchmark, or a timing-sensitive test must wrap it here;
# small CPU-only work never needs it.
set -euo pipefail
LOCK="${FREETOKEN_GPU_LOCK:-${XDG_RUNTIME_DIR:-/tmp}/freetoken-gpu.lock}"
exec flock -w "${FREETOKEN_GPU_LOCK_WAIT:-7200}" "$LOCK" "$@"
