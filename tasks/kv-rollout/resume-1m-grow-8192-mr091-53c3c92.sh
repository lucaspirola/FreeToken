#!/usr/bin/env bash
# Recreate the reviewed 1M logical-context, 8192-step growable-KV candidate at memory ratio 0.91.
# This file contains no credentials. It never stops or replaces another service.
set -euo pipefail

UNIT="freetoken-serve.service"
HOST="127.0.0.1"
PORT="1919"
PROJECT="/home/lucas/ai/FreeToken"
UV="/home/lucas/.local/bin/uv"
MODEL="/home/lucas/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
SPILL_DIR="/home/lucas/.cache/freetoken/spill"
# Approved production identity from commit 53c3c92. Docs/tests-only commits may advance
# HEAD; source or launch-dependency changes must be explicitly reviewed again.
APPROVED_BASELINE_COMMIT="53c3c92"
APPROVED_PYTHON_TREE="f706c2450376b1c4b69d1cb71d3ecfcc49e4a853"
APPROVED_PYPROJECT_BLOB="649ba5820a8d9ca96d9d06117390406da9b32edd"
APPROVED_SETUP_BLOB="fd1bfe335c1498ba0feb3db119e06a7653a50a0d"
APPROVED_UV_LOCK_SHA256="3543675f6ce3a3c6a66baee8ff085d530eac87949bd65429841305b8f9849315"
# The successful rollout used about 20 GiB host residency once steady.  Requiring that
# amount plus the profile's 8 GiB reserve prevents a new launch from consuming the host's
# last working memory.  This is an admission check, not an OOM guarantee.
MIN_MEM_AVAILABLE_GIB=28
# The spill store retains at most 50 GiB; leave 10 GiB filesystem headroom beyond that.
MIN_SPILL_FREE_GIB=60

dry_run=false
case "${1:-}" in
  "") ;;
  --dry-run) dry_run=true ;;
  *) echo "usage: $0 [--dry-run]" >&2; exit 2 ;;
esac

for path in "$PROJECT" "$UV" "$MODEL" "$SPILL_DIR"; do
  if [[ ! -e "$path" ]]; then
    echo "refusing resume: required path is absent: $path" >&2
    exit 1
  fi
done

actual_python_tree="$(git -C "$PROJECT" rev-parse HEAD:python/freetoken 2>/dev/null || true)"
actual_pyproject_blob="$(git -C "$PROJECT" rev-parse HEAD:pyproject.toml 2>/dev/null || true)"
actual_setup_blob="$(git -C "$PROJECT" rev-parse HEAD:setup.py 2>/dev/null || true)"
if [[ "$actual_python_tree" != "$APPROVED_PYTHON_TREE" || "$actual_pyproject_blob" != "$APPROVED_PYPROJECT_BLOB" || "$actual_setup_blob" != "$APPROVED_SETUP_BLOB" ]]; then
  echo "refusing resume: production identity differs from approved baseline $APPROVED_BASELINE_COMMIT" >&2
  echo "  python tree: ${actual_python_tree:-unavailable}" >&2
  echo "  pyproject blob: ${actual_pyproject_blob:-unavailable}" >&2
  echo "  setup blob: ${actual_setup_blob:-unavailable}" >&2
  exit 1
fi
production_paths=(python/freetoken pyproject.toml setup.py)
if ! git -C "$PROJECT" diff --quiet -- "${production_paths[@]}" || ! git -C "$PROJECT" diff --cached --quiet -- "${production_paths[@]}"; then
  echo "refusing resume: staged or unstaged production changes are present" >&2
  exit 1
fi
untracked_production="$(git -C "$PROJECT" ls-files --others --exclude-standard -- python/freetoken)"
if [[ -n "$untracked_production" ]]; then
  echo "refusing resume: untracked production files are present under python/freetoken" >&2
  exit 1
fi
if [[ ! -f "$PROJECT/uv.lock" ]]; then
  echo "refusing resume: required ignored dependency lock is absent: $PROJECT/uv.lock" >&2
  exit 1
fi
actual_uv_lock_sha256="$(sha256sum "$PROJECT/uv.lock" | awk '{print $1}')"
if [[ "$actual_uv_lock_sha256" != "$APPROVED_UV_LOCK_SHA256" ]]; then
  echo "refusing resume: uv.lock SHA-256 differs from approved baseline" >&2
  echo "  uv.lock SHA-256: $actual_uv_lock_sha256" >&2
  exit 1
fi

args=(
  "$UV" run --locked --directory "$PROJECT" ft serve
  --model "$MODEL"
  --host "$HOST" --port "$PORT"
  --max-running-requests 1
  --num-tokens 1048576
  --max-seq-len-override 1048576
  --kv-cache-dtype q8_0
  --attention-backend triton
  --moe-backend offload --moe-cache-auto --moe-cache-policy lfu
  --memory-ratio 0.91 --max-prefill-length 4096 --kv-grow-step-tokens 8192
  --linear-state-slots 6
  --host-ram-reserve-gb 8
  --session-spill-ram-gb 1 --session-spill-disk-gb 50 --session-spill-limit-gb 50
  --session-spill-dir "$SPILL_DIR"
  --enable-cache-report --cuda-memory-telemetry
  --served-model-name nemotron-3.5-lightning
  --served-model-alias nemotron-3.5-lightning-judge
  --served-model-alias nemotron-3.5-lightning-collect
  --reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder
  --force-nonempty-content --max-output-tokens 16384
  --trace-dir /home/lucas/.cache/freetoken/trace
  --hidden-states-dir /home/lucas/.cache/freetoken/hidden-states
  --hidden-states-max-tokens 4096
  --pooled-sink-dir /home/lucas/.cache/freetoken/pooled-sink
  --pin-prefix-min-tokens 1024
)

if "$dry_run"; then
  printf 'resume command:'
  printf ' %q' systemd-run --user --unit="$UNIT" --property=OOMScoreAdjust=1000 \
    --setenv=PATH=/home/lucas/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib \
    --setenv=TVM_FFI_CUDA_ARCH_LIST=12.0 "${args[@]}"
  printf '\n'
  exit 0
fi

if systemctl --user is-active --quiet "$UNIT"; then
  echo "refusing resume: $UNIT is already active" >&2
  exit 1
fi
if ! listening_sockets="$(ss -H -ltn "sport = :$PORT")"; then
  echo "refusing resume: cannot inspect listening sockets" >&2
  exit 1
fi
if [[ -n "$listening_sockets" ]]; then
  echo "refusing resume: TCP port $PORT is already occupied" >&2
  exit 1
fi

mem_available_kib="$({ awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo; })"
if [[ ! "$mem_available_kib" =~ ^[0-9]+$ ]]; then
  echo "refusing resume: cannot read MemAvailable from /proc/meminfo" >&2
  exit 1
fi
min_mem_kib=$((MIN_MEM_AVAILABLE_GIB * 1024 * 1024))
if (( mem_available_kib < min_mem_kib )); then
  echo "refusing resume: MemAvailable is ${mem_available_kib} KiB; need ${MIN_MEM_AVAILABLE_GIB} GiB" >&2
  exit 1
fi

spill_free_kib="$(df -Pk "$SPILL_DIR" | awk 'NR == 2 { print $4 }')"
if [[ ! "$spill_free_kib" =~ ^[0-9]+$ ]]; then
  echo "refusing resume: cannot determine free space for $SPILL_DIR" >&2
  exit 1
fi
min_spill_kib=$((MIN_SPILL_FREE_GIB * 1024 * 1024))
if (( spill_free_kib < min_spill_kib )); then
  echo "refusing resume: spill volume has ${spill_free_kib} KiB free; need ${MIN_SPILL_FREE_GIB} GiB" >&2
  exit 1
fi

# A previously stopped transient unit can remain in failed state after a graceful stop.
systemctl --user reset-failed "$UNIT" || true
exec systemd-run --user --unit="$UNIT" --property=OOMScoreAdjust=1000 \
  --setenv=PATH=/home/lucas/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib \
  --setenv=TVM_FFI_CUDA_ARCH_LIST=12.0 "${args[@]}"
