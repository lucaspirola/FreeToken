#!/usr/bin/env bash
# Recreate the approved 1M static-KV 4096-prefill candidate with allocator telemetry.
# This file contains no credentials. It never stops or replaces another service.
set -euo pipefail

UNIT="freetoken-serve.service"
HOST="127.0.0.1"
PORT="1919"
PROJECT="/home/lucas/ai/FreeToken"
UV="/home/lucas/.local/bin/uv"
MODEL="/home/lucas/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
SPILL_DIR="/home/lucas/.cache/freetoken/spill"
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

args=(
  "$UV" run --directory "$PROJECT" ft serve
  --model "$MODEL"
  --host "$HOST" --port "$PORT"
  --max-running-requests 1
  --num-tokens 1048576
  --max-seq-len-override 1048576
  --kv-cache-dtype q8_0
  --attention-backend triton
  --moe-backend offload --moe-cache-auto --moe-cache-policy lfu
  --memory-ratio 0.85 --max-prefill-length 4096
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
