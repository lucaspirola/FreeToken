#!/usr/bin/env bash
# nsys1m-local.sh's step-1 traces on a rented native-Linux box (no systemd; Vast container):
# fresh server per arm, three 8K warm-ups, one pass-1 1M request, nsys started NS_START_AT s
# after the request (before the first token) with --cuda-graph-trace=node, window NS_WINDOW
# tokens. Summary: nsys_bins.py (16-step bins: step ms, write-back MB, fetch us, and the share of
# the write-back DMA that runs beside attention vs beside fetches).
#   NS_ARMS  arms (default "whole mirror mirror-nogate"):
#            whole          whole model in RAM
#            mirror         mirror, reserve 256, write-back DMA gated on attention (the default)
#            mirror-nogate  same with FREETOKEN_MIRROR_WB_AT_ATTENTION=0 (the pre-wb-schedule DMA)
#   NS_OUT (default results/nsys1m-box), NS_SIZE (1000000), NS_START_AT (760: box TTFT ~790 s),
#   NS_SKIP / NS_WINDOW (0 / 160), NS_TAG suffix.
# /root/gpu.lock is held for all arms. Run detached: setsid nohup .../nsys1m-box.sh > log 2>&1 &
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
REPO="$PWD"; HERE="$REPO/tasks/exclusive-expert-ram"
O="${NS_OUT:-$HERE/results/nsys1m-box}"; mkdir -p "$O"; SIZE="${NS_SIZE:-1000000}"
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
NSYS=/usr/local/cuda/bin/nsys
T="${NS_TAG:-}"
echo "nsys: $($NSYS --version) code $(git -C "$REPO" log --oneline -1 | cut -c1-80)"
exec 9>/root/gpu.lock
flock 9
for a in ${NS_ARMS:-whole mirror mirror-nogate}; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$used" -le 16 ] || { echo "GPU holds $used MiB, skipping $a"; continue; }
  avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  [ "$avail" -ge 22 ] || { echo "MemAvailable $avail GiB, skipping $a"; continue; }
  E="$O/$a$T.env"
  { echo "export FREETOKEN_MODEL=/root/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
    echo "export FREETOKEN_MEMORY_RATIO=1.00"
    if [ $a = whole ]; then echo "export FREETOKEN_MIRROR_EXPERT_RAM=0"; echo "export FREETOKEN_MIRROR_HOST_ROWS=0"
    else echo "export FREETOKEN_MIRROR_EXPERT_RAM=1"; echo "export FREETOKEN_MIRROR_HOST_ROWS=-1"; echo "export FREETOKEN_MIRROR_RESERVE_ROWS=256"; fi
    [ $a = mirror-nogate ] && echo "export FREETOKEN_MIRROR_WB_AT_ATTENTION=0"
    echo "export UV_PROJECT_ENVIRONMENT=/root/venv"; echo "export UV_NO_SYNC=1"
    echo "export PYTHONPATH=$REPO/python"; } > "$E"
  echo "=== nsys $a $SIZE $(date -u +%FT%TZ)"
  L="$O/$a$T-journal.txt"; : > "$L"
  FREETOKEN_HOST_ENV="$E" FREETOKEN_PORT=1920 PATH="$HOME/.local/bin:$PATH" \
    setsid nohup "$NSYS" launch --session-new=ft1m$a$T --trace=cuda,nvtx --cuda-graph-trace=node \
    "$REPO/scripts/serve-default.sh" >> "$L" 2>&1 < /dev/null &
  launched=$!; sleep 1
  PG=$(ps -o pgid= -p $launched 2>/dev/null | tr -d ' '); PG=${PG:-$launched}
  t0=$(date +%s)
  until grep -q "API server is ready" "$L" || ! kill -0 -- -$PG 2>/dev/null || [ $(( $(date +%s) - t0 )) -gt 900 ]; do sleep 5; done
  echo "ready after $(( $(date +%s) - t0 ))s"
  export FREETOKEN_URL=http://127.0.0.1:1920
  for i in 1 2 3; do PROBE_GEN_TOKENS=64 /root/venv/bin/python "$REPO/scripts/probe_decode.py" 8000 > /dev/null 2>&1; sleep 10; done
  NSYS=$NSYS START_AT="${NS_START_AT:-760}" /root/venv/bin/python "$HERE/nsys_window.py" "$SIZE" ft1m$a$T "$O/$a$T" \
    "${NS_SKIP:-0}" "${NS_WINDOW:-160}" | tee "$O/$a$T-window.jsonl"
  "$NSYS" shutdown --session=ft1m$a$T --kill=sigterm 2>&1 | tail -2
  sleep 10
  kill -TERM -- -$PG 2>/dev/null
  for _ in $(seq 60); do kill -0 -- -$PG 2>/dev/null || break; sleep 2; done
  kill -KILL -- -$PG 2>/dev/null
  sed -i 's/\x1b\[[0-9;]*m//g' "$L"
  echo "captures=$(grep -c 'Start capturing CUDA graphs' "$L") tracebacks=$(grep -c Traceback "$L") gate=$(grep -c 'behind that layer' "$L")"
  sleep 20
done
flock -u 9
for a in ${NS_ARMS:-whole mirror mirror-nogate}; do
  [ -f "$O/$a$T.nsys-rep" ] || continue
  "$NSYS" export --type=sqlite --force-overwrite=true --output="$O/$a$T.sqlite" "$O/$a$T.nsys-rep" >/dev/null 2>&1
done
/root/venv/bin/python "$HERE/nsys_bins.py" "$O"/*"$T".sqlite > "$O/bins$T.txt" 2>&1; cat "$O/bins$T.txt"
echo "nsys1m-box done"
