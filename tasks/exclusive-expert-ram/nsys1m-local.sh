#!/usr/bin/env bash
# nsys of one decode window at 1M context, whole-1m vs mirror-1m (reserve 256), pass 1 (fresh
# server, three 8K warm-ups, then the 1M request). nsys_window.py traces ~200 decode tokens after
# the first 33, with --cuda-graph-trace=node, so prefill is not in the report. Summaries:
# nsys_steps.py (graph level is not available with node trace; kept for API/memcpy) and
# nsys_graph_nodes.py (in-graph kernel time per step), plus a per-step overlap table
# (nsys_overlap.py): attention kernels vs the pool's copy kernels.
#   NS_OUT  output dir (default results/nsys1m-local); NS_SIZE (default 1000000)
# Waits for MemAvailable >= 23 GiB and 0 MiB on the GPU before each arm [agent practice].
# Run as a systemd transient user unit (--setenv=PATH), never from an agent shell.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
REPO="$PWD"; HERE="$REPO/tasks/exclusive-expert-ram"
O="${NS_OUT:-$HERE/results/nsys1m-local}"; mkdir -p "$O"; SIZE="${NS_SIZE:-1000000}"
NSYS=/usr/local/cuda/bin/nsys
echo "nsys: $($NSYS --version) code $(git -C "$REPO" rev-parse --short HEAD)"
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock
for a in ${NS_ARMS:-whole mirror}; do
  until [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -ge 23 ]; do sleep 60; done
  flock 9
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$used" = 0 ] || { echo "GPU holds $used MiB, skipping $a"; flock -u 9; continue; }
  E="$O/$a.env"
  grep -vE '^[[:space:]]*export[[:space:]]+(FREETOKEN_MIRROR_EXPERT_RAM|FREETOKEN_MIRROR_HOST_ROWS|FREETOKEN_MODEL|FREETOKEN_MEMORY_RATIO|FREETOKEN_MIRROR_RESERVE_ROWS)=' \
    "$HOME/.config/freetoken/serve.env" > "$E" 2>/dev/null || : > "$E"
  { echo "export FREETOKEN_MODEL=$HOME/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
    echo "export FREETOKEN_MEMORY_RATIO=1.00"
    if [ $a = whole ]; then echo "export FREETOKEN_MIRROR_EXPERT_RAM=0"; echo "export FREETOKEN_MIRROR_HOST_ROWS=0"
    else echo "export FREETOKEN_MIRROR_EXPERT_RAM=1"; echo "export FREETOKEN_MIRROR_HOST_ROWS=-1"; echo "export FREETOKEN_MIRROR_RESERVE_ROWS=256"; fi
    echo "export UV_PROJECT_ENVIRONMENT=/home/lucas/ai/FreeToken/.venv"; echo "export UV_NO_SYNC=1"
    echo "export PYTHONPATH=$REPO/python"; } >> "$E"
  echo "=== nsys $a $SIZE $(date -Is)"
  U=ft-measure-nsys1m-$a
  systemctl --user reset-failed $U 2>/dev/null || true
  systemd-run --user --unit=$U --property=OOMScoreAdjust=1000 \
    --setenv=PATH="$HOME/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib" \
    --setenv=FREETOKEN_HOST_ENV="$E" --setenv=FREETOKEN_PORT=1920 \
    "$NSYS" launch --session-new=ft1m$a --trace=cuda,nvtx --cuda-graph-trace=node "$REPO/scripts/serve-default.sh" >/dev/null
  t0=$(date +%s)
  until journalctl --user -u $U -o cat --no-pager | grep -q "API server is ready" || ! systemctl --user is-active --quiet $U \
        || [ $(( $(date +%s) - t0 )) -gt 900 ]; do sleep 5; done
  echo "ready after $(( $(date +%s) - t0 ))s"
  export FREETOKEN_URL=http://127.0.0.1:1920
  for i in 1 2 3; do PROBE_GEN_TOKENS=64 python3 "$REPO/scripts/probe_decode.py" 8000 > /dev/null 2>&1; sleep 10; done
  NSYS=$NSYS python3 "$HERE/nsys_window.py" "$SIZE" ft1m$a "$O/$a" 32 200 | tee "$O/$a-window.jsonl"
  "$NSYS" shutdown --session=ft1m$a --kill=sigterm 2>&1 | tail -2
  sleep 10
  systemctl --user stop $U 2>/dev/null || true
  for _ in $(seq 60); do systemctl --user is-active --quiet $U || break; sleep 2; done
  journalctl --user -u $U -o cat --no-pager | sed 's/\x1b\[[0-9;]*m//g' > "$O/$a-journal.txt" || true
  flock -u 9
  sleep 20
done
for a in whole mirror; do
  [ -f "$O/$a.nsys-rep" ] || continue
  "$NSYS" export --type=sqlite --force-overwrite=true --output="$O/$a.sqlite" "$O/$a.nsys-rep" >/dev/null 2>&1
done
N="$HERE/results/ck4dma-g5/ck4dma-nsys"
python3 "$N/nsys_graph_nodes.py" "$O/whole.sqlite" "$O/mirror.sqlite" > "$O/in-graph-node-level.txt" 2>&1 || true
python3 "$HERE/nsys_overlap.py" "$O/whole.sqlite" "$O/mirror.sqlite" > "$O/overlap.txt" 2>&1 || true
cat "$O/in-graph-node-level.txt" "$O/overlap.txt"
echo "nsys1m-local done"
