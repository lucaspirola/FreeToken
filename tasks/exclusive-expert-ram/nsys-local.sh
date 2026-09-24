#!/usr/bin/env bash
# nsys trace of one 8K request (512 decode tokens, two passes) on the owner's machine, whole vs
# pool (mirror, reserve 256): where the pool's ~0.5 ms/token at 8K goes (recheck-local.sh). The
# same method as ck4dma-g5/ck4dma-nsys/g5-nsys.sh (native Linux box). Here the server runs as a
# systemd user unit under `nsys launch`, and `nsys start/stop` brackets the measured request
# after three warmups. Summaries: nsys_steps.py (from ck4dma-g5) per arm.
#   NS_OUT  output dir (default results/nsys-local)
# Waits for MemAvailable >= 23 GiB and 0 MiB on the GPU before each arm [agent practice].
# Run as a systemd transient user unit (--setenv=PATH), never from an agent shell.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
REPO="$PWD"; HERE="$REPO/tasks/exclusive-expert-ram"
O="${NS_OUT:-$HERE/results/nsys-local}"; mkdir -p "$O"
NSYS=/usr/local/cuda/bin/nsys
echo "nsys: $($NSYS --version) code $(git -C "$REPO" rev-parse --short HEAD)"
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock
for a in whole mirror; do
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
  echo "=== nsys $a $(date -Is)"
  U=ft-measure-nsys-$a
  systemctl --user reset-failed $U 2>/dev/null || true
  systemd-run --user --unit=$U --property=OOMScoreAdjust=1000 \
    --setenv=PATH="$HOME/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib" \
    --setenv=FREETOKEN_HOST_ENV="$E" --setenv=FREETOKEN_PORT=1920 \
    "$NSYS" launch --session-new=ft$a --trace=cuda,nvtx --cuda-graph-trace=graph "$REPO/scripts/serve-default.sh" >/dev/null
  t0=$(date +%s)
  until journalctl --user -u $U -o cat --no-pager | grep -q "API server is ready" || ! systemctl --user is-active --quiet $U \
        || [ $(( $(date +%s) - t0 )) -gt 900 ]; do sleep 5; done
  echo "ready after $(( $(date +%s) - t0 ))s"
  export FREETOKEN_URL=http://127.0.0.1:1920
  for i in 1 2 3; do PROBE_GEN_TOKENS=64 python3 "$REPO/scripts/probe_decode.py" 8000 > /dev/null 2>&1; sleep 10; done
  "$NSYS" start --session=ft$a --output="$O/$a" --force-overwrite=true
  PROBE_GEN_TOKENS=512 PROBE_PASSES=2 python3 "$REPO/scripts/probe_decode.py" 8000 | tee "$O/$a-probe.jsonl"
  "$NSYS" stop --session=ft$a
  "$NSYS" shutdown --session=ft$a --kill=sigterm 2>&1 | tail -2
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
python3 "$HERE/results/ck4dma-g5/ck4dma-nsys/nsys_steps.py" "$O/whole.sqlite" "$O/mirror.sqlite" > "$O/steps-graph-level.txt" 2>&1 || true
cat "$O/steps-graph-level.txt"
echo "nsys-local done"
