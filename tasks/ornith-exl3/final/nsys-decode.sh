#!/usr/bin/env bash
# nsys of Ornith EXL3 decode at 8K, whole vs saver, on the owner's machine, with the ck6o settings
# (262144 tokens, q8_0 KV, ratio 1.00, pin budget 20; saver = mirror, auto rows, pool default reserve).
# measure.sh's warm-up (two 64-token 8K requests + 20 s) runs first, so the traced request sequence
# is the probe's: 8K pass 1 (prefix hit on the warm-up prompt) then 8K pass 2 (new prompt, full
# prefill), 512 tokens each, traced by `nsys start/stop`.
#   nsys-ornith.sh WORKTREE OUTDIR [ARMS (default "whole saver")] [GRAPH_TRACE (graph|node)]
# Run as a systemd --user unit, never from an agent shell. Waits for 0..32 MiB on the GPU and
# MemAvailable >= 23 GiB, under the GPU host lock [agent practice].
set -uo pipefail
REPO=$1; O=$2; ARMS=${3:-whole saver}; GT=${4:-graph}
mkdir -p "$O"
NSYS=/usr/local/cuda/bin/nsys
MODEL=$HOME/ai/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq
EXTRA="--text-model-only --num-tokens 262144 --max-seq-len-override 262144 --served-model-name ornith --reasoning-parser qwen3"
echo "nsys: $($NSYS --version) code $(git -C "$REPO" rev-parse --short HEAD) trace=$GT"
gpu() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock
for a in $ARMS; do
  flock 9
  until [ "$(gpu)" -le 32 ]; do sleep 5; done
  until [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -ge 23 ]; do sleep 20; done
  n=$a-$GT; E="$O/$n.env"
  grep -vE '^[[:space:]]*export[[:space:]]+(FREETOKEN_MIRROR_EXPERT_RAM|FREETOKEN_MIRROR_HOST_ROWS|FREETOKEN_MODEL|FREETOKEN_MEMORY_RATIO|FREETOKEN_MIRROR_RESERVE_ROWS|FREETOKEN_MIRROR_TIEBREAK)=' \
    "$HOME/.config/freetoken/serve.env" > "$E" 2>/dev/null || : > "$E"
  { echo "export FREETOKEN_MODEL=$MODEL"; echo "export FREETOKEN_MEMORY_RATIO=1.00"
    echo "export FREETOKEN_PIN_BUDGET_GB=20"
    if [ $a = whole ]; then echo "export FREETOKEN_MIRROR_EXPERT_RAM=0"; echo "export FREETOKEN_MIRROR_HOST_ROWS=0"
    else echo "export FREETOKEN_MIRROR_EXPERT_RAM=1"; echo "export FREETOKEN_MIRROR_HOST_ROWS=-1"; fi
    for kv in ${NS_ENVS:-}; do echo "export $kv"; done
    echo "export UV_PROJECT_ENVIRONMENT=/home/lucas/ai/FreeToken/.venv"; echo "export UV_NO_SYNC=1"
    echo "export PYTHONPATH=$REPO/python"; } >> "$E"
  echo "=== $n $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg) gpu $(gpu) MiB"
  U=ft-measure-nsys-$n-$(date +%H%M%S)  # unique: a reused name greps an old "API server is ready"
  systemctl --user reset-failed $U 2>/dev/null || true
  systemd-run --user --unit=$U --property=OOMScoreAdjust=1000 \
    --setenv=PATH="$HOME/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib" \
    --setenv=FREETOKEN_HOST_ENV="$E" --setenv=FREETOKEN_PORT=1920 --setenv=FREETOKEN_EXTRA_ARGS="$EXTRA" \
    "$NSYS" launch --session-new=ft$a --trace=cuda,nvtx --cuda-graph-trace=$GT "$REPO/scripts/serve-default.sh" >/dev/null
  t0=$(date +%s)
  until journalctl --user -u $U -o cat --no-pager | grep -q "API server is ready" || ! systemctl --user is-active --quiet $U \
        || [ $(( $(date +%s) - t0 )) -gt 900 ]; do sleep 5; done
  echo "ready after $(( $(date +%s) - t0 ))s"
  export FREETOKEN_URL=http://127.0.0.1:1920 FREETOKEN_MODEL_NAME=ornith
  PROBE_GEN_TOKENS=64 python3 "$REPO/scripts/probe_decode.py" 8000 > /dev/null 2>&1; sleep 20
  PROBE_GEN_TOKENS=64 python3 "$REPO/scripts/probe_decode.py" 8000 > /dev/null 2>&1
  "$NSYS" start --session=ft$a --output="$O/$n" --force-overwrite=true
  PROBE_STATS=1 PROBE_GEN_TOKENS=512 PROBE_PASSES=2 python3 "$REPO/scripts/probe_decode.py" 8000 | tee "$O/$n-probe.jsonl"
  "$NSYS" stop --session=ft$a
  "$NSYS" shutdown --session=ft$a --kill=sigterm 2>&1 | tail -2
  sleep 10
  systemctl --user stop $U 2>/dev/null || true
  for _ in $(seq 60); do systemctl --user is-active --quiet $U || break; sleep 2; done
  journalctl --user -u $U -o cat --no-pager | sed 's/\x1b\[[0-9;]*m//g' > "$O/$n-journal.txt" || true
  systemctl --user reset-failed $U 2>/dev/null || true
  flock -u 9
  sleep 20
  [ -f "$O/$n.nsys-rep" ] && "$NSYS" export --type=sqlite --force-overwrite=true --output="$O/$n.sqlite" "$O/$n.nsys-rep" >/dev/null 2>&1
done
echo NSYSDONE
