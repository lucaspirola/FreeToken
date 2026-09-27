#!/usr/bin/env bash
# kfix: nsys of Ornith EXL3 prefill at 300 and 1000 tokens, base (kbase) vs new (this tree), from final/nsys-prefill.sh; ratio 1.00, q8_0 KV,
# 262144 ceiling, chunk 8192 (serve-default's --max-prefill-length). Run as a systemd --user unit.
# Per arm: warm-up (two 64-token 8K requests), an untraced 32K + 80K pass (KV grows to its size
# there), then one traced request per size with fresh prompts (PROBE_TAG), 16 decode tokens.
#   nsys-prefill.sh OUTDIR [ARMS (default "saver whole")]
set -uo pipefail
F=$(dirname "$(readlink -f "$0")"); O=$1; ARMS=${2:-base:saver new:saver}
export PATH=/usr/lib/wsl/lib:/usr/local/cuda/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
mkdir -p "$O"
NSYS=/usr/local/cuda/bin/nsys
MODEL=$HOME/ai/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq
EXTRA="--text-model-only --num-tokens 262144 --max-seq-len-override 262144 --served-model-name ornith --reasoning-parser qwen3"
gpu() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock
for ta in $ARMS; do
  t=${ta%%:*}; a=${ta#*:}
  WT=$([ $t = base ] && echo /home/lucas/ai/FreeToken-wt/kbase || (cd "$F/../../.." && pwd))
  flock 9
  until [ "$(gpu)" -le 32 ]; do sleep 5; done
  until [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -ge 23 ]; do sleep 20; done
  n=mid-$t-$a; E="$O/$n.env"
  grep -vE '^[[:space:]]*export[[:space:]]+(FREETOKEN_MIRROR_EXPERT_RAM|FREETOKEN_MIRROR_HOST_ROWS|FREETOKEN_MODEL|FREETOKEN_MEMORY_RATIO|FREETOKEN_MIRROR_RESERVE_ROWS|FREETOKEN_MIRROR_TIEBREAK)=' \
    "$HOME/.config/freetoken/serve.env" > "$E" 2>/dev/null || : > "$E"
  { echo "export FREETOKEN_MODEL=$MODEL"; echo "export FREETOKEN_MEMORY_RATIO=1.00"; echo "export FREETOKEN_PIN_BUDGET_GB=20"
    if [ $a = whole ]; then echo "export FREETOKEN_MIRROR_EXPERT_RAM=0"; echo "export FREETOKEN_MIRROR_HOST_ROWS=0"
    else echo "export FREETOKEN_MIRROR_EXPERT_RAM=1"; echo "export FREETOKEN_MIRROR_HOST_ROWS=-1"; fi
    echo "export UV_PROJECT_ENVIRONMENT=/home/lucas/ai/FreeToken/.venv"; echo "export UV_NO_SYNC=1"
    echo "export PYTHONPATH=$WT/python"; } >> "$E"
  echo "=== $n $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg) gpu $(gpu) MiB sm $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader) code $(git -C $WT rev-parse --short HEAD)"
  U=ft-measure-nsys-$n
  systemctl --user reset-failed $U 2>/dev/null || true
  systemd-run --user --unit=$U --property=OOMScoreAdjust=1000 \
    --setenv=PATH="$HOME/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib" \
    --setenv=FREETOKEN_HOST_ENV="$E" --setenv=FREETOKEN_PORT=1920 --setenv=FREETOKEN_EXTRA_ARGS="$EXTRA" \
    "$NSYS" launch --session-new=ftm$t$a --trace=cuda,nvtx --cuda-graph-trace=graph "$WT/scripts/serve-default.sh" >/dev/null
  t0=$(date +%s)
  until journalctl --user -u $U -o cat --no-pager | grep -q "API server is ready" || ! systemctl --user is-active --quiet $U \
        || [ $(( $(date +%s) - t0 )) -gt 900 ]; do sleep 5; done
  echo "ready after $(( $(date +%s) - t0 ))s"
  export FREETOKEN_URL=http://127.0.0.1:1920 FREETOKEN_MODEL_NAME=ornith
  PROBE_GEN_TOKENS=64 python3 "$WT/scripts/probe_decode.py" 8000 > /dev/null 2>&1; sleep 20
  PROBE_GEN_TOKENS=64 python3 "$WT/scripts/probe_decode.py" 8000 > /dev/null 2>&1
  PROBE_GEN_TOKENS=16 python3 "$WT/scripts/probe_decode.py" 300 1000 | tee "$O/$n-untraced.jsonl"
  for s in 300 1000; do
    "$NSYS" start --session=ftm$t$a --output="$O/$n-$s" --force-overwrite=true
    PROBE_TAG="t$s " PROBE_GEN_TOKENS=16 python3 "$WT/scripts/probe_decode.py" $s | tee "$O/$n-$s-probe.jsonl"
    "$NSYS" stop --session=ftm$t$a
  done
  "$NSYS" shutdown --session=ftm$t$a --kill=sigterm 2>&1 | tail -2
  sleep 10
  systemctl --user stop $U 2>/dev/null || true
  for _ in $(seq 60); do systemctl --user is-active --quiet $U || break; sleep 2; done
  journalctl --user -u $U -o cat --no-pager | sed 's/\x1b\[[0-9;]*m//g' > "$O/$n-journal.txt" || true
  systemctl --user reset-failed $U 2>/dev/null || true
  flock -u 9
  sleep 20
  for s in 300 1000; do
    [ -f "$O/$n-$s.nsys-rep" ] && "$NSYS" export --type=sqlite --force-overwrite=true --output="$O/$n-$s.sqlite" "$O/$n-$s.nsys-rep" >/dev/null 2>&1
  done
done
echo NSYSDONE
