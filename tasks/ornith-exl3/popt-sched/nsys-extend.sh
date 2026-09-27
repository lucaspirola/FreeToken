#!/usr/bin/env bash
# nsys of the owner's shapes on one tree: file-read extends at depth (10K and 30K at 100K, 30K at
# 200K), a 1000-token fresh prompt and an 80K fresh prefill. Saver and/or whole, ratio 1.00, q8_0,
# 262144. Run as a systemd --user unit; holds the GPU host lock per arm.
#   nsys-extend.sh OUTDIR [ARMS (default "saver whole")]    TREE=worktree (default this one)
# Per arm: warm-up (two 8K requests), one untraced extend pass (KV grows to 200K+), then traced:
# each extend of a fresh pass (its context untraced), then 1000 and 80000 fresh probes.
set -uo pipefail
F=$(dirname "$(readlink -f "$0")"); WT=${TREE:-$(cd "$F/../../.." && pwd)}; O=$1; ARMS=${2:-saver whole}
mkdir -p "$O"
NSYS=/usr/local/cuda/bin/nsys
MODEL=$HOME/ai/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq
EXTRA="--text-model-only --num-tokens 262144 --max-seq-len-override 262144 --served-model-name ornith --reasoning-parser qwen3"
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:/usr/lib/wsl/lib"
gpu() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock
for a in $ARMS; do
  flock 9
  until [ "$(gpu)" -le 32 ]; do sleep 5; done
  until [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -ge 23 ]; do sleep 20; done
  n=ext-$a; E="$O/$n.env"
  grep -vE '^[[:space:]]*export[[:space:]]+(FREETOKEN_MIRROR_EXPERT_RAM|FREETOKEN_MIRROR_HOST_ROWS|FREETOKEN_MODEL|FREETOKEN_MEMORY_RATIO|FREETOKEN_MIRROR_RESERVE_ROWS|FREETOKEN_MIRROR_TIEBREAK)=' \
    "$HOME/.config/freetoken/serve.env" > "$E" 2>/dev/null || : > "$E"
  { echo "export FREETOKEN_MODEL=$MODEL"; echo "export FREETOKEN_MEMORY_RATIO=1.00"; echo "export FREETOKEN_PIN_BUDGET_GB=20"
    if [ $a = whole ]; then echo "export FREETOKEN_MIRROR_EXPERT_RAM=0"; echo "export FREETOKEN_MIRROR_HOST_ROWS=0"
    else echo "export FREETOKEN_MIRROR_EXPERT_RAM=1"; echo "export FREETOKEN_MIRROR_HOST_ROWS=-1"; fi
    for kv in ${NS_ENVS:-}; do echo "export $kv"; done
    echo "export UV_PROJECT_ENVIRONMENT=/home/lucas/ai/FreeToken/.venv"; echo "export UV_NO_SYNC=1"
    echo "export PYTHONPATH=$WT/python"; } >> "$E"
  echo "=== $n $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg) gpu $(gpu) MiB sm $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader) code $(git -C $WT rev-parse --short HEAD)"
  U=ft-nsys-$n-$(date +%H%M%S)   # unique: a reused name greps an old "API server is ready"
  S=fte$a$$
  systemd-run --user --unit=$U --property=OOMScoreAdjust=1000 \
    --setenv=PATH="$PATH" --setenv=FREETOKEN_HOST_ENV="$E" --setenv=FREETOKEN_PORT=1920 --setenv=FREETOKEN_EXTRA_ARGS="$EXTRA --session-spill-dir $O/spill-$n ${NS_EXTRA:-}" \
    "$NSYS" launch --session-new=$S --trace=cuda,nvtx --cuda-graph-trace=graph "$WT/scripts/serve-default.sh" >/dev/null
  t0=$(date +%s)
  until journalctl --user -u $U -o cat --no-pager | grep -q "API server is ready" || ! systemctl --user is-active --quiet $U \
        || [ $(( $(date +%s) - t0 )) -gt 900 ]; do sleep 5; done
  echo "ready after $(( $(date +%s) - t0 ))s"
  export FREETOKEN_URL=http://127.0.0.1:1920 FREETOKEN_MODEL_NAME=ornith
  PROBE_GEN_TOKENS=64 python3 "$WT/scripts/probe_decode.py" 8000 > /dev/null 2>&1; sleep 20
  PROBE_GEN_TOKENS=64 python3 "$WT/scripts/probe_decode.py" 8000 > /dev/null 2>&1
  EXT_PASSES=1 EXT_TAG=u EXT_GEN=16 python3 "$F/extend_probe.py" | tee "$O/$n-untraced.jsonl"
  EXT_PASSES=1 EXT_TAG=t EXT_GEN=16 \
    EXT_PRE="$NSYS start --session=$S --output=$O/$n-d{depth}-a{add} --force-overwrite=true" \
    EXT_POST="$NSYS stop --session=$S" python3 "$F/extend_probe.py" | tee "$O/$n-traced.jsonl"
  for s in 1000 80000; do
    PROBE_TAG="u$s " PROBE_GEN_TOKENS=16 python3 "$WT/scripts/probe_decode.py" $s > /dev/null
    "$NSYS" start --session=$S --output="$O/$n-fresh-$s" --force-overwrite=true
    PROBE_TAG="t$s " PROBE_GEN_TOKENS=16 python3 "$WT/scripts/probe_decode.py" $s | tee "$O/$n-fresh-$s-probe.jsonl"
    "$NSYS" stop --session=$S
  done
  "$NSYS" shutdown --session=$S --kill=sigterm 2>&1 | tail -2
  sleep 10
  systemctl --user stop $U 2>/dev/null || true
  for _ in $(seq 60); do systemctl --user is-active --quiet $U || break; sleep 2; done
  journalctl --user -u $U -o cat --no-pager | sed 's/\x1b\[[0-9;]*m//g' > "$O/$n-journal.txt" || true
  rm -rf "$O/spill-$n"; flock -u 9
  sleep 20
  for r in "$O"/$n-*.nsys-rep; do
    "$NSYS" export --type=sqlite --force-overwrite=true --output="${r%.nsys-rep}.sqlite" "$r" >/dev/null 2>&1
  done
done
echo NSYSDONE
