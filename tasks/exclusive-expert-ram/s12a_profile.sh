#!/usr/bin/env bash
# Plan S12a "gap explained from a profile": where does an Ornith 128K prefill spend its
# time? One nsys capture of one 128K request on a warmed server (whole model in RAM,
# arena, ratio 1.00, empty GPU), then per-kernel / memcpy / API summaries.
# The warm-up includes a 120K request so the capture is not the first-after-start
# slow period; its prompt differs from the 128K one from the first token (no cache hit).
# Run as a systemd transient unit (--setenv=PATH), never an agent shell.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
REPO="$PWD"; HERE="$REPO/tasks/exclusive-expert-ram"
OUT="$HERE/results/ornith-s12a/profile"; mkdir -p "$OUT"
PORT=1920; NAME=ornith; UNIT=ft-profile-ornith; SESSION=ft-profile-ornith
MODEL="$HOME/ai/models/Ornith-1.5-35B-A3B-NVFP4"
REP="$OUT/prefill128k"
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock
flock 9
[ -z "$(git status --porcelain -- python)" ] || { echo "python/ dirty"; exit 1; }
systemctl is-active --quiet freetoken-serve && { echo "freetoken-serve is up"; exit 1; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ "$used" = 0 ] || { echo "GPU holds $used MiB"; exit 1; }
avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
[ "$avail" -ge 22 ] || { echo "MemAvailable $avail GiB"; exit 1; }
echo "preflight ok: GPU 0 MiB, MemAvailable $avail GiB, code $(git log --oneline -1)"

ARMENV="$OUT/profile.env"
grep -vE '^[[:space:]]*export[[:space:]]+(FREETOKEN_MIRROR_EXPERT_RAM|FREETOKEN_MIRROR_HOST_ROWS|FREETOKEN_MODEL|FREETOKEN_MEMORY_RATIO|FREETOKEN_MIRROR_RESERVE_ROWS|FREETOKEN_MIRROR_TIEBREAK)=' \
  "$HOME/.config/freetoken/serve.env" > "$ARMENV" 2>/dev/null || : > "$ARMENV"
{ echo "export FREETOKEN_MODEL=$MODEL"; echo "export FREETOKEN_MEMORY_RATIO=1.00"
  echo "export FREETOKEN_MIRROR_EXPERT_RAM=0"; echo "export FREETOKEN_MIRROR_HOST_ROWS=0"; } >> "$ARMENV"

systemctl --user reset-failed "$UNIT" 2>/dev/null || true
systemd-run --user --unit="$UNIT" --property=OOMScoreAdjust=1000 \
  --setenv=PATH="$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib" \
  --setenv=FREETOKEN_PORT="$PORT" --setenv=FREETOKEN_HOST_ENV="$ARMENV" \
  --setenv=UV_PROJECT_ENVIRONMENT=/home/lucas/ai/FreeToken/.venv --setenv=UV_NO_SYNC=1 \
  --setenv=PYTHONPATH="$REPO/python" \
  --setenv=FREETOKEN_EXTRA_ARGS="--num-tokens 262144 --max-seq-len-override 262144 --served-model-name ornith" \
  -- nsys launch --session="$SESSION" --trace=cuda,nvtx,osrt --cuda-graph-trace=graph \
     -- "$REPO/scripts/serve-default.sh" >/dev/null
trap 'systemctl --user stop "$UNIT" >/dev/null 2>&1 || true' EXIT

started=$(date +%s)
until [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 -H 'Content-Type: application/json' \
  -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
  "http://127.0.0.1:$PORT/v1/chat/completions")" = 200 ]; do
  systemctl --user is-active --quiet "$UNIT" || { journalctl --user -u "$UNIT" -n 40 --no-pager; exit 1; }
  [ $(( $(date +%s) - started )) -gt 900 ] && { echo "readiness timeout"; exit 1; }
  sleep 5
done
echo "ready after $(( $(date +%s) - started ))s"
P="$REPO/scripts/probe_decode.py"
export FREETOKEN_URL="http://127.0.0.1:$PORT" FREETOKEN_MODEL_NAME="$NAME"
PROBE_GEN_TOKENS=16 "$P" 8000 >/dev/null 2>&1 || true
PROBE_GEN_TOKENS=16 "$P" 8000 120000 > "$OUT/warmup.jsonl" 2>&1 || true
echo "warm-up done: $(tail -1 "$OUT/warmup.jsonl" | cut -c1-160)"

nsys start --session="$SESSION" --output="$REP.nsys-rep"
PROBE_GEN_TOKENS=16 "$P" 128000 > "$OUT/prefill128k-probe.jsonl" || true
nsys stop --session="$SESSION"
cat "$OUT/prefill128k-probe.jsonl"
journalctl --user -u "$UNIT" -o cat --no-pager | sed 's/\x1b\[[0-9;]*m//g' > "$OUT/journal.txt" || true
systemctl --user stop "$UNIT" || true
trap - EXIT

nsys export --type=sqlite --force-overwrite=true --output="$REP.sqlite" "$REP.nsys-rep"
for r in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_gpu_mem_size_sum cuda_api_sum nvtx_sum osrt_sum; do
  nsys stats --report=$r --format=csv --output="$REP" "$REP.sqlite" >/dev/null 2>&1 || echo "report $r failed"
done
ls -la "$OUT"
echo "profile done"
