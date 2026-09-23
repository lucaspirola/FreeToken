#!/usr/bin/env bash
# A/B for the extend-tile fix (016412c): is the pass-2 decode drop seen in ornith-s12a-attn
# (32K 153 vs 165, 80K 133 vs 149) caused by the patch or by run-to-run state? Same commit,
# same arm, three runs back to back: OLD tile forced by env (64/32/4/2), NEW tile (default),
# OLD again. nvidia-smi samples clocks/temp/power every second for each arm.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
HERE="$PWD/tasks/exclusive-expert-ram"
OUT="$HERE/results/ornith-s12a-attn-ab"
mkdir -p "$OUT"
export FT_VENV=/home/lucas/ai/FreeToken/.venv FT_RATIO=1.00
export FT_MODEL="$HOME/ai/models/Ornith-1.5-35B-A3B-NVFP4" FT_NAME=ornith FT_ROWS=0
export FT_EXTRA="--num-tokens 262144 --max-seq-len-override 262144 --served-model-name ornith"
export FT_SIZES="8000 32000 80000 128000"
OLD="FREETOKEN_EXTEND_BLOCK_M=64 FREETOKEN_EXTEND_BLOCK_N=32 FREETOKEN_EXTEND_NUM_WARPS=4 FREETOKEN_EXTEND_NUM_STAGES=2"
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock
flock 9
empty_gpu() {
  [ -z "$(git status --porcelain -- python)" ] || { echo "python/ dirty"; exit 1; }
  systemctl is-active --quiet freetoken-serve && { echo "freetoken-serve is up"; exit 1; }
  systemctl --user list-units --state=active --no-legend 'ft-measure-*' | grep -q . && { echo "ft-measure unit up"; exit 1; }
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$used" = 0 ] || { echo "GPU holds $used MiB"; exit 1; }
  avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  [ "$avail" -ge 22 ] || { echo "MemAvailable $avail GiB"; exit 1; }
  echo "preflight ok: GPU 0 MiB, MemAvailable $avail GiB, code $(git log --oneline -1)"
}
run() {  # run ARM ENVS
  empty_gpu
  nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,clocks_throttle_reasons.active \
    --format=csv,noheader -l 1 > "$OUT/$1-gpu.csv" 2>&1 & smi=$!
  FT_ENVS="$2" "$HERE/measure.sh" "$1" || echo "$1 measure exit $?"
  kill $smi 2>/dev/null || true
  journalctl --user -u ft-measure-$1 -o cat --no-pager | sed 's/\x1b\[[0-9;]*m//g' > "$OUT/$1-journal.txt" || true
  mv "$HERE"/results/$1-* "$OUT/" 2>/dev/null || true
  sleep 30
}
run ornith-ab-old1 "$OLD"
run ornith-ab-new ""
run ornith-ab-old2 "$OLD"
echo "ab done"
