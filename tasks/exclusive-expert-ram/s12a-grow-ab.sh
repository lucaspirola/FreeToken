#!/usr/bin/env bash
# Is the 80K prefill bimodality (pass-2 TTFT 13 s vs 20 s; R-S12a) caused by the KV grow that
# lands before the 80K request's final chunk? Investigation (campaign T-S12A-80K-BIMODAL): the
# whole extra time sits in the chunk right after the 65536->131072 grow, GPU half idle; Nemotron
# never shows it. Control: the same arm with --kv-grow-step-tokens 131072, so an 80K request
# never grows mid-prefill. Bracketed: step 64K, step 128K, step 64K. nvidia-smi dmon samples
# power/util/clocks/PCIe every second (a copy-bound vs host-starved stall looks different).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
HERE="$PWD/tasks/exclusive-expert-ram"
OUT="$HERE/results/ornith-s12a-grow-ab"
mkdir -p "$OUT"
export FT_VENV=/home/lucas/ai/FreeToken/.venv FT_RATIO=1.00
export FT_MODEL="$HOME/ai/models/Ornith-1.5-35B-A3B-NVFP4" FT_NAME=ornith FT_ROWS=0
BASE="--num-tokens 262144 --max-seq-len-override 262144 --served-model-name ornith"
export FT_SIZES="8000 32000 80000 128000"
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
run() {  # run ARM EXTRA
  empty_gpu
  nvidia-smi dmon -s puct -d 1 -o T > "$OUT/$1-dmon.txt" 2>&1 & smi=$!
  FT_EXTRA="$BASE $2" "$HERE/measure.sh" "$1" || echo "$1 measure exit $?"
  kill $smi 2>/dev/null || true
  journalctl --user -u ft-measure-$1 -o cat --no-pager | sed 's/\x1b\[[0-9;]*m//g' > "$OUT/$1-journal.txt" || true
  mv "$HERE"/results/$1-* "$OUT/" 2>/dev/null || true
  sleep 30
}
run ornith-grow64-a ""
run ornith-grow128 "--kv-grow-step-tokens 131072"
run ornith-grow64-b ""
echo "grow ab done"
