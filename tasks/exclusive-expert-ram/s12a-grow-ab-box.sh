#!/usr/bin/env bash
# s12a-grow-ab.sh on a rented native-Linux RTX 5080 (no systemd, root, Docker): is Ornith's
# bimodal 80K pass-2 prefill a WSL/WDDM effect (H1) or the mid-prefill KV grow (H3)?
# Same three whole-model arms, ratio 1.00, sizes and passes as the local script; the server
# runs under measure.sh's FT_LAUNCHER=nohup. Every GPU run holds /root/gpu.lock (another
# worker may share the box). Per arm: nvidia-smi dmon -s puct -d 1, the server log stamped
# on a monotonic clock (logstamp.py, for startup phases), the probe with time.monotonic().
#
#   ARMS="startup-cold startup-warm grow64-a grow128 grow64-b" s12a-grow-ab-box.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
HERE="$PWD/tasks/exclusive-expert-ram"
OUT="$HERE/results/ornith-s12a-grow-ab-box"
mkdir -p "$OUT"
export CUDA_HOME=/usr/local/cuda-13.0 PATH="/usr/local/cuda-13.0/bin:$PATH"
export FT_LAUNCHER=nohup FT_VENV=/root/venv FT_RATIO=1.00
export FT_MODEL=/root/models/Ornith-1.5-35B-A3B-NVFP4 FT_NAME=ornith FT_ROWS=0
BASE="--num-tokens 262144 --max-seq-len-override 262144 --served-model-name ornith"
AB_SIZES="8000 32000 80000 128000"
empty_gpu() {
  [ -z "$(git status --porcelain -- python)" ] || { echo "python/ dirty"; exit 1; }
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$used" -le 16 ] || { echo "GPU holds $used MiB"; nvidia-smi; exit 1; }
  avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  echo "preflight ok: GPU $used MiB, MemAvailable $avail GiB, code $(git log --oneline -1)"
}
run() {  # run ARM SIZES EXTRA
  local arm="ornith-$1"
  exec 9>/root/gpu.lock
  echo "[$arm] waiting for /root/gpu.lock"; flock 9
  empty_gpu
  nvidia-smi dmon -s puct -d 1 -o T > "$OUT/$arm-dmon.txt" 2>&1 & local smi=$!
  rm -f "$HERE/results/$arm-server.log"
  python3 "$HERE/logstamp.py" "$HERE/results/$arm-server.log" "$OUT/$arm-stamped.txt" & local st=$!
  FT_SIZES="$2" FT_EXTRA="$BASE $3" "$HERE/measure.sh" "$arm" || echo "$arm measure exit $?"
  kill $smi $st 2>/dev/null || true
  mv "$HERE/results/$arm-server.log" "$OUT/$arm-journal.txt" 2>/dev/null || true
  mv "$HERE"/results/$arm-* "$HERE"/results/$arm.env "$OUT/" 2>/dev/null || true
  flock -u 9; exec 9>&-
  sleep 30
}
for a in ${ARMS:-startup-cold startup-warm grow64-a grow128 grow64-b}; do
  case "$a" in
    startup-*) run "$a" "8000" "" ;;
    grow128*)  run "$a" "$AB_SIZES" "--kv-grow-step-tokens 131072" ;;
    grow64*)   run "$a" "$AB_SIZES" "" ;;
    serial-*)  run "$a" "8000" "--expert-load serial" ;;
    *) echo "unknown arm $a"; exit 1 ;;
  esac
done
echo "grow ab box done"
