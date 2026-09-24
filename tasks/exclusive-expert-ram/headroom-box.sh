#!/usr/bin/env bash
# Prefill-headroom acceptance on a rented native-Linux RTX 5080 (no systemd, root, Docker).
# Branch reorg-headroom measures one --max-prefill-length chunk's transient at startup and
# makes the growable-KV headroom (startup arena fill, every grow's arena shrink, the ceiling
# plan, the mirror pool estimate) price it. Native Linux turns a missing byte of headroom
# into an OOM instead of WDDM paging, so ratio 1.00 starting and serving here is the test.
# Server under measure.sh's FT_LAUNCHER=nohup; every GPU run holds /root/gpu.lock.
#
#   ARMS="whole saver nemotron" headroom-box.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
HERE="$PWD/tasks/exclusive-expert-ram"
OUT="$HERE/results/ornith-headroom-box"
mkdir -p "$OUT"
export CUDA_HOME=/usr/local/cuda-13.0 PATH="/usr/local/cuda-13.0/bin:$PATH"
export FT_LAUNCHER=nohup FT_VENV=/root/venv FT_RATIO="${FT_RATIO:-1.00}"
ORNITH=/root/models/Ornith-1.5-35B-A3B-NVFP4
NEMOTRON=/root/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
ORNITH_BASE="--num-tokens 262144 --max-seq-len-override 262144 --served-model-name ornith"
ORNITH_SIZES="8000 32000 80000 128000"
NEMOTRON_SIZES="${NEMOTRON_SIZES:-8000 80000 256000 713000}"
empty_gpu() {
  [ -z "$(git status --porcelain -- python)" ] || { echo "python/ dirty"; exit 1; }
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$used" -le 16 ] || { echo "GPU holds $used MiB"; nvidia-smi; exit 1; }
  avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  echo "preflight ok: GPU $used MiB, MemAvailable $avail GiB, code $(git log --oneline -1)"
}
run() {  # run ARM MODEL NAME ROWS SIZES EXTRA
  local arm="$1"
  exec 9>/root/gpu.lock
  echo "[$arm] waiting for /root/gpu.lock"; flock 9
  empty_gpu
  nvidia-smi dmon -s puct -d 1 -o T > "$OUT/$arm-dmon.txt" 2>&1 & local smi=$!
  rm -f "$HERE/results/$arm-server.log"
  python3 "$HERE/logstamp.py" "$HERE/results/$arm-server.log" "$OUT/$arm-stamped.txt" & local st=$!
  FT_MODEL="$2" FT_NAME="$3" FT_ROWS="$4" FT_SIZES="$5" FT_EXTRA="$6" \
    FT_POST_TIMEOUT=7200 "$HERE/measure.sh" "$arm" || echo "$arm measure exit $?"
  kill $smi $st 2>/dev/null || true
  mv "$HERE/results/$arm-server.log" "$OUT/$arm-journal.txt" 2>/dev/null || true
  mv "$HERE"/results/$arm-* "$HERE"/results/$arm.env "$OUT/" 2>/dev/null || true
  # The acceptance lines: measured transient + fill, grows, captures, OOMs, tracebacks.
  local j="$OUT/$arm-journal.txt"
  { echo "captures=$(grep -c 'Start capturing CUDA graphs' "$j" || true)" \
         "kv_grows=$(grep -c 'KV grew' "$j" || true)" \
         "tracebacks=$(grep -c 'Traceback' "$j" || true)" \
         "ooms=$(grep -ci 'out of memory' "$j" || true)"
    grep -E "Prefill headroom|Expert arena parked|Growable-KV (ceiling|pre-commit|arena)|Committed growable|KV grew|capturing CUDA graphs|Free GPU memory|Free memory after|OutOfMemory|out of memory|Traceback|Prefill warmup|moe-cache-auto resolved" \
      "$j" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-400; } > "$OUT/$arm-headroom.txt" || true
  flock -u 9; exec 9>&-
  sleep 30
}
tag="r$(echo "$FT_RATIO" | tr -d .)"
for a in ${ARMS:-whole saver nemotron}; do
  case "$a" in
    whole)    run "ornith-whole-$tag" "$ORNITH" ornith 0 "$ORNITH_SIZES" "$ORNITH_BASE" ;;
    saver)    run "ornith-saver-$tag" "$ORNITH" ornith -1 "$ORNITH_SIZES" "$ORNITH_BASE" ;;
    nemotron) run "nemotron-whole-$tag" "$NEMOTRON" nemotron-3.5-lightning 0 "$NEMOTRON_SIZES" "" ;;
    *) echo "unknown arm $a"; exit 1 ;;
  esac
done
echo "headroom box done"
