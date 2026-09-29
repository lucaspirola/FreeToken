#!/usr/bin/env bash
# ctx393 measurement chain (run as the systemd --user unit ctx393-chain, never from an agent shell).
# Every arm is arm.sh on :1920, q8_0 KV, ratio 1.00, under the GPU host lock; y-* = 393216 ceiling with
# YaRN factor 2 over 262144, c-* = same-day control at the gated 262144 without YaRN.
#   *-sp/*-wp  probe (fresh prefill, TTFT, decode), 2 passes, saver / whole
#   *-sn/*-wn  natural text (5 tasks x 3500 tokens)
#   y-sq/y-wq  needles: gate battery (needles.py + recall.py) and depth_needles.py at 300K-390K
# ARMS limits the arms (default all, in this order). Samplers: results/load5s.txt, results/mem5s.txt.
F=$(dirname "$(readlink -f "$0")"); R=$F/results; mkdir -p $R
export PATH=/usr/lib/wsl/lib:/usr/local/cuda/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
Y="CEIL=393216 YARN=2 YARN_ORIG=262144"
PY="1000 8000 32000 80000 128000 256000 262000 300000 380000"; PC="1000 8000 32000 80000 128000 256000"
( while [ ! -e $R/chain.stop ]; do
    echo "$(date +%T) $(cut -d' ' -f1-3 /proc/loadavg) $(nvidia-smi --query-gpu=clocks.sm,utilization.gpu --format=csv,noheader)" >> $R/load5s.txt
    echo "$(date +%T) gpu_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ') avail_gib=$(awk '/MemAvailable/{printf "%.2f", $2/1048576}' /proc/meminfo) swap_used_mib=$(awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{printf "%d", (t-f)/1024}' /proc/meminfo)" >> $R/mem5s.txt
    sleep 5; done ) &
rm -f $R/chain.stop
run() { local n=$1 rows=$2 s=$3; shift 3; env "$@" $F/arm.sh $n $rows "$s"; }
for a in ${ARMS:-c-sp y-sp y-wp c-wp y-sq y-wq c-sn y-sn y-wn}; do
  case $a in
    c-sp) run $a -1 "$PC" ;;
    y-sp) run $a -1 "$PY" $Y ;;
    y-wp) run $a 0 "$PY" $Y ;;
    c-wp) run $a 0 "$PC" ;;
    c-sn) run $a -1 8000 NAT=1 ;;
    y-sn) run $a -1 8000 NAT=1 $Y ;;
    y-wn) run $a 0 8000 NAT=1 $Y ;;
    y-sq) run $a -1 8000 NEEDLE=1 $Y ;;
    y-wq) run $a 0 8000 NEEDLE=1 $Y ;;
  esac
done
echo "CHAINDONE $(date -Is)" >> $R/status.txt
touch $R/chain.stop
