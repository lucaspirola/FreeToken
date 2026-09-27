#!/usr/bin/env bash
# Final numbers, Ornith EXL3 5.0bpw on the owner's RTX 5080 (run as a systemd --user unit).
#   chain.sh [PARTS]   PARTS: "1 2" (default)
# Part 1 (q8_0 KV, 262144): probe 8K/32K/80K/128K/256K x 2 passes, 512 tokens, whole/saver ABBA;
#         natural text whole/saver ABBA.
# Part 2 (saver): KV lanes q8q8 (reference), q8q6, q6q5, q4q4 (control) at a 256K ceiling (probe 8K
#         + 256K) and at 384K (--rope-yarn-factor 2, ceiling 393216, probe 8K + 384K), each with a
#         natural-text arm; q8q8 256K is repeated at the end as a drift check.
# Load / SM clock every 5 s -> results/load5s.txt.
set -u
F=$(dirname "$(readlink -f "$0")"); R=$F/results; mkdir -p $R; A=$F/arm.sh
rm -f $R/load.stop
( while [ ! -e $R/load.stop ]; do echo "$(date +%T) $(cut -d' ' -f1-3 /proc/loadavg) $(nvidia-smi --query-gpu=clocks.sm,utilization.gpu,memory.used --format=csv,noheader)" >> $R/load5s.txt; sleep 5; done ) &
st() { echo "$* $(date -Is)" >> $R/status.txt; }
lane_flags() { case $1 in q4q4) echo "--kv-cache-dtype q4_0";; *) echo "$1";; esac; }
for part in ${1:-1 2}; do
  case $part in
  1)
    st "part 1 start"
    for a in whole saver saver whole; do
      k=$(( $(ls $R/p1-$a-*-record.json 2>/dev/null | wc -l) + 1 ))
      rows=$([ $a = whole ] && echo 0 || echo -1)
      $A p1-$a-$k $rows "8000 32000 80000 128000 256000" FT_GEN=512
    done
    for a in whole saver saver whole; do
      k=$(( $(ls $R/p1nat-$a-*-record.json 2>/dev/null | wc -l) + 1 ))
      rows=$([ $a = whole ] && echo 0 || echo -1)
      NAT=1 $A p1nat-$a-$k $rows 8000
    done
    st "part 1 done" ;;
  2)
    st "part 2 start"
    for c in 256k 384k; do
      for l in q8q8 q8q6 q6q5 q4q4; do
        if [ $c = 256k ]; then FT_KV="$(lane_flags $l)" $A kv-$l-$c -1 "8000 256000" FT_GEN=512
        else CEIL=393216 YARN=2 FT_KV="$(lane_flags $l)" $A kv-$l-$c -1 "8000 384000" FT_GEN=512; fi
      done
      for l in q8q8 q8q6 q6q5 q4q4; do
        if [ $c = 256k ]; then NAT=1 FT_KV="$(lane_flags $l)" $A kvnat-$l-$c -1 8000
        else NAT=1 CEIL=393216 YARN=2 FT_KV="$(lane_flags $l)" $A kvnat-$l-$c -1 8000; fi
      done
    done
    FT_KV=q8q8 $A kv-q8q8-256k-rep -1 "8000 256000" FT_GEN=512
    st "part 2 done" ;;
  esac
done
touch $R/load.stop
st "CHAINDONE"
