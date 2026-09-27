#!/usr/bin/env bash
# Re-runs after chain.sh, because the owner's host was loaded (1-min load means 4-12.5 in
# several arm windows; the q8q8 256K repeat at load 8.2 ran 10% slower than the same arm at 1.5).
# Before each arm: wait for a quiet host (1-min load < 3), polling every 60 s for up to 30 min,
# then run anyway; the table reports every arm's window load, and the README uses the quiet ones.
# Part 1: probe ABBA x2 more and natural ABBA x2 more (numbered after chain.sh's arms).
# Part 2: fresh q8q8 references at 256K and 384K, and the lane arms that ran loaded.
set -u
F=$(dirname "$(readlink -f "$0")"); R=$F/results; A=$F/arm.sh
while systemctl --user is-active --quiet ft-final-profile; do sleep 30; done
rm -f $R/load.stop
( while [ ! -e $R/load.stop ]; do echo "$(date +%T) $(cut -d' ' -f1-3 /proc/loadavg) $(nvidia-smi --query-gpu=clocks.sm,utilization.gpu,memory.used --format=csv,noheader)" >> $R/load5s.txt; sleep 5; done ) &
st() { echo "$* $(date -Is)" >> $R/status.txt; }
quiet() { for i in $(seq 30); do awk '{exit !($1 < 3.0)}' /proc/loadavg && return; sleep 60; done; st "not quiet after 30 min: running anyway"; }
lane_flags() { case $1 in q4q4) echo "--kv-cache-dtype q4_0";; *) echo "$1";; esac; }
st "chain2 start"
for a in whole saver saver whole; do
  k=$(( $(ls $R/p1-$a-*-record.json 2>/dev/null | wc -l) + 1 )); rows=$([ $a = whole ] && echo 0 || echo -1)
  quiet; $A p1-$a-$k $rows "8000 32000 80000 128000 256000" FT_GEN=512
done
for a in whole saver saver whole; do
  k=$(( $(ls $R/p1nat-$a-*-record.json 2>/dev/null | wc -l) + 1 )); rows=$([ $a = whole ] && echo 0 || echo -1)
  quiet; NAT=1 $A p1nat-$a-$k $rows 8000
done
quiet; FT_KV=q8q8 $A kv-q8q8-256k-r2 -1 "8000 256000" FT_GEN=512
quiet; FT_KV=q6q5 $A kv-q6q5-256k-r2 -1 "8000 256000" FT_GEN=512
for l in q8q8 q8q6; do quiet; NAT=1 FT_KV="$(lane_flags $l)" $A kvnat-$l-256k-r2 -1 8000; done
quiet; CEIL=393216 YARN=2 FT_KV=q8q8 $A kv-q8q8-384k-r2 -1 "8000 384000" FT_GEN=512
for l in q8q6 q4q4; do quiet; CEIL=393216 YARN=2 FT_KV="$(lane_flags $l)" $A kv-$l-384k-r2 -1 "8000 384000" FT_GEN=512; done
quiet; NAT=1 CEIL=393216 YARN=2 FT_KV="$(lane_flags q4q4)" $A kvnat-q4q4-384k-r2 -1 8000
touch $R/load.stop
st "CHAIN2DONE"
