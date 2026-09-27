#!/usr/bin/env bash
# ABBA A/B round 1: lever A (sized prefill headroom) vs 757caee, saver and whole; whole also with
# --moe-prefill-hit-d2d. Probe 300/1000/8K/32K/80K then file-read extends. Output results/ab1/.
F=$(dirname "$(readlink -f "$0")"); B=/home/lucas/ai/FreeToken-wt/popt-base
export OUT=$F/results/ab1 EXT=1
S="300 1000 8000 32000 80000"
run() { local n=$1 tree=$2 rows=$3 more=${4:-}; TREE=$tree FT_EXTRA_MORE="$more" $F/arm.sh $n $rows "$S"; }
P=$(cd $F/../../.. && pwd)
run b-s1 $B -1; run n-s1 $P -1; run n-s2 $P -1; run b-s2 $B -1
run b-w1 $B 0; run n-w1 $P 0; run h-w1 $P 0 --moe-prefill-hit-d2d; run h-w2 $P 0 --moe-prefill-hit-d2d; run n-w2 $P 0; run b-w2 $B 0
echo AB1DONE >> $OUT/status.txt
