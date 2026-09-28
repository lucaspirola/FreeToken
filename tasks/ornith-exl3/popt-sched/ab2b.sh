#!/usr/bin/env bash
# ABBA A/B round 2b: 5012f6c + prime only where it cannot drain the GPU, vs 757caee; whole then saver.
# --moe-prefill-hit-d2d. Probe 300/1000/8K/32K/80K then file-read extends. Output results/ab2b/.
F=$(dirname "$(readlink -f "$0")"); B=/home/lucas/ai/FreeToken-wt/popt-base
export OUT=$F/results/ab2b EXT=1
S="300 1000 8000 32000 80000"
run() { local n=$1 tree=$2 rows=$3 more=${4:-}; TREE=$tree FT_EXTRA_MORE="$more" $F/arm.sh $n $rows "$S"; }
P=$(cd $F/../../.. && pwd)
run b-w1 $B 0; run n-w1 $P 0; run n-w2 $P 0; run b-w2 $B 0
run n-s3 $P -1; run b-s3 $B -1; run b-s4 $B -1; run n-s4 $P -1
echo AB2BDONE >> $OUT/status.txt
