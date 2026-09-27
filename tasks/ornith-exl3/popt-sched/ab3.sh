#!/usr/bin/env bash
# Decode round (no-regression check, ~2% bar): 8K/80K/256K probe decode, then natural text
# (5 tasks, 3500 tokens), ABBA new tree vs 757caee, saver and whole. Output results/ab3/.
F=$(dirname "$(readlink -f "$0")"); B=/home/lucas/ai/FreeToken-wt/popt-base; P=$(cd $F/../../.. && pwd)
export OUT=$F/results/ab3
S="8000 80000 256000"
run() { local n=$1 tree=$2 rows=$3; TREE=$tree $F/arm.sh $n $rows "$S"; }
nat() { local n=$1 tree=$2 rows=$3; TREE=$tree NAT=1 $F/arm.sh $n $rows "8000"; }
run b-sd1 $B -1; run n-sd1 $P -1; run n-sd2 $P -1; run b-sd2 $B -1
run b-wd1 $B 0; run n-wd1 $P 0; run n-wd2 $P 0; run b-wd2 $B 0
nat b-sn1 $B -1; nat n-sn1 $P -1; nat n-sn2 $P -1; nat b-sn2 $B -1
nat b-wn1 $B 0; nat n-wn1 $P 0; nat n-wn2 $P 0; nat b-wn2 $B 0
echo AB3DONE >> $OUT/status.txt
