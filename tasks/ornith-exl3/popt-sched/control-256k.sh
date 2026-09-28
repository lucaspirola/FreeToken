#!/usr/bin/env bash
# Same-day control for the gate's 256K p1 saver/whole point (ck9s 88.3%): 256K pairs in the gate's
# shape ("8000 256000"), this tree and 757caee, alternated. Output results/control-256k/.
F=$(dirname "$(readlink -f "$0")"); B=/home/lucas/ai/FreeToken-wt/popt-base; P=$(cd $F/../../.. && pwd)
export OUT=$F/results/control-256k
S="8000 256000"
run() { TREE=$2 $F/arm.sh $1 $3 "$S"; }
run cn-w1 $P 0; run cn-s1 $P -1; run cb-s1 $B -1; run cb-w1 $B 0
run cn-s2 $P -1; run cn-w2 $P 0; run cb-w2 $B 0; run cb-s2 $B -1
echo CTRLDONE >> $OUT/status.txt
