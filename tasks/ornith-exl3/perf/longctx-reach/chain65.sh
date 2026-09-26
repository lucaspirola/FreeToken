#!/bin/bash
# exl3 worker 01fb928a: resume chain64 after the 2026-09-25 stop. p13 (kv-transient-record, record present) vs p11.
P11=/root/FT-exl3p11; P13=/root/FT-exl3p13; J=/root/K/lc2/job-reach2.sh
R="FREETOKEN_PREFILL_TRANSIENT_RECORD=/root/K/rec/p13.json"
for c in 128000 256000 384000; do WT=$P13 CTX=$c NUMTOK=393216 KVD=q8_0 YARN=2 ENVS="$R" $J ab-C-$((c/1000))k; done
for c in 128000 256000 384000; do WT=$P11 CTX=$c NUMTOK=393216 KVD=q8_0 YARN=2 $J ab-A3-$((c/1000))k; done
WT=$P13 CTX=384000 NUMTOK=393216 KVD=q4_0 YARN=2 ENVS="$R" $J p13-q4-384k
echo CHAIN65-DONE
