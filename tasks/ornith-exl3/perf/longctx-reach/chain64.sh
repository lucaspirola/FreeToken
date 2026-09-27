#!/bin/bash
# exl3 worker 01fb928a: kv-transient-record on /root/FT-exl3p13 (= p11 + engine.py record/validation). Fresh record file.
while pgrep -f "[c]hain61.sh|[c]hain63.sh" >/dev/null; do sleep 30; done
P11=/root/FT-exl3p11; P13=/root/FT-exl3p13; T=/root/FT-os2/tasks/overlap-sync; J=/root/K/lc2/job-reach2.sh
for s in engine scheduler moe; do WT=$P13 $T/job-tests.sh p13-$s tests/$s; done
R="FREETOKEN_PREFILL_TRANSIENT_RECORD=/root/K/rec/p13.json"; rm -f /root/K/rec/p13.json
WT=$P13 CTX=256000 NUMTOK=262144 KVD=q8_0 ENVS="$R" $J p13-q8-256k-start1
cp /root/K/rec/p13.json /root/K/rec/p13-after-start1.json 2>/dev/null
WT=$P13 CTX=256000 NUMTOK=262144 KVD=q8_0 ENVS="$R" $J p13-q8-256k-start2
WT=$P13 CTX=256000 NUMTOK=262144 KVD=q4_0 ENVS="$R" $J p13-q4-256k-start1
WT=$P13 CTX=256000 NUMTOK=262144 KVD=q4_0 ENVS="$R" $J p13-q4-256k-start2
cp /root/K/rec/p13.json /root/K/rec/p13-after-q4.json
for c in 128000 256000 384000; do WT=$P13 CTX=$c NUMTOK=393216 KVD=q8_0 YARN=2 ENVS="$R" $J ab-C-$((c/1000))k; done
WT=$P13 CTX=384000 NUMTOK=393216 KVD=q4_0 YARN=2 ENVS="$R" $J p13-q4-384k
for c in 128000 256000 384000; do WT=$P11 CTX=$c NUMTOK=393216 KVD=q8_0 YARN=2 $J ab-A3-$((c/1000))k; done
echo CHAIN64-DONE
