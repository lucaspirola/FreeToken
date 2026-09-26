#!/bin/bash
# exl3 worker 01fb928a: p14 = p11 + transient record + prediction-bound pricing + margin check. First starts, fresh records.
P14=/root/FT-exl3p14; T=/root/FT-os2/tasks/overlap-sync; J=/root/K/lc2/job-reach2.sh
for s in engine scheduler; do WT=$P14 $T/job-tests.sh p14-$s tests/$s; done
rm -f /root/K/rec/p14-*.json
WT=$P14 CTX=256000 NUMTOK=262144 KVD=q8_0 ENVS="FREETOKEN_PREFILL_TRANSIENT_RECORD=/root/K/rec/p14-a.json" $J p14-q8-256k-first
WT=$P14 CTX=256000 NUMTOK=262144 KVD=q4_0 ENVS="FREETOKEN_PREFILL_TRANSIENT_RECORD=/root/K/rec/p14-b.json" $J p14-q4-256k-first
WT=$P14 CTX=384000 NUMTOK=393216 KVD=q8_0 YARN=2 ENVS="FREETOKEN_PREFILL_TRANSIENT_RECORD=/root/K/rec/p14-c.json" $J p14-q8-384k-first
echo CHAIN66-DONE
