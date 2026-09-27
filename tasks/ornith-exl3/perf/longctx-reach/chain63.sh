#!/bin/bash
# exl3 worker 01fb928a: discriminating test -- pool sized for the MEASURED transient (1146 MiB) instead of the estimate.
while pgrep -f "[c]hain62.sh" >/dev/null; do sleep 30; done
J=/root/K/lc2/job-reach2.sh; P11=/root/FT-exl3p11
WT=$P11 CTX=256000 NUMTOK=262144 KVD=q8_0 ENVS="FREETOKEN_PREFILL_TRANSIENT_MB=1146" $J p11-q8-256k-tmb
WT=$P11 CTX=256000 NUMTOK=262144 KVD=q4_0 ENVS="FREETOKEN_PREFILL_TRANSIENT_MB=1146" $J p11-q4-256k-tmb
echo CHAIN63-DONE
