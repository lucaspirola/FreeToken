#!/bin/bash
# exl3 worker 01fb928a: Ornith EXL3 saver long-context reach on exp/exl3-on-reorg (p11). q8_0 at 256K and 384K, q4_0 at 384K.
J=/root/K/lc2/job-reach.sh; P=/root/FT-exl3p11
WT=$P CTX=256000 NUMTOK=262144 KVD=q8_0 $J p11-q8-256k
WT=$P CTX=384000 NUMTOK=393216 KVD=q8_0 YARN=2 $J p11-q8-384k
WT=$P CTX=384000 NUMTOK=393216 KVD=q4_0 YARN=2 $J p11-q4-384k
echo CHAIN60-DONE
