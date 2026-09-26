#!/bin/bash
# exl3 worker 01fb928a: exact VRAM ledger of the refused 256K q8_0 and q4_0 grows on p11 (diagnostic).
J=/root/K/lc2/job-ledger.sh; P11=/root/FT-exl3p11
WT=$P11 CTX=256000 NUMTOK=262144 KVD=q8_0 $J ledger-p11-q8-256k
WT=$P11 CTX=256000 NUMTOK=262144 KVD=q4_0 $J ledger-p11-q4-256k
echo CHAIN62-DONE
