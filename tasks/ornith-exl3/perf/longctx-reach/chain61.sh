#!/bin/bash
# exl3 worker 01fb928a: long-context reach fix (kv-floor-idle) on /root/FT-exl3p12 = p11 + growable_kv floor fix.
while pgrep -f "[c]hain60.sh" >/dev/null; do sleep 30; done
P11=/root/FT-exl3p11; P12=/root/FT-exl3p12; P8=/root/FT-exl3p8; T=/root/FT-os2/tasks/overlap-sync; J=/root/K/lc2/job-reach2.sh
for s in engine scheduler moe; do WT=$P12 $T/job-tests.sh p12-$s tests/$s; done
WT=$P12 CTX=256000 NUMTOK=262144 KVD=q8_0 $J p12-q8-256k
WT=$P11 CTX=256000 NUMTOK=262144 KVD=q4_0 $J p11-q4-256k
WT=$P12 CTX=256000 NUMTOK=262144 KVD=q4_0 $J p12-q4-256k
E=$P8/tasks/ornith-exl3/fuse/job-e2e.sh
for i in 1 2; do
  WT=$P8 PREROT=1 RESIDENCY=whole SIZES="8000" $E w8k-A$i
  WT=$P11 PREROT=1 RESIDENCY=whole SIZES="8000" $E w8k-B$i
done
WT=$P8 PREROT=1 RESIDENCY=whole SIZES="8000" $E w8k-A3
for arm in A1:$P11 B:$P12 A2:$P11; do
  n=${arm%%:*}; w=${arm#*:}
  for c in 128000 256000 384000; do WT=$w CTX=$c NUMTOK=393216 KVD=q8_0 YARN=2 $J ab-$n-$((c/1000))k; done
done
echo CHAIN61-DONE
