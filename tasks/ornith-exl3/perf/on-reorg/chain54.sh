#!/bin/bash
# exl3 worker 01fb928a: EXL3 re-merge exp/exl3-on-reorg 2fb1026 (= exp/reorg 3b19f4a round 2 + exp/ornith-exl3 f980722)
# on /root/FT-exl3p11. ft-dev suite, greedy output vs p9 (a43622edc2, 114K prompt, 1023 tokens), then the short (a)/(b)
# table: saver and whole at 8K/32K/80K, A = p8 (exp/ornith-exl3 52020bc content) bracketing B = p11.
while pgrep -f "chain5[3].sh" >/dev/null; do sleep 30; done
P11=/root/FT-exl3p11; P8=/root/FT-exl3p8; T=/root/FT-os2/tasks/overlap-sync
for s in moe engine scheduler kernels; do WT=$P11 $T/job-tests.sh p11-$s tests/$s; done
WT=$P11 TAG=onreorg $P8/tasks/ornith-exl3/fuse/job-greedy-f16acc.sh ref
E=$P8/tasks/ornith-exl3/fuse/job-e2e.sh
for R in mirror whole; do
  WT=$P8 PREROT=1 RESIDENCY=$R SIZES="8000 32000 80000" $E onreorg-$R-A1
  WT=$P11 PREROT=1 RESIDENCY=$R SIZES="8000 32000 80000" $E onreorg-$R-B
  WT=$P8 PREROT=1 RESIDENCY=$R SIZES="8000 32000 80000" $E onreorg-$R-A2
done
echo CHAIN54-DONE
