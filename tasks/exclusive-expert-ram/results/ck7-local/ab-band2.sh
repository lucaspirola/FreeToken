#!/usr/bin/env bash
# Round 5 policy A/B, part 2: tb1 (DUP_BAND=0, one-bucket tie-break) vs band2 (DUP_BAND=2: the
# noise band floor(sqrt(min)) WITHOUT the minimum of 1), --moe-collect-stats. Probe 8K+80K x2
# (tb1 band2 band2 tb1), then natural text: whole tb1 band2 band2 tb1 whole.
A=/home/lucas/ai/FreeToken-wt/round5/tasks/exclusive-expert-ram/results/ck7-local/arm.sh; D=/home/lucas/ai/FreeToken-wt/round5/tasks/exclusive-expert-ram/results/ck7-local/policy2
export FT_EXTRA_MORE=--moe-collect-stats
k=0
for p in tb1 band2 band2 tb1; do
  k=$((k+1)); e=FREETOKEN_MIRROR_DUP_BAND=0; [ $p = band2 ] && e=FREETOKEN_MIRROR_DUP_BAND=2
  FT_ENVS_EXTRA=$e $A /home/lucas/ai/FreeToken-wt/round5 pol2-$k-$p -1 "8000 80000" $D FT_GEN=512
done
k=0
for p in whole tb1 band2 band2 tb1 whole; do
  k=$((k+1)); e=FREETOKEN_MIRROR_DUP_BAND=0; rows=-1
  [ $p = band2 ] && e=FREETOKEN_MIRROR_DUP_BAND=2; [ $p = whole ] && rows=0
  n=pol2-nat$k-$p
  FT_ENVS_EXTRA=$e $A /home/lucas/ai/FreeToken-wt/round5 $n $rows 8000 $D FT_GEN=128 FT_POST_TIMEOUT=5400     "FT_POST=python3 /home/lucas/ai/FreeToken-wt/harvest/tasks/harvest/belady/natural_gen.py --doc /home/lucas/ai/FreeToken-wt/round5/docs/nemotron.md --doc /home/lucas/ai/FreeToken-wt/round5/docs/cli.md --max-tokens 3500 --out $D/$n-natural.json --texts $D/$n"
done
echo POL2DONE >> $D/status.txt
