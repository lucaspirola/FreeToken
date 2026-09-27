#!/usr/bin/env bash
# Saver victim-policy A/B on Ornith EXL3 262K (round 5), --moe-collect-stats on every arm so the
# decode hit rate is comparable with the whole model's. Policies:
#   band  = default at bf1a48e (FREETOKEN_MIRROR_DUP_BAND=1)
#   tb1   = DUP_BAND=0: the one-bucket (+1) pool-row tie-break
#   exact = DUP_BAND=0 + FREETOKEN_MIRROR_TIEBREAK_EXACT=1: pool row breaks exact count ties only
# Probe 8K+80K 512 tokens, order whole band tb1 exact exact tb1 band whole; then natural text once each.
A=/home/lucas/ai/FreeToken-wt/round5/tasks/exclusive-expert-ram/results/ck7-local/arm.sh; D=/home/lucas/ai/FreeToken-wt/round5/tasks/exclusive-expert-ram/results/ck7-local/policy
export FT_EXTRA_MORE=--moe-collect-stats
run() { local n=$1 rows=$2; shift 2; FT_ENVS_EXTRA="$*" $A /home/lucas/ai/FreeToken-wt/round5 $n $rows "8000 80000" $D FT_GEN=512; }
env_of() { case $1 in band) echo FREETOKEN_MIRROR_DUP_BAND=1;; tb1) echo FREETOKEN_MIRROR_DUP_BAND=0;; exact) echo FREETOKEN_MIRROR_DUP_BAND=0 FREETOKEN_MIRROR_TIEBREAK_EXACT=1;; esac; }
k=0
for p in whole band tb1 exact exact tb1 band whole; do
  k=$((k+1)); if [ $p = whole ]; then run pol-$k-whole 0; else run pol-$k-$p -1 $(env_of $p); fi
done
for p in whole band tb1 exact; do
  n=pol-nat-$p; rows=-1; [ $p = whole ] && rows=0
  FT_ENVS_EXTRA="$(env_of $p)" $A /home/lucas/ai/FreeToken-wt/round5 $n $rows 8000 $D FT_GEN=128 FT_POST_TIMEOUT=5400     "FT_POST=python3 /home/lucas/ai/FreeToken-wt/harvest/tasks/harvest/belady/natural_gen.py --doc /home/lucas/ai/FreeToken-wt/round5/docs/nemotron.md --doc /home/lucas/ai/FreeToken-wt/round5/docs/cli.md --max-tokens 3500 --out $D/$n-natural.json --texts $D/$n"
done
echo POLDONE >> $D/status.txt
