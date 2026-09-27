#!/usr/bin/env bash
# Idle-seed A/B (exp/r5-seed fc66bf1 = round 5 e472d20 + MirrorResidency.idle_seed), after the
# ck7o gate. --moe-collect-stats + PROBE_STATS on every arm. Arms:
#   whole  = round 5, whole model            tb1   = round 5 saver (tie-break, no seeding)
#   seed   = r5-seed saver (tie-break + idle seeding)
#   seed0  = r5-seed saver, FREETOKEN_MIRROR_TIEBREAK=0 (pure LFU victims + idle seeding)
# A: 8K+80K 512 tokens, whole tb1 seed seed0 seed0 seed tb1 whole. B: natural text once each.
# C: 8K+256K whole and seed.
while systemctl --user is-active --quiet ft-r5-gate; do sleep 30; done
A=/home/lucas/ai/FreeToken-wt/round5/tasks/exclusive-expert-ram/results/ck7-local/arm.sh; D=/home/lucas/ai/FreeToken-wt/round5/tasks/exclusive-expert-ram/results/ck7-local/seed
export FT_EXTRA_MORE=--moe-collect-stats
wt_of() { case $1 in whole|tb1) echo /home/lucas/ai/FreeToken-wt/round5;; *) echo /home/lucas/ai/FreeToken-wt/r5-seed;; esac; }
rows_of() { [ $1 = whole ] && echo 0 || echo -1; }
env_of() { [ $1 = seed0 ] && echo FREETOKEN_MIRROR_TIEBREAK=0; }
k=0
for p in whole tb1 seed seed0 seed0 seed tb1 whole; do
  k=$((k+1)); FT_ENVS_EXTRA="$(env_of $p)" $A $(wt_of $p) sd-$k-$p $(rows_of $p) "8000 80000" $D FT_GEN=512
done
for p in whole tb1 seed seed0; do
  n=sd-nat-$p
  FT_ENVS_EXTRA="$(env_of $p)" $A $(wt_of $p) $n $(rows_of $p) 8000 $D FT_GEN=128 FT_POST_TIMEOUT=5400     "FT_POST=python3 /home/lucas/ai/FreeToken-wt/harvest/tasks/harvest/belady/natural_gen.py --doc /home/lucas/ai/FreeToken-wt/round5/docs/nemotron.md --doc /home/lucas/ai/FreeToken-wt/round5/docs/cli.md --max-tokens 3500 --out $D/$n-natural.json --texts $D/$n"
done
for p in whole seed; do
  FT_ENVS_EXTRA="$(env_of $p)" $A $(wt_of $p) sd-256k-$p $(rows_of $p) "8000 256000" $D
done
mv /home/lucas/ai/FreeToken-wt/round5/tasks/exclusive-expert-ram/results/sd-*.env /home/lucas/ai/FreeToken-wt/r5-seed/tasks/exclusive-expert-ram/results/sd-*.env $D/ 2>/dev/null
echo SEEDDONE >> $D/status.txt
