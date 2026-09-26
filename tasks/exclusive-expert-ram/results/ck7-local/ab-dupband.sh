#!/usr/bin/env bash
# Saver victim policy A/B (8K+80K, 512 tokens, PROBE_STATS): default (DUP_BAND + tie-break on),
# DUP_BAND off, DUP_BAND and tie-break off; whole model last for the bracket.
A=/home/lucas/ai/FreeToken-wt/round5/tasks/exclusive-expert-ram/results/ck7-local/arm.sh; D=/home/lucas/ai/FreeToken-wt/round5/tasks/exclusive-expert-ram/results/ck7-local/dupband
FT_ENVS_EXTRA= $A /home/lucas/ai/FreeToken-wt/round5 base -1 "8000 80000" $D FT_GEN=512
FT_ENVS_EXTRA=FREETOKEN_MIRROR_DUP_BAND=0 $A /home/lucas/ai/FreeToken-wt/round5 db0 -1 "8000 80000" $D FT_GEN=512
FT_ENVS_EXTRA="FREETOKEN_MIRROR_DUP_BAND=0 FREETOKEN_MIRROR_TIEBREAK=0" $A /home/lucas/ai/FreeToken-wt/round5 db0tb0 -1 "8000 80000" $D FT_GEN=512
FT_ENVS_EXTRA= $A /home/lucas/ai/FreeToken-wt/round5 whole 0 "8000 80000" $D FT_GEN=512
echo ABDONE >> $D/status.txt
