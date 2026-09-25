#!/bin/bash
# Decode regression, step 2: nsys of one 1M decode window (whole arm) on base (reorg-next e2f136f)
# and round 2 (01548e7), same script, skip 32 / window 200 tokens. Output in scratchpad/nsys-dec.
S=/tmp/claude-1000/-home-lucas-ai-FreeToken/606da56c-cd31-49f4-adb4-b4ab53d18508/scratchpad/nsys-dec
NS_OUT=$S NS_ARMS=whole NS_TAG=-base /home/lucas/ai/FreeToken-wt/reorg-next/tasks/exclusive-expert-ram/nsys1m-local.sh
NS_OUT=$S NS_ARMS=whole NS_TAG=-r2 /home/lucas/ai/FreeToken-wt/round2/tasks/exclusive-expert-ram/nsys1m-local.sh
H=/home/lucas/ai/FreeToken-wt/round2/tasks/exclusive-expert-ram
python3 $H/results/ck4dma-g5/ck4dma-nsys/nsys_graph_nodes.py $S/whole-base.sqlite $S/whole-r2.sqlite > $S/in-graph.txt 2>&1
python3 $H/nsys_overlap.py $S/whole-base.sqlite $S/whole-r2.sqlite > $S/overlap.txt 2>&1
echo NSYS-DEC-DONE
