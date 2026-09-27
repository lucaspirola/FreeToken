#!/bin/bash
# D3a v2: side-stream shared expert with its own split-K counter lane
W=/root/FT-exl3p6; J=$W/tasks/ornith-exl3/fuse
WT=$W $J/job-tests.sh d3a2
grep -q " passed" /root/K/results/exl3tests-d3a2/pytest.txt && ! grep -q " failed\| error" /root/K/results/exl3tests-d3a2/pytest.txt || { echo "tests not green, stop"; exit 1; }
FREETOKEN_SHARED_EXPERT_OVERLAP=1 WT=$W TAG=ov1b SIZE="62000" GREEDY_REPEATS=1 $J/job-greedy-f16acc.sh ref
FREETOKEN_SHARED_EXPERT_OVERLAP=0 WT=$W TAG=ov0b SIZE="62000" GREEDY_REPEATS=1 $J/job-greedy-f16acc.sh ref
a=$(python3 -c "import json;print(json.loads(open(\"/root/K/results/exl3greedy-f16acc-ov1b/ref.jsonl\").readline())[\"sha1\"])"); b=$(python3 -c "import json;print(json.loads(open(\"/root/K/results/exl3greedy-f16acc-ov0b/ref.jsonl\").readline())[\"sha1\"])")
echo "greedy ov1 $a ov0 $b"; [ "$a" = "$b" ] || { echo "overlap changes the output, stop"; exit 1; }
for t in 0 1 0; do FREETOKEN_SHARED_EXPERT_OVERLAP=$t WT=$W PREROT=1 SIZES="8000 32000 80000" $J/job-e2e.sh d3a2-ov$t-$RANDOM; done
for t in 0 1; do FREETOKEN_SHARED_EXPERT_OVERLAP=$t RESIDENCY=whole WT=$W PREROT=1 SIZES="8000 32000 80000" $J/job-e2e.sh d3a2-whole-ov$t; done
FREETOKEN_SHARED_EXPERT_OVERLAP=1 PREROT=1 $J/job-prof.sh d3a2
echo CHAIN44-DONE
