#!/bin/bash
# D3c: shared-expert gate joins the shared expert on the side stream; wall-clock profile split
W=/root/FT-exl3p8; B=/root/FT-exl3p6; J=$W/tasks/ornith-exl3/fuse
WT=$W $J/job-tests.sh d3c2 tests/moe/test_fused_moe.py
grep -q " failed\| error" /root/K/results/exl3tests-d3c2/pytest.txt && { echo "tests not green, stop"; exit 1; }
WT=$W TAG=d3c2 SIZE="62000" GREEDY_REPEATS=1 $J/job-greedy-f16acc.sh ref
a=$(python3 -c "import json;print(json.loads(open(\"/root/K/results/exl3greedy-f16acc-d3c2/ref.jsonl\").readline())[\"sha1\"])")
echo "greedy d3c2 $a ref ed5c265a64dc237189e1b67010af7f17a8799d8f"; [ "$a" = ed5c265a64dc237189e1b67010af7f17a8799d8f ] || { echo "output changed, stop"; exit 1; }
PREROT=1 $B/tasks/ornith-exl3/fuse/job-prof.sh d3c2-base
PREROT=1 $J/job-prof.sh d3c2
WT=$B PREROT=1 SIZES="8000 32000 80000" $J/job-e2e.sh d3c2-base1
WT=$W PREROT=1 SIZES="8000 32000 80000" $J/job-e2e.sh d3c2
WT=$B PREROT=1 SIZES="8000 32000 80000" $J/job-e2e.sh d3c2-base2
WT=$B RESIDENCY=whole PREROT=1 SIZES="8000 32000 80000" $J/job-e2e.sh d3c2-whole-base
WT=$W RESIDENCY=whole PREROT=1 SIZES="8000 32000 80000" $J/job-e2e.sh d3c2-whole
echo CHAIN47-DONE
