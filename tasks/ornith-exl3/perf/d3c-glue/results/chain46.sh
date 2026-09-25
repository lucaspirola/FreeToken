#!/bin/bash
# D3c: shared-expert gate on the side stream, packed-key router top-k, fused mirror pre-ensure
W=/root/FT-exl3p8; B=/root/FT-exl3p6; J=$W/tasks/ornith-exl3/fuse
MT="tests/moe/test_fused_moe.py tests/moe/test_mirror_device.py tests/moe/test_mirror_prefill.py tests/moe/test_mirror_pool_golden.py tests/moe/test_mirror_dma_writeback.py"
WT=$W $J/job-tests.sh d3c $MT
grep -q " failed\| error" /root/K/results/exl3tests-d3c/pytest.txt && { WT=$B $J/job-tests.sh d3c-base $MT; echo "tests not green (base run above for comparison), stop"; exit 1; }
( exec 9>/root/gpu.lock; flock 9
  while pgrep -f "ft serve" >/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 16 ]; do sleep 5; done
  mkdir -p /root/K/results/exl3bench-d3c; cd $W
  PYTHONPATH=$W/python CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH timeout 900 /root/venv/bin/python tasks/ornith-exl3/perf/d3c-glue/bench_glue.py 2>&1 | tee /root/K/results/exl3bench-d3c/bench.txt )
grep -q "MISMATCH" /root/K/results/exl3bench-d3c/bench.txt && { echo "router not bit-equal, stop"; exit 1; }
WT=$W TAG=d3c SIZE="62000" GREEDY_REPEATS=1 $J/job-greedy-f16acc.sh ref
a=$(python3 -c "import json;print(json.loads(open(\"/root/K/results/exl3greedy-f16acc-d3c/ref.jsonl\").readline())[\"sha1\"])")
echo "greedy d3c $a ref ed5c265a64dc237189e1b67010af7f17a8799d8f"; [ "$a" = ed5c265a64dc237189e1b67010af7f17a8799d8f ] || { echo "output changed, stop"; exit 1; }
WT=$B PREROT=1 SIZES="8000 32000 80000" $J/job-e2e.sh d3c-base1
WT=$W PREROT=1 SIZES="8000 32000 80000" $J/job-e2e.sh d3c
WT=$B PREROT=1 SIZES="8000 32000 80000" $J/job-e2e.sh d3c-base2
WT=$B RESIDENCY=whole PREROT=1 SIZES="8000 32000 80000" $J/job-e2e.sh d3c-whole-base
WT=$W RESIDENCY=whole PREROT=1 SIZES="8000 32000 80000" $J/job-e2e.sh d3c-whole
PREROT=1 $J/job-prof.sh d3c
echo CHAIN46-DONE
