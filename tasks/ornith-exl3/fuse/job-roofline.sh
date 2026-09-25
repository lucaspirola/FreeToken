#!/bin/bash
# Roofline microbench (ours + exllamav3) and E2E torch.profiler breakdowns, Ornith EXL3 5.0bpw, whole model
# (no RAM saver) and saver, ratio 1.00. Usage: WT=<tree> job-roofline.sh TAG [bench|prof|all]
TAG=${1:?tag}; WHAT=${2:-all}; W=${WT:?tree}; O=/root/K/results/exl3roof-$TAG; mkdir -p $O
cd $W; export PYTHONPATH=$W/python
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
export FREETOKEN_EXPERT_ARENA=1 FREETOKEN_ARENA_STEP_SLOTS=8 FREETOKEN_GROWABLE_OVERLAP=1
M=/root/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq
FLAGS="--model $M --text-model-only --max-running-requests 1 --kv-grow-step-tokens 65536 --num-tokens 262144 --max-seq-len-override 262144 --kv-cache-dtype q8_0 --attention-backend triton --moe-strategy offload --moe-cache-auto --moe-cache-policy lfu --memory-ratio 1.00 --max-prefill-length 8192"
SAVER="--expert-residency mirror --moe-mirror-host-rows -1"
idle() { while pgrep -f "ft serve" >/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 16 ]; do sleep 5; done; }
if [ $WHAT = bench ] || [ $WHAT = all ]; then
  ( flock 9; idle
    echo "=== exl3 roofline bench $TAG $(date -u +%FT%TZ) code $(git -C $W log --oneline -1) dirty=$(git -C $W status --porcelain python | wc -l)"
    timeout 1200 /root/venv/bin/python tasks/ornith-exl3/perf/bench_roofline.py --json $O/roofline.jsonl > $O/roofline.txt 2>&1; echo "roofline rc=$?"
    PYTHONPATH=/root/exllamav3-src timeout 1200 /root/exl3-ref/bin/python tasks/ornith-exl3/perf/bench_roofline_exllamav3.py 8192 > $O/exllamav3.txt 2>&1; echo "exllamav3 rc=$?"
  ) 9>/root/gpu.lock
fi
if [ $WHAT = prof ] || [ $WHAT = all ]; then
  for mode in whole saver; do
    X=""; [ $mode = saver ] && X="$SAVER"
    for spec in "decode 8000 128" "decode 80000 128" "prefill 80000"; do
      set -- $spec; name=$1-$(( $2 / 1000 ))k-$mode
      ( flock 9; idle
        echo "=== exl3 prof $TAG $name $(date -u +%FT%TZ)"
        timeout 1800 /root/venv/bin/python tasks/ornith-exl3/perf/profile_step.py $1 $O/$name ${@:2} -- $FLAGS $X > $O/$name.log 2>&1; echo "$name rc=$?"
      ) 9>/root/gpu.lock
    done
  done
fi
echo "=== exl3 roofline $TAG done $(date -u +%FT%TZ)"
