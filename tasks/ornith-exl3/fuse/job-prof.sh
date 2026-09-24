#!/bin/bash
# torch.profiler decode breakdown of Ornith EXL3 5.0bpw (saver, ratio 1.00) at ctx 8000 on this tree.
# Usage: job-prof.sh TAG   (flocks /root/gpu.lock; box ft-dev)
TAG=${1:-base}; R=$(cd "$(dirname "$0")/../../.." && pwd); O=/root/K/results/exl3prof-$TAG; mkdir -p $O
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
export FREETOKEN_EXL3_GEMV_PREROT=${PREROT:-0}  # base = the separate had_rows launch
export FREETOKEN_EXPERT_ARENA=1 FREETOKEN_ARENA_STEP_SLOTS=8 FREETOKEN_GROWABLE_OVERLAP=1 PYTHONPATH=$R/python
M=/root/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq
exec 9>/root/gpu.lock; flock 9
echo "=== exl3 prof $TAG prerot=$FREETOKEN_EXL3_GEMV_PREROT $(date -u +%FT%TZ) code $(git -C $R log --oneline -1) dirty=$(git -C $R status --porcelain python | wc -l)"
while pgrep -f "ft serve" >/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 16 ]; do sleep 5; done
cd $R
timeout 2400 /root/venv/bin/python tasks/ornith-exl3/perf/profile_step.py decode $O/decode-8k 8000 128 -- \
  --model $M --text-model-only --max-running-requests 1 --kv-grow-step-tokens 65536 \
  --num-tokens 262144 --max-seq-len-override 262144 --kv-cache-dtype q8_0 \
  --attention-backend triton --moe-strategy offload --moe-cache-auto --moe-cache-policy lfu \
  --expert-residency mirror --moe-mirror-host-rows -1 --memory-ratio 1.00 --max-prefill-length 8192 > $O/decode-8k.log 2>&1
echo "rc=$?"; head -40 $O/decode-8k.txt
echo "=== exl3 prof $TAG done $(date -u +%FT%TZ)"
