#!/bin/bash
cd /root/FreeToken
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
export FREETOKEN_EXPERT_ARENA=1 FREETOKEN_ARENA_STEP_SLOTS=8 FREETOKEN_GROWABLE_OVERLAP=1
M=/root/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq
flock /root/gpu.lock timeout 2400 /root/venv/bin/python tasks/ornith-exl3/perf/profile_prefill.py /root/prof-8k 8000 -- \
  --model $M --text-model-only --max-running-requests 1 --kv-grow-step-tokens 65536 \
  --num-tokens 262144 --max-seq-len-override 262144 --kv-cache-dtype q8_0 \
  --attention-backend triton --moe-strategy offload --moe-cache-auto --moe-cache-policy lfu \
  --expert-residency mirror --moe-mirror-host-rows -1 --memory-ratio 1.00 --max-prefill-length 8192 > /root/prof-8k.log 2>&1
echo "prof rc=$?" > /root/prof-status
