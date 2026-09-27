#!/bin/bash
cd /root/FreeToken
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0 FREETOKEN_EXPERT_ARENA=1
M=/root/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq
COMMON="--model $M --text-model-only --max-running-requests 1 --moe-strategy offload --moe-cache-auto --moe-cache-policy lfu --attention-backend triton --memory-ratio 0.95 --max-seq-len-override 65536 --max-prefill-length 8192"
run() { # tag repeat residency...
  local tag=$1 rep=$2; shift 2
  FT_PROMPT_REPEAT=$rep flock /root/gpu.lock timeout 1800 /root/venv/bin/python tasks/ornith-exl3/compare/ft_logits.py /root/s2-$tag.pt 32 -- $COMMON "$@" > /root/s2-$tag.log 2>&1
  echo "$tag rc=$?" >> /root/s2-status
}
rm -f /root/s2-status
run short-mirror 1 --expert-residency mirror --moe-mirror-host-rows -1
run short-whole 1 --expert-residency whole
run long-mirror 40 --expert-residency mirror --moe-mirror-host-rows -1
run long-whole 40 --expert-residency whole
echo DONE >> /root/s2-status
