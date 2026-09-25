#!/bin/bash
# mirror worker 2026-09-25: dt128m (dh2 last arm), then lfunat resume from nA2. Each takes gpu.lock.
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
wait_idle() {
  while pgrep -f "[f]t serve" >/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 16 ]; do sleep 5; done
}
exec 9>/root/gpu.lock; flock 9
wait_idle
echo "dt128m start $(date -u +%T) gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader) $(uptime)" >> /root/dh2-status
cd /root/FT-dh
FT_GEN=512 FT_EXTRA="--moe-collect-stats" FT_ENVS="FREETOKEN_DECODE_MEM_PROBE=1 FREETOKEN_DECODE_FREE_TARGET_MB=128" \
  ARMS=mirror-1m tasks/exclusive-expert-ram/checkpoint-box.sh dt128m > /root/dh2-1m.log 2>&1
echo "dt128m rc=$? $(date -u +%T)" >> /root/dh2-status
echo ALLDONE >> /root/dh2-status
wait_idle
flock -u 9; exec 9>&-
sleep 20
/root/belady/lfunat-resume.sh
