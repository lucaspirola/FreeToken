#!/bin/bash
cd /root/FreeToken
until grep -q DONE /root/s2-status 2>/dev/null; do sleep 20; done
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
M=/root/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq
rm -f /root/s2b-status
for tag in short long; do
  for off in 24 30 36; do
    EXL3_MOE_CPU_OFFLOAD=$off PYTHONPATH=/root/exllamav3-src flock /root/gpu.lock timeout 3600 /root/exl3-ref/bin/python tasks/ornith-exl3/compare/exl3_logits.py $M /root/s2-$tag-mirror.pt /root/s2-exl3-$tag.pt > /root/s2-exl3-$tag.log 2>&1
    rc=$?; echo "exl3-$tag off=$off rc=$rc" >> /root/s2b-status
    [ $rc = 0 ] && break
  done
done
echo DONE >> /root/s2b-status
