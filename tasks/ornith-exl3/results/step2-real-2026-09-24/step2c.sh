#!/bin/bash
cd /root/FreeToken
until grep -q DONE /root/s3-status 2>/dev/null; do sleep 20; done
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
M=/root/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq
for tag in short long; do
EXL3_MOE_CPU_OFFLOAD=36 PYTHONPATH=/root/exllamav3-src flock /root/gpu.lock timeout 3600 /root/exl3-ref/bin/python tasks/ornith-exl3/compare/exl3_logits.py $M /root/s2-$tag-mirror.pt /root/s2-exl3off36-$tag.pt > /root/s2-exl3off36-$tag.log 2>&1
echo "exl3off36-$tag rc=$?" >> /root/s2c-status
done
echo DONE >> /root/s2c-status
