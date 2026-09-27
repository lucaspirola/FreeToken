#!/bin/bash
# Run one bench script under the GPU lock on a box tree. Usage: WT=<tree> job-bench.sh TAG script.py [args...]
TAG=${1:?tag}; S=${2:?script}; shift 2; W=${WT:?tree}; O=/root/K/results/exl3bench-$TAG; mkdir -p $O
cd $W; export PYTHONPATH=$W/python CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
exec 9>/root/gpu.lock; flock 9
while pgrep -f "ft serve" >/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 16 ]; do sleep 5; done
echo "=== bench $TAG $S $* $(date -u +%FT%TZ) code $(git -C $W log --oneline -1) dirty=$(git -C $W status --porcelain python | wc -l)"
git -C $W diff > $O/tree.diff
timeout 2400 /root/venv/bin/python $S "$@" > $O/$(basename $S .py).txt 2>&1; echo "rc=$?"
echo "=== bench $TAG done $(date -u +%FT%TZ)"
