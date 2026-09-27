#!/bin/bash
# EXL3 kernel tests on a box tree under the GPU lock. Usage: WT=<tree> job-tests.sh TAG [pytest args]
TAG=${1:?tag}; shift; W=${WT:?tree}; O=/root/K/results/exl3tests-$TAG; mkdir -p $O
cd $W; export PYTHONPATH=$W/python CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
exec 9>/root/gpu.lock; flock 9
while pgrep -f "ft serve" >/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 16 ]; do sleep 5; done
echo "=== tests $TAG $(date -u +%FT%TZ) code $(git -C $W log --oneline -1) dirty=$(git -C $W status --porcelain python | wc -l)"
timeout 2400 /root/venv/bin/python -m pytest -q -p no:cacheprovider tests/kernels/test_exl3.py "$@" > $O/pytest.txt 2>&1; echo "rc=$? $(tail -1 $O/pytest.txt)"
