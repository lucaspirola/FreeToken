#!/bin/bash
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
cd /root/FT-pin; O=/root/pinned-err-ftdev; mkdir -p $O
{ git log --oneline -1; git status --short; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; date -u; } > $O/env.txt
PYTHONPATH=$PWD/python flock /root/gpu.lock /root/venv/bin/python -m pytest -q -p no:cacheprovider tests/moe tests/engine tests/scheduler tests/kernels > $O/tests.txt 2>&1
echo "rc=$? $(tail -1 $O/tests.txt)" > $O/summary.txt
