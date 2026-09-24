#!/bin/bash
# ft-dev: exp/dt-dma merge -- tests/moe tests/engine tests/scheduler under /root/gpu.lock.
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
S=/root/dtdma-status; rm -f $S
cd /root/FT-dtdma && git log --oneline -1 >> $S
PYTHONPATH=$PWD/python flock /root/gpu.lock /root/venv/bin/python -m pytest -q -p no:cacheprovider tests/moe tests/engine tests/scheduler > /root/dtdma-tests.txt 2>&1
echo "tests rc=$? $(tail -1 /root/dtdma-tests.txt) $(date -u +%T)" >> $S
echo ALLDONE >> $S
