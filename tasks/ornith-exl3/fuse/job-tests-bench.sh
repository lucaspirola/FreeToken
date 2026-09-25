#!/bin/bash
# EXL3 kernel tests + layer bench (CUDA-graph replay) on this tree. Usage: job-tests-bench.sh TAG
TAG=${1:-v1}; R=$(cd "$(dirname "$0")/../../.." && pwd); O=/root/K/results/exl3tb-$TAG; mkdir -p $O
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0 PYTHONPATH=$R/python
exec 9>/root/gpu.lock; flock 9
echo "=== exl3 tests+bench $TAG $(date -u +%FT%TZ) code $(git -C $R log --oneline -1) dirty=$(git -C $R status --porcelain python | wc -l)"
while pgrep -f "ft serve" >/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 16 ]; do sleep 5; done
cd $R
timeout 1800 /root/venv/bin/python -m pytest -q -p no:cacheprovider tests/kernels/test_exl3.py tests/layers/test_exl3_linear_layer.py tests/moe/test_exl3_banks.py > $O/tests.txt 2>&1
echo "tests rc=$? $(tail -1 $O/tests.txt)"
timeout 1200 /root/venv/bin/python tasks/ornith-exl3/perf/bench_decode_epilogue.py > $O/bench.txt 2>&1
echo "bench rc=$?"; cat $O/bench.txt | tail -12
echo "=== exl3 tests+bench $TAG done $(date -u +%FT%TZ)"
