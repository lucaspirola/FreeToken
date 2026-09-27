#!/bin/bash
# pytest on a box tree under the GPU lock. Usage: WT=<tree> job-tests.sh TAG <pytest paths/args...>
TAG=${1:?tag}; shift; W=${WT:?tree}; O=/root/K/results/tests-$TAG; mkdir -p $O
cd $W; export PYTHONPATH=$W/python CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
exec 9>/root/gpu.lock; flock 9
while pgrep -f "ft serve" >/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 16 ]; do sleep 5; done
echo "=== tests $TAG $(date -u +%FT%TZ) tree $W $(cat $W/SOURCE 2>/dev/null) :: $*"
timeout 2700 /root/venv/bin/python -m pytest -q -p no:cacheprovider --timeout=600 "$@" > $O/pytest.txt 2>&1 || true
tail -3 $O/pytest.txt | sed 's/^/  /'
