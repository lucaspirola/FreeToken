#!/bin/bash
# round3 worker: exp/reorg-round3 full 7-directory suite, fresh JIT, under /root/gpu.lock, on an idle GPU.
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
O=/root/r3suite; S=$O/status; T=/root/FT-round3
exec 9>/root/gpu.lock; flock 9
while pgrep -f "[f]t serve" >/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 64 ]; do sleep 15; done
echo "start $(date +%T) gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader) $(uptime)" >> $S
cd $T || exit 1
echo "tree $(git log --oneline -1 | cut -c1-120)" >> $S
export PYTHONPATH=$T/python
rm -rf $O/ext
TORCH_EXTENSIONS_DIR=$O/ext /root/venv/bin/python -m pytest -q -p no:cacheprovider tests/moe tests/engine tests/scheduler tests/kernels tests/kvcache tests/server tests/tokenizer > $O/suite.txt 2>&1
rc=$?
echo "suite rc=$rc $(tail -1 $O/suite.txt) | illegal: $(grep -c 'illegal memory access' $O/suite.txt) | failed-lines: $(grep -c '^FAILED' $O/suite.txt)" >> $S
echo ALLDONE >> $S
