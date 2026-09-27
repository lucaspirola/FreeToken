#!/bin/bash
# round5 worker: full 8-directory suite, fresh JIT, under /root/gpu.lock (held once for both
# trees), on an idle GPU: exp/reorg-round5 (/root/FT-round5f -> /root/r5fsuite) and exp/r5-seed-b
# (/root/FT-r5seed -> /root/r5seedsuite). Same command as r4bsuite.sh / r5fsuite.sh.
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
exec 9>/root/gpu.lock; flock 9
while pgrep -f "[f]t serve" >/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 64 ]; do sleep 15; done
for pair in /root/FT-round5f:/root/r5fsuite /root/FT-r5seed:/root/r5seedsuite; do
  T=${pair%%:*}; O=${pair##*:}; S=$O/status; mkdir -p $O
  echo "start $(date +%T) gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader) $(uptime)" >> $S
  cd $T || { echo "no $T" >> $S; continue; }
  echo "tree $(git log --oneline -1 | cut -c1-120)" >> $S
  rm -rf $O/ext
  PYTHONPATH=$T/python TORCH_EXTENSIONS_DIR=$O/ext /root/venv/bin/python -m pytest -q -p no:cacheprovider tests/moe tests/engine tests/scheduler tests/kernels tests/kvcache tests/server tests/tokenizer tests/layers > $O/suite.txt 2>&1
  rc=$?
  echo "suite rc=$rc $(tail -1 $O/suite.txt) | illegal: $(grep -c 'illegal memory access' $O/suite.txt) | failed-lines: $(grep -c '^FAILED' $O/suite.txt)" >> $S
  echo ALLDONE >> $S
done
echo BOTHDONE > /root/r5suites.done
