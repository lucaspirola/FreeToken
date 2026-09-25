#!/bin/bash
# round2 worker: verify d56a058 (complete y_q pad) on top of fe95df1.
# V1 memcheck no-caching on test_gguf_mma (d31a680+fix); V2.1/V2.2 full E sequence with per-test
# sync on d31a680+fix (fresh JIT each); S round 2 (8cc98cf + d56a058) full 7-dir suite, fresh JIT.
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
O=/root/r2fix2; S=$O/status; P="$O/p/0001-*.patch $O/p/0002-*.patch"
while pgrep -f "[r]2fix/run.sh|[.]/run.sh" >/dev/null; do sleep 10; done
exec 9>/root/gpu.lock; flock 9
echo "start $(date +%T)" >> $S
G="git -c user.name=probe -c user.email=probe@example.com"
cd /root/FT-wbfix || exit 1
git checkout -q -B fixd31c fixd31 && $G am -q $P || { git am --abort; echo "V am failed" >> $S; exit 1; }
echo "V tree $(git log --oneline -3 | tr '\n' ' ' | cut -c1-160)" >> $S
export PYTHONPATH=$PWD/python
rm -rf $O/ext-V1
PYTORCH_NO_CUDA_MEMORY_CACHING=1 TORCH_EXTENSIONS_DIR=$O/ext-V1 compute-sanitizer --tool memcheck --print-limit 100000 --log-file $O/V1-memcheck.log /root/venv/bin/python -m pytest -q -p no:cacheprovider tests/kernels/test_gguf_mma.py tests/kernels/test_gguf_dispatch.py tests/kernels/test_gguf_quant_types.py tests/moe/test_cpu_moe_mixed_gguf.py > $O/V1-tests.txt 2>&1
echo "V1 d31a680+fe95df1+d56a058+f528b2e gguf files memcheck no-caching rc=$? $(tail -1 $O/V1-tests.txt) | $(grep 'ERROR SUMMARY' $O/V1-memcheck.log | tail -1) | invalid: $(grep -cE 'Invalid (__global__|__shared__|__local__)' $O/V1-memcheck.log)" >> $S
export PYTHONPATH=$PWD/python:/root/wbfix
for k in 1 2; do
  rm -rf $O/ext-E$k
  SYNCPLUG_LOG=$O/syncplug-E$k.log TORCH_EXTENSIONS_DIR=$O/ext-E$k /root/venv/bin/python -m pytest -q -p no:cacheprovider -p syncplug tests/moe tests/engine tests/scheduler tests/kernels tests/kvcache tests/server tests/tokenizer > $O/V2-$k.txt 2>&1
  echo "V2.$k d31a680+fix full sequence rc=$? $(tail -1 $O/V2-$k.txt) | $(tail -1 $O/syncplug-E$k.log) | illegal: $(grep -c 'illegal memory access' $O/V2-$k.txt)" >> $S
done
cd /root/FT-round2 || exit 1
git checkout -q -B r2fix2 8cc98cf && $G am -q $P || { git am --abort; echo "S am failed" >> $S; exit 1; }
echo "S tree $(git log --oneline -3 | tr '\n' ' ' | cut -c1-160)" >> $S
export PYTHONPATH=$PWD/python
rm -rf $O/ext-S
TORCH_EXTENSIONS_DIR=$O/ext-S /root/venv/bin/python -m pytest -q -p no:cacheprovider tests/moe tests/engine tests/scheduler tests/kernels tests/kvcache tests/server tests/tokenizer > $O/suite.txt 2>&1
echo "S round2+d56a058+f528b2e suite rc=$? $(tail -1 $O/suite.txt) | illegal: $(grep -c 'illegal memory access' $O/suite.txt)" >> $S
echo ALLDONE >> $S
