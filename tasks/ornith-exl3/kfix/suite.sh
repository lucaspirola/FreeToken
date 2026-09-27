#!/usr/bin/env bash
# 8-dir suite, model unloaded, under the host GPU lock (as _orch/r5/local-suite/run.sh). Usage: suite.sh OUTDIR
export PATH=/usr/lib/wsl/lib:/usr/local/cuda/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
F=$(dirname "$(readlink -f "$0")"); WT=$(cd "$F/../../.." && pwd); O=$F/$1; mkdir -p $O
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock; flock 9
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -le 32 ]; do sleep 5; done
echo "suite start $(date -Is) gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader) $(uptime)" >> $O/status
echo "tree $(git -C $WT log --oneline -1 | cut -c1-120); so: $(md5sum $WT/python/freetoken/kernel/*.so | cut -c1-10 | paste -sd,)" >> $O/status
rm -rf $O/ext
( cd $WT && PYTHONPATH=$WT/python TVM_FFI_CUDA_ARCH_LIST=12.0 TORCH_EXTENSIONS_DIR=$O/ext /home/lucas/ai/FreeToken/.venv/bin/python -m pytest -q -p no:cacheprovider tests/moe tests/engine tests/scheduler tests/kernels tests/kvcache tests/server tests/tokenizer tests/layers > $O/suite.txt 2>&1 )
rc=$?
echo "suite rc=$rc $(tail -1 $O/suite.txt) | illegal: $(grep -c 'illegal memory access' $O/suite.txt) | failed-lines: $(grep -c '^FAILED' $O/suite.txt)" >> $O/status
rm -rf $O/ext; echo SUITEDONE >> $O/status
