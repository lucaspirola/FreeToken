#!/usr/bin/env bash
export PATH=/usr/lib/wsl/lib:/usr/local/cuda/bin:$PATH
F=$(dirname "$(readlink -f "$0")"); WT=$(cd "$F/../../.." && pwd); BASE=/home/lucas/ai/FreeToken-wt/kbase; O=$F/results/$1; shift
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock; flock 9
for t in base:$BASE new:$WT; do for M in "$@"; do echo "=== ${t%%:*}" >> $O
  PYTHONPATH=${t#*:}/python TVM_FFI_CUDA_ARCH_LIST=12.0 /home/lucas/ai/FreeToken/.venv/bin/python $F/prof_prefill.py $M >> $O 2>&1; done; done; echo PROFDONE >> $O
