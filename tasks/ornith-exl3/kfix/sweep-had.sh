#!/usr/bin/env bash
export PATH=/usr/lib/wsl/lib:/usr/local/cuda/bin:$PATH
F=$(dirname "$(readlink -f "$0")"); WT=$(cd "$F/../../.." && pwd); O=$F/results/$1
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock; flock 9
echo "# $(date -Is) sm $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader) load $(cut -d' ' -f1-3 /proc/loadavg)" > $O
cd $F && PYTHONPATH=$WT/python:$F TVM_FFI_CUDA_ARCH_LIST=12.0 /home/lucas/ai/FreeToken/.venv/bin/python $F/${2:-sweep_had_rows.py} >> $O 2>&1; echo SWEEPDONE >> $O
