#!/usr/bin/env bash
export PATH=/usr/lib/wsl/lib:/usr/local/cuda/bin:$PATH
F=$(dirname "$(readlink -f "$0")"); WT=$(cd "$F/../../.." && pwd); O=$F/results/crossover2.txt
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock; flock 9
cd $F; export PYTHONPATH=$WT/python:$F TVM_FFI_CUDA_ARCH_LIST=12.0
echo "# new tree, shipped block_m" > $O; /home/lucas/ai/FreeToken/.venv/bin/python crossover2.py 768,1024,1280,1536,1792,2048 >> $O 2>&1
echo "# new tree, inline block_m capped at 32" >> $O; /home/lucas/ai/FreeToken/.venv/bin/python crossover2.py 1024,1536,2048,3072 bm32 >> $O 2>&1; echo SWEEPDONE >> $O
