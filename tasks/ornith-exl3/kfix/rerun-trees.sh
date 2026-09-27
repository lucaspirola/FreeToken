#!/usr/bin/env bash
# A script from this dir on the base tree then this tree, under the host GPU lock. Usage: rerun-trees.sh OUT SCRIPT
export PATH=/usr/lib/wsl/lib:/usr/local/cuda/bin:$PATH
F=$(dirname "$(readlink -f "$0")"); WT=$(cd "$F/../../.." && pwd); BASE=/home/lucas/ai/FreeToken-wt/kbase; O=$F/results/$1
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock; flock 9
echo "# $(date -Is) sm $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader) load $(cut -d' ' -f1-3 /proc/loadavg)" > $O
cd $F; for t in base:$BASE new:$WT; do echo "=== ${t%%:*} $(git -C ${t#*:} log --oneline -1 | cut -c1-8) +$(git -C ${t#*:} diff --stat | tail -1)" >> $O
  PYTHONPATH=${t#*:}/python:$F TVM_FFI_CUDA_ARCH_LIST=12.0 /home/lucas/ai/FreeToken/.venv/bin/python $F/$2 >> $O 2>&1; done; echo SWEEPDONE >> $O
