#!/usr/bin/env bash
# Run pytest args in the kfix tree under the host GPU lock (model unloaded). Usage: pytest-unit.sh TAG args...
export PATH=/usr/lib/wsl/lib:/usr/local/cuda/bin:$PATH
F=$(dirname "$(readlink -f "$0")"); WT=$(cd "$F/../../.." && pwd); TAG=$1; shift
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock; flock 9
cd $WT && { echo "start $(date -Is) $(git log --oneline -1 | cut -c1-50) +diff $(git diff --stat | tail -1)"; nvidia-smi --query-gpu=memory.used --format=csv,noheader;
  PYTHONPATH=$WT/python TVM_FFI_CUDA_ARCH_LIST=12.0 TORCH_EXTENSIONS_DIR=$F/ext \
  /home/lucas/ai/FreeToken/.venv/bin/python -m pytest -q -p no:cacheprovider "$@"; echo "rc=$? PYTESTDONE $(date -Is)"; } > $F/results/$TAG.log 2>&1
