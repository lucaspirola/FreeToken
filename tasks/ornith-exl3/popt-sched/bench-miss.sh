#!/usr/bin/env bash
# bench_miss.py under the GPU host lock, no model loaded. Output in results/bench-miss.txt.
F=$(dirname "$(readlink -f "$0")"); WT=$(cd "$F/../../.." && pwd)
export PATH=/usr/lib/wsl/lib:/usr/local/cuda/bin:/usr/bin:/bin
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock; flock 9
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -le 32 ]; do sleep 5; done
mkdir -p $F/results
{ echo "== $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg) code $(git -C $WT rev-parse --short HEAD)"
  for n in ${NS:-135 256}; do PYTHONPATH=$WT/python /home/lucas/ai/FreeToken/.venv/bin/python $F/bench_miss.py $n; done; } >> $F/results/bench-miss.txt 2>&1
