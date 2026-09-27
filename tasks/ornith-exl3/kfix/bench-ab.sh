#!/usr/bin/env bash
# bench_prefill.py base/new alternating (ROUNDS pairs) under the host GPU lock. Usage: bench-ab.sh OUT ROUNDS
export PATH=/usr/lib/wsl/lib:/usr/local/cuda/bin:$PATH
F=$(dirname "$(readlink -f "$0")"); WT=$(cd "$F/../../.." && pwd); BASE=/home/lucas/ai/FreeToken-wt/kbase; O=$F/results/$1
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock; flock 9
echo "# $(date -Is) new=$(git -C $WT log --oneline -1 | cut -c1-8)+$(git -C $WT diff --stat | tail -1 | tr -d ' ') load $(cut -d' ' -f1-3 /proc/loadavg) sm $(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv,noheader)" >> $O
for r in $(seq 1 ${2:-3}); do for t in base:$BASE new:$WT; do
  PYTHONPATH=${t#*:}/python TVM_FFI_CUDA_ARCH_LIST=12.0 /home/lucas/ai/FreeToken/.venv/bin/python $F/bench_prefill.py ${t%%:*} >> $O 2>>$O.err
done; done; echo BENCHDONE >> $O
