#!/bin/bash
# Nsight Systems timeline of Ornith EXL3 decode at ctx 8000 (saver, ratio 1.00), graphs traced as single
# nodes (--cuda-graph-trace=graph), no torch.profiler: how much of a decode step is graph replay, eager
# host-driven glue and GPU idle. NSYS_EXTRA adds nsys flags (e.g. --python-backtrace=cuda).
# Usage: [RESIDENCY=whole] [NSYS_EXTRA=...] job-nsys.sh TAG   (flocks /root/gpu.lock; box)
TAG=${1:?tag}; R=$(cd "$(dirname "$0")/../../.." && pwd); O=/root/K/results/exl3nsys-$TAG; mkdir -p $O
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
export FREETOKEN_EXL3_GEMV_PREROT=${PREROT:-1} NSYS=1
export FREETOKEN_EXPERT_ARENA=1 FREETOKEN_ARENA_STEP_SLOTS=8 FREETOKEN_GROWABLE_OVERLAP=1 PYTHONPATH=$R/python
M=/root/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq
RES="--expert-residency mirror --moe-mirror-host-rows -1"; [ "${RESIDENCY:-mirror}" = whole ] && RES="--expert-residency whole"
exec 9>/root/gpu.lock; flock 9
echo "=== exl3 nsys $TAG residency=${RESIDENCY:-mirror} $(date -u +%FT%TZ) code $(git -C $R log --oneline -1) dirty=$(git -C $R status --porcelain python | wc -l)"
while pgrep -f "ft serve" >/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 16 ]; do sleep 5; done
cd $R
timeout 2400 nsys profile -t cuda ${NSYS_EXTRA:-} --cuda-graph-trace=graph --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o $O/decode-8k --force-overwrite true \
  /root/venv/bin/python tasks/ornith-exl3/perf/profile_step.py decode $O/decode-8k 8000 128 -- \
  --model $M --text-model-only --max-running-requests 1 --kv-grow-step-tokens 65536 \
  --num-tokens 262144 --max-seq-len-override 262144 --kv-cache-dtype q8_0 \
  --attention-backend triton --moe-strategy offload --moe-cache-auto --moe-cache-policy lfu \
  $RES --memory-ratio 1.00 --max-prefill-length 8192 > $O/decode-8k.log 2>&1
echo "rc=$?"; grep "nsys window" $O/decode-8k.log
nsys export --type sqlite --force-overwrite true -o $O/decode-8k.sqlite $O/decode-8k.nsys-rep > /dev/null 2>&1
/root/venv/bin/python $R/tasks/ornith-exl3/perf/nsys_split.py $O/decode-8k.sqlite 127 | tee $O/split.txt
echo "=== exl3 nsys $TAG done $(date -u +%FT%TZ)"
