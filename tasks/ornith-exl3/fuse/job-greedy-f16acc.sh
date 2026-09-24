#!/bin/bash
# P2b gate by output: greedy generation on Ornith EXL3 5.0bpw + RAM saver (the job-e2e.sh server, ratio 1.00),
# a haystack prompt per SIZE (83000 = 114K Ornith tokens; SIZE may list several), GREEDY_TOKENS (1024) tokens,
# temperature 0, GREEDY_REPEATS (2) repeats per arm; TAG names the result dir:
#   ref  = FREETOKEN_EXL3_F16ACC=0 (fp32 accumulation)
#   f16  = FREETOKEN_EXL3_F16ACC=1
#   tile = F16ACC=0 with the triton extend kernel on another valid tile (M64 N64 w8 s1): the noise yardstick
# Usage: WT=<tree> [SIZE=..] [TAG=..] job-greedy-f16acc.sh ref|f16|tile   (holds /root/gpu.lock for the arm)
ARM=${1:?ref|f16|tile}; W=${WT:?tree}; O=/root/K/results/exl3greedy-f16acc${TAG:+-$TAG}; mkdir -p $O; SIZE=${SIZE:-83000}
cd $W; export PYTHONPATH=$W/python
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
export FREETOKEN_EXPERT_ARENA=1 FREETOKEN_ARENA_STEP_SLOTS=8 FREETOKEN_GROWABLE_OVERLAP=1 FREETOKEN_SCHEDULER_INVARIANT=warn
export FREETOKEN_EXL3_GEMV_PREROT=1
if [ $ARM = f16 ]; then export FREETOKEN_EXL3_F16ACC=1; else export FREETOKEN_EXL3_F16ACC=0; fi
if [ $ARM = tile ]; then export FREETOKEN_EXTEND_BACKEND=triton FREETOKEN_EXTEND_BLOCK_M=64 FREETOKEN_EXTEND_BLOCK_N=64 FREETOKEN_EXTEND_NUM_WARPS=8 FREETOKEN_EXTEND_NUM_STAGES=1; fi
M=/root/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq; PORT=30100; LOG=$O/$ARM-server.log
C=/root/K/ft-cache; mkdir -p $C/spill
exec 9>/root/gpu.lock; flock 9
echo "=== exl3 greedy-f16acc $ARM size=$SIZE $(date -u +%FT%TZ) code $(git -C $W log --oneline -1) dirty=$(git -C $W status --porcelain python | wc -l)"
while pgrep -f "ft serve" >/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 16 ]; do sleep 5; done
git -C $W diff > $O/tree.diff
setsid nohup /root/venv/bin/ft serve --model $M --text-model-only --host 127.0.0.1 --port $PORT \
  --max-running-requests 1 --kv-grow-step-tokens 65536 \
  --num-tokens 262144 --max-seq-len-override 262144 --kv-cache-dtype q8_0 \
  --attention-backend triton --moe-backend offload --moe-cache-auto --moe-cache-policy lfu \
  --expert-residency mirror --moe-mirror-host-rows -1 --memory-ratio 1.00 --max-prefill-length 8192 \
  --session-spill-ram-gb 1 --session-spill-disk-gb 50 --session-spill-limit-gb 50 --session-spill-dir $C/spill \
  --served-model-name ornith --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --force-nonempty-content --max-output-tokens 65536 > $LOG 2>&1 < /dev/null &
SPID=$!
for i in $(seq 1 360); do
  kill -0 $SPID 2>/dev/null || { echo "SERVER DIED"; exit 1; }
  grep -q "API server is ready" $LOG && break; sleep 5; done
grep -q "API server is ready" $LOG || { echo "NOT READY"; kill $SPID; exit 1; }
echo "ready after $((i*5))s"
rm -f $O/$ARM.jsonl
FREETOKEN_URL=http://127.0.0.1:$PORT FREETOKEN_MODEL_NAME=ornith GREEDY_TOKENS=${GREEDY_TOKENS:-1024} GREEDY_REPEATS=${GREEDY_REPEATS:-2} \
  timeout 1800 /root/venv/bin/python $W/tasks/attn-prefill/box/greedy_client.py $O/$ARM.jsonl $SIZE
echo "rc=$? captures=$(grep -c 'Start capturing CUDA graphs' $LOG) tracebacks=$(grep -c Traceback $LOG)"
kill $SPID; wait $SPID
echo "=== exl3 greedy-f16acc $ARM done $(date -u +%FT%TZ)"
