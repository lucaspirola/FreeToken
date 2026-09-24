#!/bin/bash
# E2E Ornith EXL3 5.0bpw + RAM saver (auto pool, ratio 1.00): two chats, then probe 8K/32K/80K/128K x2.
# Same recipe as /root/s5wt.sh (step 5/6), outputs under /root/K/results/exl3e2e-$TAG.
# Usage: WT=<tree> PREROT=0|1 job-e2e.sh TAG     (holds /root/gpu.lock for its life)
TAG=${1:?tag}; W=${WT:?tree}; O=/root/K/results/exl3e2e-$TAG; mkdir -p $O
cd $W; export PYTHONPATH=$W/python
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
export FREETOKEN_EXPERT_ARENA=1 FREETOKEN_ARENA_STEP_SLOTS=8 FREETOKEN_GROWABLE_OVERLAP=1 FREETOKEN_SCHEDULER_INVARIANT=warn
export FREETOKEN_EXL3_GEMV_PREROT=${PREROT:?0 or 1}
M=/root/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq; PORT=30100; LOG=$O/server.log
C=/root/K/ft-cache; mkdir -p $C/spill
exec 9>/root/gpu.lock; flock 9
echo "=== exl3 e2e $TAG prerot=$PREROT $(date -u +%FT%TZ) code $(git -C $W log --oneline -1) dirty=$(git -C $W status --porcelain python | wc -l)"
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
for n in 1 2; do
  [ $n = 1 ] && Q="Explain in one paragraph why the sky is blue, then give the Rayleigh scattering intensity dependence on wavelength." \
             || Q="Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."
  curl -s -m 600 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"ornith\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":1500,\"temperature\":0}" > $O/chat$n.json
done
FREETOKEN_URL=http://127.0.0.1:$PORT FREETOKEN_MODEL_NAME=ornith PROBE_PASSES=2 timeout 3000 /root/venv/bin/python scripts/probe_decode.py ${SIZES:-8000 32000 80000 128000} | tee $O/probe.jsonl
curl -s http://127.0.0.1:$PORT/v1/stats > $O/stats.json
echo "captures=$(grep -c 'Start capturing CUDA graphs' $LOG) tracebacks=$(grep -c Traceback $LOG)"
kill $SPID; wait $SPID
echo "=== exl3 e2e $TAG done $(date -u +%FT%TZ)"
