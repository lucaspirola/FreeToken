#!/bin/bash
# Step 3+4: serve full Ornith EXL3 5.0bpw with the RAM saver (auto pool), one chat, then probe.
# Runs under the GPU lock for its whole life; server killed at the end.
cd /root/FreeToken
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
export FREETOKEN_EXPERT_ARENA=1 FREETOKEN_ARENA_STEP_SLOTS=8 FREETOKEN_GROWABLE_OVERLAP=1 FREETOKEN_SCHEDULER_INVARIANT=warn
M=/root/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq
KV=${KV:-q8_0}; RATIO=${RATIO:-1.00}; SLOTS=${SLOTS:-}
PORT=30100
TAG=${TAG:-hr}; LOG=/root/s5-$TAG-r$RATIO.log; ST=/root/s5-status
C=/root/ft-cache-s5; mkdir -p $C/spill
nvidia-smi --query-gpu=memory.used,memory.total --format=csv > /root/s5-$TAG-r$RATIO-nvsmi-before.txt
/root/venv/bin/ft serve --model $M --text-model-only \
  --host 127.0.0.1 --port $PORT \
  --max-running-requests 1 --kv-grow-step-tokens 65536 \
  --num-tokens 262144 --max-seq-len-override 262144 --kv-cache-dtype $KV \
  --attention-backend triton --moe-backend offload --moe-cache-auto --moe-cache-policy lfu \
  --expert-residency mirror --moe-mirror-host-rows -1 \
  --memory-ratio $RATIO ${SLOTS:+--linear-state-slots $SLOTS} --max-prefill-length 8192 \
  --session-spill-ram-gb 1 --session-spill-disk-gb 50 --session-spill-limit-gb 50 --session-spill-dir $C/spill \
  --served-model-name ornith --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --force-nonempty-content --max-output-tokens 65536 --cuda-memory-telemetry ${EXTRA:-} > $LOG 2>&1 &
SPID=$!
(while kill -0 $SPID 2>/dev/null; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)"; sleep 1; done) > /root/s5-$TAG-r$RATIO-vram.tsv &
echo "server pid $SPID kv=$KV ratio=$RATIO slots=${SLOTS:-default}" >> $ST
for i in $(seq 1 360); do
  kill -0 $SPID 2>/dev/null || { echo "SERVER DIED" >> $ST; echo DONE >> $ST; exit 1; }
  grep -q "API server is ready" $LOG && break
  sleep 5
done
grep -q "API server is ready" $LOG || { echo "NOT READY" >> $ST; kill $SPID; echo DONE >> $ST; exit 1; }
echo "ready after $((i*5))s" >> $ST
free -g > /root/s5-$TAG-r$RATIO-free-ready.txt; nvidia-smi > /root/s5-$TAG-r$RATIO-nvsmi-ready.txt
curl -s -m 600 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"ornith","messages":[{"role":"user","content":"Explain in one paragraph why the sky is blue, then give the Rayleigh scattering intensity dependence on wavelength."}],"max_tokens":1500,"temperature":0}' > /root/s5-$TAG-r$RATIO-chat1.json
echo "chat1 rc=$?" >> $ST
curl -s -m 600 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"ornith","messages":[{"role":"user","content":"Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."}],"max_tokens":1500,"temperature":0}' > /root/s5-$TAG-r$RATIO-chat2.json
echo "chat2 rc=$?" >> $ST
curl -s http://127.0.0.1:$PORT/v1/stats > /root/s5-$TAG-r$RATIO-stats-chat.json
FREETOKEN_URL=http://127.0.0.1:$PORT FREETOKEN_MODEL_NAME=ornith PROBE_PASSES=2 timeout 3000 /root/venv/bin/python scripts/probe_decode.py ${SIZES:-8000 32000 80000 128000} > /root/s5-$TAG-r$RATIO-probe.jsonl 2> /root/s5-$TAG-r$RATIO-probe.err
echo "probe rc=$?" >> $ST
curl -s http://127.0.0.1:$PORT/v1/stats > /root/s5-$TAG-r$RATIO-stats-probe.json
free -g > /root/s5-$TAG-r$RATIO-free-after.txt; nvidia-smi > /root/s5-$TAG-r$RATIO-nvsmi-after.txt
kill $SPID; wait $SPID
echo DONE >> $ST
