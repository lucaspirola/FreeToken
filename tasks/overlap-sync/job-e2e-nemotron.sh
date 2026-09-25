#!/bin/bash
# Nemotron 3.5 Lightning NVFP4 E2E via the tree's scripts/serve-default.sh (the production profile), ratio 1.00,
# port 1920, nothing else on the GPU; warm-up, then 8K/32K/80K x2 probe passes (pass 2 of record).
# Usage: WT=<tree> RESIDENCY=whole|mirror job-e2e-nemotron.sh TAG     (flocks /root/gpu.lock; box)
TAG=${1:?tag}; W=${WT:?tree}; O=/root/K/results/nem-e2e-$TAG; mkdir -p $O; PY=/root/venv/bin/python
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
RES="--expert-residency whole"; [ "${RESIDENCY:-whole}" = mirror ] && RES="--expert-residency mirror --moe-mirror-host-rows -1"
E=$O/host.env
{ echo "export FREETOKEN_MODEL=/root/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
  echo "export FREETOKEN_MEMORY_RATIO=1.00"; echo "export FREETOKEN_MIRROR_EXPERT_RAM=0"; echo "export FREETOKEN_MIRROR_HOST_ROWS=0"
  echo "export UV_PROJECT_ENVIRONMENT=/root/venv"; echo "export UV_NO_SYNC=1"; echo "export PYTHONPATH=$W/python"
  echo "export FREETOKEN_CACHE_DIR=/root/K/ft-cache-nem"; } > $E
exec 9>/root/gpu.lock; flock 9
while pgrep -f "ft serve" >/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 16 ]; do sleep 5; done
echo "=== nem e2e $TAG residency=${RESIDENCY:-whole} $(date -u +%FT%TZ) tree $W $(cat $W/SOURCE 2>/dev/null)"
LOG=$O/server.log; : > $LOG
FREETOKEN_HOST_ENV=$E FREETOKEN_PORT=1920 FREETOKEN_EXTRA_ARGS="$RES" setsid nohup $W/scripts/serve-default.sh >> $LOG 2>&1 < /dev/null &
SPID=$!
t0=$(date +%s); until grep -q "API server is ready" $LOG; do
  kill -0 $SPID 2>/dev/null || { echo "SERVER DIED"; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 900 ] && { echo "NOT READY"; kill -- -$SPID; exit 1; }; sleep 5; done
echo "ready after $(( $(date +%s) - t0 ))s"
export FREETOKEN_URL=http://127.0.0.1:1920 FREETOKEN_MODEL_NAME=nemotron-3.5-lightning
PROBE_GEN_TOKENS=32 timeout 600 $PY $W/scripts/probe_decode.py 8000 > $O/warm.jsonl 2>&1
PROBE_GEN_TOKENS=128 PROBE_PASSES=2 timeout 2400 $PY $W/scripts/probe_decode.py ${SIZES:-8000 32000 80000} | tee $O/probe.jsonl
curl -s http://127.0.0.1:1920/v1/stats > $O/stats.json
echo "captures=$(grep -c 'Start capturing CUDA graphs' $LOG) tracebacks=$(grep -c Traceback $LOG) oom=$(grep -ci 'out of memory' $LOG) loop=$(grep -o 'Scheduler loop: [a-z]*' $LOG | head -1)" | tee $O/summary.txt
kill -- -$SPID; t0=$(date +%s); while kill -0 $SPID 2>/dev/null || pgrep -f "port 1920" >/dev/null; do [ $(( $(date +%s) - t0 )) -gt 60 ] && kill -9 -- -$SPID; sleep 2; done
echo "=== nem e2e $TAG done $(date -u +%FT%TZ)"
