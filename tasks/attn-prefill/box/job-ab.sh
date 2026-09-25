#!/bin/bash
# triton vs flashinfer extend A/B on an NVFP4 model, whole model via scripts/serve-default.sh, ratio 1.00,
# port 1920, two probe passes (pass 2 of record). Usage: R=<tree> job-ab.sh nemo|ornith tri|fi "SIZES"
WHICH=${1:?nemo|ornith}; ARM=${2:?tri|fi}; SIZES=${3:-"32000 80000 256000"}
R=${R:?tree}; K=/root/K; PY=/root/venv/bin/python
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
if [ $WHICH = ornith ]; then MODEL=/root/models/Ornith-1.5-35B-A3B-NVFP4 NAME=ornith
  EXTRA="--num-tokens 262144 --max-seq-len-override 262144 --served-model-name ornith"
else MODEL=/root/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 NAME=nemotron-3.5-lightning EXTRA=""; fi
[ $ARM = tri ] && BK=triton || BK=flashinfer
O=$K/results/attnab-$WHICH-$ARM; mkdir -p $O; E=$O/whole.env
{ echo "export FREETOKEN_MODEL=$MODEL"; echo "export FREETOKEN_MEMORY_RATIO=1.00"
  echo "export FREETOKEN_MIRROR_EXPERT_RAM=0"; echo "export FREETOKEN_MIRROR_HOST_ROWS=0"
  echo "export UV_PROJECT_ENVIRONMENT=/root/venv"; echo "export UV_NO_SYNC=1"
  echo "export PYTHONPATH=$R/python"; echo "export FREETOKEN_EXTEND_BACKEND=$BK"
  [ $WHICH = ornith ] && echo "export FREETOKEN_PIN_BUDGET_GB=40"; } > $E
exec 9>/root/gpu.lock; flock 9
while pgrep -f "ft serve" >/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 16 ]; do sleep 5; done
echo "=== attn ab $WHICH $ARM ($BK) sizes=$SIZES $(date -u +%FT%TZ) code $(git -C $R log --oneline -1) dirty=$(git -C $R status --porcelain python | wc -l)"
LOG=$O/server.log; : > $LOG
FREETOKEN_HOST_ENV=$E FREETOKEN_PORT=1920 FREETOKEN_EXTRA_ARGS="$EXTRA" setsid nohup $R/scripts/serve-default.sh >> $LOG 2>&1 < /dev/null &
t0=$(date +%s); until grep -q "API server is ready" $LOG; do
  if ! pgrep -f "ft serve" >/dev/null && grep -q Traceback $LOG; then echo "SERVER DIED"; exit 1; fi
  [ $(( $(date +%s) - t0 )) -gt 900 ] && { echo "NOT READY"; pkill -f "ft serve"; exit 1; }; sleep 5; done
echo "ready after $(( $(date +%s) - t0 ))s"
export FREETOKEN_URL=http://127.0.0.1:1920 FREETOKEN_MODEL_NAME=$NAME
PROBE_GEN_TOKENS=32 timeout 600 $PY $R/scripts/probe_decode.py 8000 > $O/warm.jsonl 2>&1
PROBE_GEN_TOKENS=128 PROBE_PASSES=2 timeout 2400 $PY $R/scripts/probe_decode.py $SIZES | tee $O/probe.jsonl
echo "captures=$(grep -c 'Start capturing CUDA graphs' $LOG) tracebacks=$(grep -c Traceback $LOG) oom=$(grep -ci 'out of memory' $LOG) extend=$(grep -c 'extend attention: flashinfer' $LOG)" | tee $O/summary.txt
pkill -f "ft serve"; t0=$(date +%s); while pgrep -f "ft serve" >/dev/null; do [ $(( $(date +%s) - t0 )) -gt 60 ] && pkill -9 -f "ft serve"; sleep 2; done
echo "=== attn ab $WHICH $ARM done $(date -u +%FT%TZ)"
