#!/bin/bash
# Greedy generation, triton vs flashinfer extend (+ a second triton tile as the noise yardstick), on an
# NVFP4 model served the production way (serve-default.sh, whole model, ratio 1.00, port 1920):
# a haystack prompt per SIZE (the client estimates 16 tokens a line; SIZE 83000 is ~107K Nemotron / ~114K
# Ornith tokens), GREEDY_TOKENS (1024) generated tokens, temperature 0, GREEDY_REPEATS (2) repeats.
# SIZE may list several prompt sizes ("62000 70000 ..."); TAG names the result dir.
# Usage: R=<tree> [SIZE=..] [TAG=..] job-greedy.sh nemo|ornith tri|fi|tile   (flocks /root/gpu.lock per arm)
WHICH=${1:?nemo|ornith}; ARM=${2:?tri|fi|tile}; SIZE=${SIZE:-83000}
R=${R:?tree}; K=/root/K; PY=/root/venv/bin/python; D=$(cd "$(dirname "$0")" && pwd)
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
if [ $WHICH = ornith ]; then MODEL=/root/models/Ornith-1.5-35B-A3B-NVFP4 NAME=ornith
  EXTRA="--num-tokens 262144 --max-seq-len-override 262144 --served-model-name ornith"
else MODEL=/root/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 NAME=nemotron-3.5-lightning EXTRA=""; fi
O=$K/results/attngreedy${TAG:+-$TAG}-$WHICH; mkdir -p $O; E=$O/$ARM.env
{ echo "export FREETOKEN_MODEL=$MODEL"; echo "export FREETOKEN_MEMORY_RATIO=1.00"
  echo "export FREETOKEN_MIRROR_EXPERT_RAM=0"; echo "export FREETOKEN_MIRROR_HOST_ROWS=0"
  echo "export UV_PROJECT_ENVIRONMENT=/root/venv"; echo "export UV_NO_SYNC=1"
  echo "export PYTHONPATH=$R/python"
  if [ $ARM = fi ]; then echo "export FREETOKEN_EXTEND_BACKEND=flashinfer"
  else echo "export FREETOKEN_EXTEND_BACKEND=triton"; fi
  if [ $ARM = tile ]; then echo "export FREETOKEN_EXTEND_BLOCK_M=64 FREETOKEN_EXTEND_BLOCK_N=64 FREETOKEN_EXTEND_NUM_WARPS=8 FREETOKEN_EXTEND_NUM_STAGES=1"; fi
  [ $WHICH = ornith ] && echo "export FREETOKEN_PIN_BUDGET_GB=40"; } > $E
exec 9>/root/gpu.lock; flock 9
while pgrep -f "ft serve" >/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 16 ]; do sleep 5; done
echo "=== attn greedy $WHICH $ARM size=$SIZE $(date -u +%FT%TZ) code $(git -C $R log --oneline -1) dirty=$(git -C $R status --porcelain python | wc -l)"
LOG=$O/$ARM-server.log; : > $LOG
FREETOKEN_HOST_ENV=$E FREETOKEN_PORT=1920 FREETOKEN_EXTRA_ARGS="$EXTRA" setsid nohup $R/scripts/serve-default.sh >> $LOG 2>&1 < /dev/null &
t0=$(date +%s); until grep -q "API server is ready" $LOG; do
  if ! pgrep -f "ft serve" >/dev/null && grep -q Traceback $LOG; then echo "SERVER DIED"; exit 1; fi
  [ $(( $(date +%s) - t0 )) -gt 900 ] && { echo "NOT READY"; pkill -f "ft serve"; exit 1; }; sleep 5; done
echo "ready after $(( $(date +%s) - t0 ))s"
export FREETOKEN_URL=http://127.0.0.1:1920 FREETOKEN_MODEL_NAME=$NAME
rm -f $O/$ARM.jsonl
GREEDY_TOKENS=${GREEDY_TOKENS:-1024} GREEDY_REPEATS=${GREEDY_REPEATS:-2} timeout 2400 $PY $D/greedy_client.py $O/$ARM.jsonl $SIZE
echo "rc=$? captures=$(grep -c 'Start capturing CUDA graphs' $LOG) tracebacks=$(grep -c Traceback $LOG) extend_fi=$(grep -c 'extend attention: flashinfer' $LOG)"
pkill -f "ft serve"; t0=$(date +%s); while pgrep -f "ft serve" >/dev/null; do [ $(( $(date +%s) - t0 )) -gt 60 ] && pkill -9 -f "ft serve"; sleep 2; done
echo "=== attn greedy $WHICH $ARM done $(date -u +%FT%TZ)"
