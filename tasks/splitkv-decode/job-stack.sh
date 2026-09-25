#!/bin/bash
# CUDA stack A/B, report only. Usage: job-stack.sh venv|stack whole|saver
#   venv  = /root/venv (torch 2.11.0+cu130, triton 3.6.0, flashinfer 0.6.17, sglang-kernel 0.4.5), tree /root/FT-splitkv-base
#   stack = /root/venv-stack (torch 2.13.0+cu130, triton 3.7.1, flashinfer 0.6.18.post1, sglang-kernel 0.4.7),
#           tree /root/FT-stack (same commit, extensions rebuilt against torch 2.13)
ARM=$1; MODE=${2:-whole}
if [ "$ARM" = stack ]; then R=/root/FT-stack; VENV=/root/venv-stack; else R=/root/FT-splitkv-base; VENV=/root/venv; fi
. "$(dirname "$0")/box-env.sh"; PY=$VENV/bin/python
exec 9>/root/gpu.lock; flock 9
echo "=== stack $ARM $MODE $(date -u +%FT%TZ) code $(git -C $R log --oneline -1) torch $($PY -c 'import torch;print(torch.__version__)' 2>&1)"; gpu_idle || exit 1
O=$K/results/stack-$ARM-$MODE; mkdir -p $O; E=$O/arm.env
{ echo "export FREETOKEN_MODEL=$MODEL"; echo "export FREETOKEN_MEMORY_RATIO=1.00"
  if [ "$MODE" = saver ]; then echo "export FREETOKEN_MIRROR_EXPERT_RAM=1"; echo "export FREETOKEN_MIRROR_HOST_ROWS=-1"
  else echo "export FREETOKEN_MIRROR_EXPERT_RAM=0"; echo "export FREETOKEN_MIRROR_HOST_ROWS=0"; fi
  echo "export UV_PROJECT_ENVIRONMENT=$VENV"; echo "export UV_NO_SYNC=1"; echo "export PYTHONPATH=$R/python"; } > $E
LOG=$O/server.log; : > $LOG
free -g > $O/free-before.txt
FREETOKEN_HOST_ENV=$E FREETOKEN_PORT=1920 setsid nohup $R/scripts/serve-default.sh >> $LOG 2>&1 < /dev/null &
wait_ready $LOG 900 || { stop_server; exit 1; }
export FREETOKEN_URL=http://127.0.0.1:1920
PROBE_GEN_TOKENS=32 $PY $R/scripts/probe_decode.py 8000 > $O/warm.jsonl 2>&1
PROBE_GEN_TOKENS=128 PROBE_PASSES=2 timeout 2400 $PY $R/scripts/probe_decode.py 8000 80000 | tee $O/probe.jsonl
{ echo "captures=$(grep -c 'Start capturing CUDA graphs' $LOG) tracebacks=$(grep -c Traceback $LOG) settled_pageable=$(grep -c 'settled pageable' $LOG)"
  grep -E "Triton decode launch" $LOG | sed 's/\x1b\[[0-9;]*m//g' | tail -1; } | tee $O/summary.txt
stop_server
echo "=== stack $ARM $MODE done $(date -u +%FT%TZ)"
