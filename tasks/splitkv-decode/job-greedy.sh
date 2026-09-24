#!/bin/bash
# Greedy transcripts for one arm. Usage: [R=tree] job-greedy.sh TAG nemo|ornith "SIZES"
TAG=$1; WHICH=${2:-nemo}; SIZES=${3:-"8000 80000 256000"}
if [ "$WHICH" = ornith ]; then
  export MODEL=/root/models/Ornith-1.5-35B-A3B-NVFP4 NAME=ornith
  EXTRA="--num-tokens 262144 --max-seq-len-override 262144 --served-model-name ornith"
else NAME=nemotron-3.5-lightning; EXTRA=""; fi
. "$(dirname "$0")/box-env.sh"
exec 9>/root/gpu.lock; flock 9
echo "=== greedy $WHICH $TAG sizes=$SIZES $(date -u +%FT%TZ) code $(git -C $R log --oneline -1) dirty=$(git -C $R status --porcelain -uno | wc -l)"; gpu_idle || exit 1
O=$K/results/greedy-$WHICH-$TAG; mkdir -p $O; E=$O/whole.env; mk_env $E
[ "$WHICH" = ornith ] && echo "export FREETOKEN_PIN_BUDGET_GB=40" >> $E
LOG=$O/server.log; : > $LOG
FREETOKEN_HOST_ENV=$E FREETOKEN_PORT=1920 FREETOKEN_EXTRA_ARGS="$EXTRA" setsid nohup $R/scripts/serve-default.sh >> $LOG 2>&1 < /dev/null &
wait_ready $LOG 900 || { stop_server; exit 1; }
: > $O/greedy.jsonl
FREETOKEN_URL=http://127.0.0.1:1920 FREETOKEN_MODEL_NAME=$NAME timeout 2400 $PY $(dirname "$0")/greedy_client.py $O/greedy.jsonl $SIZES
echo "client rc=$?"
{ echo "captures=$(grep -c 'Start capturing CUDA graphs' $LOG) tracebacks=$(grep -c Traceback $LOG)"
  grep -E "Triton decode launch" $LOG | sed 's/\x1b\[[0-9;]*m//g' | tail -1; } | tee $O/summary.txt
stop_server
echo "=== greedy $WHICH $TAG done $(date -u +%FT%TZ)"
