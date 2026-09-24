#!/bin/bash
# Whole-model Nemotron decode rates, no tracer. Usage: job-probe.sh TAG "8000 80000 256000" [PASSES]
TAG=${1:-base}; SIZES=${2:-"8000 80000 256000"}; PASSES=${3:-2}; . "$(dirname "$0")/box-env.sh"
exec 9>/root/gpu.lock; flock 9
echo "=== probe $TAG sizes=$SIZES passes=$PASSES $(date -u +%FT%TZ) code $(git -C $R log --oneline -1)"; gpu_idle || exit 1
O=$K/results/probe-$TAG; mkdir -p $O; E=$O/whole.env; mk_env $E
LOG=$O/server.log; : > $LOG
FREETOKEN_HOST_ENV=$E FREETOKEN_PORT=1920 setsid nohup $R/scripts/serve-default.sh >> $LOG 2>&1 < /dev/null &
wait_ready $LOG 900 || { stop_server; exit 1; }
export FREETOKEN_URL=http://127.0.0.1:1920
PROBE_GEN_TOKENS=32 $PY $R/scripts/probe_decode.py 8000 > $O/warm.jsonl 2>&1
PROBE_GEN_TOKENS=128 PROBE_PASSES=$PASSES $PY $R/scripts/probe_decode.py $SIZES | tee $O/probe.jsonl
{ echo "captures=$(grep -c 'Start capturing CUDA graphs' $LOG) tracebacks=$(grep -c Traceback $LOG) kv_grows=$(grep -c 'KV grew' $LOG)"
  grep -E "Triton decode launch" $LOG | sed 's/\x1b\[[0-9;]*m//g' | tail -1; } | tee $O/summary.txt
stop_server
echo "=== probe $TAG done $(date -u +%FT%TZ)"
