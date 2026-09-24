#!/bin/bash
# Whole-model Nemotron server under nsys (--cuda-graph-trace=node); trace only the decode of
# one request per size. Usage: job-nsys.sh TAG "8000 80000 256000"   (under /root/gpu.lock)
TAG=${1:-base}; SIZES=${2:-"8000 80000 256000"}; . "$(dirname "$0")/box-env.sh"
exec 9>/root/gpu.lock; flock 9
echo "=== nsys $TAG sizes=$SIZES $(date -u +%FT%TZ) code $(git -C $R log --oneline -1) nsys=$NSYS"; gpu_idle || exit 1
O=$K/results/nsys-$TAG; mkdir -p $O; E=$O/whole.env; mk_env $E
LOG=$O/server.log; : > $LOG
FREETOKEN_HOST_ENV=$E FREETOKEN_PORT=1920 setsid nohup "$NSYS" launch --session-new=ftk$TAG --trace=cuda,nvtx \
   --cuda-graph-trace=node $R/scripts/serve-default.sh >> $LOG 2>&1 < /dev/null &
wait_ready $LOG 900 || { stop_server; "$NSYS" shutdown --session=ftk$TAG --kill=sigterm; exit 1; }
export FREETOKEN_URL=http://127.0.0.1:1920 NSYS NSYS_SESSION=ftk$TAG
PROBE_GEN_TOKENS=32 $PY $R/scripts/probe_decode.py 8000 > $O/warm.jsonl 2>&1
for s in $SIZES; do
  PROBE_GEN_TOKENS=${GEN:-144} SKIP_TOKENS=16 $PY $R/tasks/splitkv-decode/nsys_decode_client.py $s $O/dec-$s | tee -a $O/client.jsonl
done
"$NSYS" shutdown --session=ftk$TAG --kill=sigterm 2>&1 | tail -2
stop_server
for s in $SIZES; do
  f=$(ls $O/dec-$s*.nsys-rep | head -1); [ -n "$f" ] || continue
  "$NSYS" export --type=sqlite --force-overwrite=true -o $O/dec-$s.sqlite "$f" > /dev/null 2>&1
done
$PY $R/tasks/splitkv-decode/nsys_decode_split.py $O/dec-*.sqlite | tee $O/split.txt
grep -c "Start capturing CUDA graphs" $LOG | sed 's/^/captures=/'
echo "=== nsys $TAG done $(date -u +%FT%TZ)"
