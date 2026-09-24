#!/bin/bash
# ft-g5: exp/mirror-dma-wb d1ed596 (no copy-engine nodes in the decode graph, snapshots on a
# side stream). 1) GPU tests, 2) nsys 8K whole + mirror (graph level) + mirror (node level),
# 3) checkpoint arms whole mirror-1m mirror whole-1m. Box numbers.
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
R=/root/FreeToken-gap; cd $R
echo "=== g5-gap start $(date -u +%FT%TZ) $(git log --oneline -1)"
( exec 9>/root/gpu.lock; flock 9
  echo "=== tests $(date -u +%FT%TZ) GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  PYTHONPATH=$R/python /root/venv/bin/python -m pytest -q -p no:cacheprovider tests/moe tests/scheduler \
     > /root/g5-gap-tests.txt 2>&1; tail -3 /root/g5-gap-tests.txt )
NSYS=$(ls /opt/nvidia/nsight-systems/*/bin/nsys | head -1)
O=$R/tasks/exclusive-expert-ram/results/gap-nsys; mkdir -p $O
( exec 9>/root/gpu.lock; flock 9
for spec in whole:graph mirror:graph mirror:node; do
  a=${spec%%:*}; lvl=${spec##*:}; tag=$a; [ $lvl = node ] && tag=$a-node
  E=$O/$a.env
  { echo "export FREETOKEN_MODEL=/root/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
    echo "export FREETOKEN_MEMORY_RATIO=1.00"
    if [ $a = whole ]; then echo "export FREETOKEN_MIRROR_EXPERT_RAM=0"; echo "export FREETOKEN_MIRROR_HOST_ROWS=0"
    else echo "export FREETOKEN_MIRROR_EXPERT_RAM=1"; echo "export FREETOKEN_MIRROR_HOST_ROWS=-1"; echo "export FREETOKEN_MIRROR_RESERVE_ROWS=256"; fi
    echo "export UV_PROJECT_ENVIRONMENT=/root/venv"; echo "export UV_NO_SYNC=1"; echo "export PYTHONPATH=$R/python"; } > $E
  echo "=== nsys $tag $(date -u +%FT%TZ) GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  LOG=$O/$tag-server.log; : > $LOG
  FREETOKEN_HOST_ENV=$E FREETOKEN_PORT=1920 setsid nohup "$NSYS" launch --session-new=ft$a --trace=cuda,nvtx \
     --cuda-graph-trace=$lvl $R/scripts/serve-default.sh >> $LOG 2>&1 < /dev/null &
  t0=$(date +%s)
  until grep -q "API server is ready" $LOG || [ $(( $(date +%s) - t0 )) -gt 900 ]; do sleep 5; done
  echo "ready after $(( $(date +%s) - t0 ))s"
  export FREETOKEN_URL=http://127.0.0.1:1920
  for i in 1 2 3; do PROBE_GEN_TOKENS=64 /root/venv/bin/python $R/scripts/probe_decode.py 8000 > /dev/null 2>&1; sleep 10; done
  "$NSYS" start --session=ft$a --output=$O/$tag --force-overwrite=true
  PROBE_GEN_TOKENS=128 PROBE_PASSES=2 /root/venv/bin/python $R/scripts/probe_decode.py 8000 | tee $O/$tag-probe.jsonl
  "$NSYS" stop --session=ft$a
  "$NSYS" shutdown --session=ft$a --kill=sigterm 2>&1 | tail -2
  sleep 10; pkill -f "ft serve"; sleep 10
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
done
cd $O
for f in whole mirror mirror-node; do "$NSYS" export --type=sqlite --force-overwrite=true -o $f.sqlite $f.nsys-rep > /dev/null 2>&1; done
python3 /root/nsys_steps.py whole.sqlite mirror.sqlite > steps-graph-level.txt 2>&1
python3 /root/nsys_graph_nodes.py mirror-node.sqlite > in-graph-node-level.txt 2>&1
rm -f *.sqlite
echo "=== nsys done $(date -u +%FT%TZ)"; cat steps-graph-level.txt | grep -- "-- pass"
cat in-graph-node-level.txt | head -30 )
cd $R && ARMS="whole mirror-1m mirror whole-1m" tasks/exclusive-expert-ram/checkpoint-box.sh gap1
echo G5-GAP-DONE $(date -u +%FT%TZ)
