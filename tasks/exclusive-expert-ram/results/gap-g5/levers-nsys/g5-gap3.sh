#!/bin/bash
# ft-g5, after the dyn-g5 check. exp/mirror-dma-wb levers, box numbers:
#   L1 d1ed596 (gap1: no copy nodes in the decode graph, snapshots on a side stream)
#   L2 40c154d (+ writeback issue after the launch)          tree /root/FreeToken-l2
#   L3 2392b93 (+ mirror bookkeeping inside the v2 LRU kernel) tree /root/FreeToken-gap2
#   L4 e70849e (+ duplicate-aware eviction, FREETOKEN_MIRROR_DUP_BAND) tree /root/FreeToken-gap4
# 0) graph-launch microbench 1) GPU tests on L4 2) nsys 8K per lever 3) final arms on L4
# 4) mirror-np per lever 5) band 0/1 with --moe-collect-stats (hit rate)
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
until grep -q DYN-G5-DONE /root/dyn-g5.log 2>/dev/null; do sleep 20; done
F=/root/FreeToken-gap4
echo "=== g5-gap3 start $(date -u +%FT%TZ)"
( exec 9>/root/gpu.lock; flock 9; /root/venv/bin/python /root/bench_graph_launch.py > /root/graph-launch-bench.txt 2>&1; cat /root/graph-launch-bench.txt )
( exec 9>/root/gpu.lock; flock 9; cd $F
  echo "=== tests $(git log --oneline -1) $(date -u +%FT%TZ)"
  PYTHONPATH=$F/python /root/venv/bin/python -m pytest -q -p no:cacheprovider -rP tests/moe tests/scheduler > /root/g5-gap3-tests.txt 2>&1
  tail -1 /root/g5-gap3-tests.txt; grep "free eviction rate" /root/g5-gap3-tests.txt )
NSYS=$(ls /opt/nvidia/nsight-systems/*/bin/nsys | head -1)
O=$F/tasks/exclusive-expert-ram/results/levers-nsys; mkdir -p $O
nsys_one() {  # tag tree band
  local tag=$1 R=$2 band=$3 E=$O/$1.env LOG=$O/$1-server.log
  { echo "export FREETOKEN_MODEL=/root/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
    echo "export FREETOKEN_MEMORY_RATIO=1.00"
    echo "export FREETOKEN_MIRROR_EXPERT_RAM=1"; echo "export FREETOKEN_MIRROR_HOST_ROWS=-1"; echo "export FREETOKEN_MIRROR_RESERVE_ROWS=256"
    echo "export FREETOKEN_MIRROR_DUP_BAND=$band"
    echo "export UV_PROJECT_ENVIRONMENT=/root/venv"; echo "export UV_NO_SYNC=1"; echo "export PYTHONPATH=$R/python"; } > $E
  echo "=== nsys $tag $(git -C $R log --oneline -1 | cut -c1-60) band=$band $(date -u +%FT%TZ) GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  : > $LOG
  FREETOKEN_HOST_ENV=$E FREETOKEN_PORT=1920 setsid nohup "$NSYS" launch --session-new=ft$tag --trace=cuda,nvtx \
     --cuda-graph-trace=graph $R/scripts/serve-default.sh >> $LOG 2>&1 < /dev/null &
  local t0=$(date +%s)
  until grep -q "API server is ready" $LOG || [ $(( $(date +%s) - t0 )) -gt 900 ]; do sleep 5; done
  export FREETOKEN_URL=http://127.0.0.1:1920
  for i in 1 2 3; do PROBE_GEN_TOKENS=64 /root/venv/bin/python $R/scripts/probe_decode.py 8000 > /dev/null 2>&1; sleep 10; done
  "$NSYS" start --session=ft$tag --output=$O/$tag --force-overwrite=true
  PROBE_GEN_TOKENS=128 PROBE_PASSES=2 /root/venv/bin/python $R/scripts/probe_decode.py 8000 > $O/$tag-probe.jsonl
  "$NSYS" stop --session=ft$tag
  curl -s http://127.0.0.1:1920/v1/stats > $O/$tag-stats.json
  "$NSYS" shutdown --session=ft$tag --kill=sigterm > /dev/null 2>&1
  sleep 10; pkill -f "ft serve"; sleep 10
  "$NSYS" export --type=sqlite --force-overwrite=true -o $O/$tag.sqlite $O/$tag.nsys-rep > /dev/null 2>&1
  python3 /root/nsys_steps.py $O/$tag.sqlite > $O/$tag-steps.txt 2>&1
  python3 /root/nsys_gap.py $O/$tag.sqlite > $O/$tag-host-gap.txt 2>&1
  rm -f $O/$tag.sqlite
  grep -- "-- pass" $O/$tag-steps.txt; grep -- "-- pass" $O/$tag-host-gap.txt
}
( exec 9>/root/gpu.lock; flock 9
  nsys_one L2 /root/FreeToken-l2 1
  nsys_one L3 /root/FreeToken-gap2 1
  nsys_one L4 $F 1 )
cd $F && ARMS="whole mirror-1m mirror whole-1m" tasks/exclusive-expert-ram/checkpoint-box.sh gap4
cd /root/FreeToken-l2 && ARMS="mirror-np" tasks/exclusive-expert-ram/checkpoint-box.sh lev2
cd /root/FreeToken-gap2 && ARMS="mirror-np" tasks/exclusive-expert-ram/checkpoint-box.sh lev3
cd $F && ARMS="mirror-np" tasks/exclusive-expert-ram/checkpoint-box.sh lev4
cd $F && FT_EXTRA="--moe-collect-stats" FT_ENVS="FREETOKEN_MIRROR_DUP_BAND=0" ARMS="mirror-np" tasks/exclusive-expert-ram/checkpoint-box.sh band0
cd $F && FT_EXTRA="--moe-collect-stats" FT_ENVS="FREETOKEN_MIRROR_DUP_BAND=1" ARMS="mirror-np" tasks/exclusive-expert-ram/checkpoint-box.sh band1
echo G5-GAP3-DONE $(date -u +%FT%TZ)
