#!/bin/bash
# ft-g5, after the dyn2 merge-gate rerun. exp/mirror-dma-wb levers (box numbers):
#   L2 40c154d /root/FreeToken-l2, L3 2392b93 /root/FreeToken-gap2, L4 e70849e /root/FreeToken-gap4,
#   L5 e7e029f /root/FreeToken-l5 (fault check + snapshot after the launch).
# 1) GPU tests on L5 (stop if red) 2) nsys 8K L5 3) final arms on L5 4) mirror-np per lever 5) band 0/1 hit rate on L5
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
until grep -q DYN2-G5-DONE /root/dyn2-g5.log 2>/dev/null; do sleep 20; done
F=/root/FreeToken-l5
echo "=== g5-gap5 start $(date -u +%FT%TZ)"
( exec 9>/root/gpu.lock; flock 9; cd $F
  PYTHONPATH=$F/python /root/venv/bin/python -m pytest -q -p no:cacheprovider -rP tests/moe tests/scheduler > /root/g5-gap5-tests.txt 2>&1 )
tail -1 /root/g5-gap5-tests.txt
grep -q " failed\|error" <(tail -1 /root/g5-gap5-tests.txt) && { echo "L5 TESTS RED, stop"; echo G5-GAP5-DONE; exit 1; }
NSYS=$(ls /opt/nvidia/nsight-systems/*/bin/nsys | head -1)
O=$F/tasks/exclusive-expert-ram/results/levers-nsys; mkdir -p $O
( exec 9>/root/gpu.lock; flock 9
  tag=L5; R=$F; E=$O/$tag.env; LOG=$O/$tag-server.log
  { echo "export FREETOKEN_MODEL=/root/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
    echo "export FREETOKEN_MEMORY_RATIO=1.00"
    echo "export FREETOKEN_MIRROR_EXPERT_RAM=1"; echo "export FREETOKEN_MIRROR_HOST_ROWS=-1"; echo "export FREETOKEN_MIRROR_RESERVE_ROWS=256"
    echo "export UV_PROJECT_ENVIRONMENT=/root/venv"; echo "export UV_NO_SYNC=1"; echo "export PYTHONPATH=$R/python"; } > $E
  : > $LOG
  FREETOKEN_HOST_ENV=$E FREETOKEN_PORT=1920 setsid nohup "$NSYS" launch --session-new=ft$tag --trace=cuda,nvtx \
     --cuda-graph-trace=graph $R/scripts/serve-default.sh >> $LOG 2>&1 < /dev/null &
  t0=$(date +%s)
  until grep -q "API server is ready" $LOG || [ $(( $(date +%s) - t0 )) -gt 900 ]; do sleep 5; done
  export FREETOKEN_URL=http://127.0.0.1:1920
  for i in 1 2 3; do PROBE_GEN_TOKENS=64 /root/venv/bin/python $R/scripts/probe_decode.py 8000 > /dev/null 2>&1; sleep 10; done
  "$NSYS" start --session=ft$tag --output=$O/$tag --force-overwrite=true
  PROBE_GEN_TOKENS=128 PROBE_PASSES=2 /root/venv/bin/python $R/scripts/probe_decode.py 8000 > $O/$tag-probe.jsonl
  "$NSYS" stop --session=ft$tag
  curl -s http://127.0.0.1:1920/v1/stats > $O/$tag-stats.json
  "$NSYS" shutdown --session=ft$tag --kill=sigterm > /dev/null 2>&1
  sleep 10; pkill -f "[f]t serve"; sleep 10
  "$NSYS" export --type=sqlite --force-overwrite=true -o $O/$tag.sqlite $O/$tag.nsys-rep > /dev/null 2>&1
  python3 /root/nsys_steps.py $O/$tag.sqlite > $O/$tag-steps.txt 2>&1
  python3 /root/nsys_gap.py $O/$tag.sqlite > $O/$tag-host-gap.txt 2>&1
  rm -f $O/$tag.sqlite
  grep -- "-- pass" $O/$tag-steps.txt $O/$tag-host-gap.txt )
cd $F && ARMS="whole mirror-1m mirror whole-1m" tasks/exclusive-expert-ram/checkpoint-box.sh gap5
cd /root/FreeToken-l2 && ARMS="mirror-np" tasks/exclusive-expert-ram/checkpoint-box.sh lev2
cd /root/FreeToken-gap2 && ARMS="mirror-np" tasks/exclusive-expert-ram/checkpoint-box.sh lev3
cd /root/FreeToken-gap4 && ARMS="mirror-np" tasks/exclusive-expert-ram/checkpoint-box.sh lev4
cd $F && ARMS="mirror-np" tasks/exclusive-expert-ram/checkpoint-box.sh lev5
cd $F && FT_EXTRA="--moe-collect-stats" FT_ENVS="FREETOKEN_MIRROR_DUP_BAND=0" ARMS="mirror-np" tasks/exclusive-expert-ram/checkpoint-box.sh band0
cd $F && FT_EXTRA="--moe-collect-stats" FT_ENVS="FREETOKEN_MIRROR_DUP_BAND=1" ARMS="mirror-np" tasks/exclusive-expert-ram/checkpoint-box.sh band1
echo G5-GAP5-DONE $(date -u +%FT%TZ)
