#!/bin/bash
# Decode timeline on a box tree (flocks /root/gpu.lock). MODEL=nemotron|ornith-exl3, RESIDENCY=whole|mirror,
# MODE=nsys|sync|greedy. nsys: --cuda-graph-trace=graph, split by nsys_split.py (graph replay / eager / idle, and
# the gap between replays). Usage: WT=<tree> MODEL=.. RESIDENCY=.. MODE=.. job-timeline.sh TAG
TAG=${1:?tag}; W=${WT:?tree}; O=/root/K/results/timeline-$TAG; mkdir -p $O
cd $W; export PYTHONPATH=$W/python CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
export FREETOKEN_EXPERT_ARENA=1 FREETOKEN_ARENA_STEP_SLOTS=8 FREETOKEN_GROWABLE_OVERLAP=1 FREETOKEN_SCHEDULER_INVARIANT=warn
RES=""; [ "${RESIDENCY:-whole}" = mirror ] && RES="--expert-residency mirror --moe-mirror-host-rows -1"
case ${MODEL:?} in
  nemotron) export FREETOKEN_PIN_BUDGET_GB=17
    F="--model /root/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --max-running-requests 1 --linear-state-slots 13
       --kv-grow-step-tokens 65536 --num-tokens 1048576 --max-seq-len-override 1048576 --kv-cache-dtype q8_0
       --attention-backend triton --moe-backend offload --moe-cache-auto --moe-cache-policy lfu
       --memory-ratio 1.00 --max-prefill-length 8192 --host-ram-reserve-gb 0 $RES" ;;
  ornith-exl3) export FREETOKEN_EXL3_GEMV_PREROT=1
    F="--model /root/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq --text-model-only --max-running-requests 1
       --kv-grow-step-tokens 65536 --num-tokens 262144 --max-seq-len-override 262144 --kv-cache-dtype q8_0
       --attention-backend triton --moe-backend offload --moe-cache-auto --moe-cache-policy lfu
       --memory-ratio 1.00 --max-prefill-length 8192 $RES" ;;
esac
D=$(cd "$(dirname "$0")" && pwd)   # this script's own tools, whatever tree WT is
exec 9>/root/gpu.lock; flock 9
echo "=== timeline $TAG model=$MODEL residency=${RESIDENCY:-whole} mode=${MODE:?} $(date -u +%FT%TZ) tree $W $(cat $W/SOURCE 2>/dev/null)"
while pgrep -f "ft serve" >/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 16 ]; do sleep 5; done
if [ "$MODE" = nsys ]; then
  timeout 2400 nsys profile -t cuda --cuda-graph-trace=graph --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o $O/decode --force-overwrite true /root/venv/bin/python $D/decode_timeline.py nsys 8000 128 -- $F > $O/run.log 2>&1
  echo "rc=$?"; grep "window:" $O/run.log
  nsys export --type sqlite --force-overwrite true -o $O/decode.sqlite $O/decode.nsys-rep > /dev/null 2>&1
  /root/venv/bin/python $D/nsys_split.py $O/decode.sqlite 127 | grep -v "^tables" | tee $O/split.txt
elif [ "$MODE" = greedy ]; then
  timeout 2400 /root/venv/bin/python $D/decode_timeline.py greedy 8000 1024 -- $F > $O/run.log 2>&1
  echo "rc=$?"; grep "greedy sha1\|window:" $O/run.log | tee $O/greedy.txt
else
  timeout 2400 /root/venv/bin/python $D/decode_timeline.py sync 8000 128 -- $F > $O/run.log 2>&1
  echo "rc=$?"; sed -n '/window:/,$p' $O/run.log | tee $O/sites.txt | head -60
fi
echo "=== timeline $TAG done $(date -u +%FT%TZ)"
