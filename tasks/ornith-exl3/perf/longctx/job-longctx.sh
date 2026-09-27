#!/bin/bash
# Long-context decode profile, one (model, context) arm per job. Flocks /root/gpu.lock and waits for an
# empty GPU. Box ft-dev.
# Usage: WT=<tree> MODEL=nemotron|ornith-exl3 [RESIDENCY=mirror|whole] job-longctx.sh CTX
# The saver is the default residency. The tree must carry the overlap-sync fix (exp/overlap-sync).
# Output goes to /root/K/results/longctx-$MODEL-$RESIDENCY-$CTX: run.log, passes.json (timed passes, miss
# counters), decode.nsys-rep/.sqlite (pass 3, decode only, node-level graph trace), split.txt/.json.
CTX=${1:?ctx}; W=${WT:?tree}; RS=${RESIDENCY:-mirror}; O=/root/K/results/longctx-${MODEL:?}-$RS-$CTX; mkdir -p $O
cd $W; export PYTHONPATH=$W/python CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
export FREETOKEN_EXPERT_ARENA=1 FREETOKEN_ARENA_STEP_SLOTS=8 FREETOKEN_GROWABLE_OVERLAP=1 FREETOKEN_SCHEDULER_INVARIANT=warn
RES="--expert-residency whole"; [ "$RS" = mirror ] && RES="--expert-residency mirror --moe-mirror-host-rows -1"
case $MODEL in
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
  *) echo "MODEL?"; exit 2 ;;
esac
D=$(cd "$(dirname "$0")" && pwd)
exec 9>/root/gpu.lock; flock 9
while pgrep -f "ft serve" >/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 16 ]; do sleep 5; done
echo "=== longctx $MODEL $RS $CTX $(date -u +%FT%TZ) tree $W $(cat $W/SOURCE 2>/dev/null) gpu_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) memavail_gib=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)"
timeout 2700 nsys profile -t cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o $O/decode --force-overwrite true /root/venv/bin/python $D/longctx_decode.py $CTX 128 $O/passes.json -- $F > $O/run.log 2>&1
echo "rc=$?"; grep -E "^(warm|pass[0-9])" $O/run.log | cut -c1-400
nsys export --type sqlite --force-overwrite true -o $O/decode.sqlite $O/decode.nsys-rep > /dev/null 2>&1
/root/venv/bin/python $D/longctx_split.py $O/decode.sqlite $O/run.log $CTX --json $O/split.json > $O/split.txt 2>&1
head -16 $O/split.txt
echo "captures=$(grep -c 'Start capturing CUDA graphs' $O/run.log) tracebacks=$(grep -c Traceback $O/run.log) oom=$(grep -ci 'out of memory' $O/run.log)"
echo "=== longctx $MODEL $RS $CTX done $(date -u +%FT%TZ)"
