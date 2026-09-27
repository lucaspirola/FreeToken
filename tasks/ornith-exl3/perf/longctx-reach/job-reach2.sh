#!/bin/bash
# exl3 worker 01fb928a, long-context reach on Ornith EXL3 5.0bpw saver. In-process longctx_decode.py (warm, pass1-4,
# no nsys). Usage: WT=<tree> CTX=<prompt tokens> NUMTOK=<KV ceiling> [KVD=q8_0|q4_0] [GROW=65536] [YARN=2] [RES=mirror|whole]
#   [ENVS="K=V ..."] job-reach.sh TAG      -> /root/K/results/reach-TAG (run.log, passes.json, summary.txt)
TAG=${1:?tag}; W=${WT:?}; O=/root/K/results/reach-$TAG; mkdir -p $O
cd $W; export PYTHONPATH=$W/python CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
export FREETOKEN_EXPERT_ARENA=1 FREETOKEN_ARENA_STEP_SLOTS=8 FREETOKEN_GROWABLE_OVERLAP=1 FREETOKEN_SCHEDULER_INVARIANT=warn FREETOKEN_EXL3_GEMV_PREROT=1
for kv in ${ENVS:-}; do export $kv; done
R="--expert-residency mirror --moe-mirror-host-rows -1"; [ "${RES:-mirror}" = whole ] && R="--expert-residency whole"
Y=""; [ -n "${YARN:-}" ] && Y="--rope-yarn-factor $YARN"
F="--model /root/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq --text-model-only --max-running-requests 1
   --kv-grow-step-tokens ${GROW:-65536} --num-tokens ${NUMTOK:?} --max-seq-len-override ${NUMTOK} --kv-cache-dtype ${KVD:-q8_0}
   --attention-backend triton --moe-backend offload --moe-cache-auto --moe-cache-policy lfu
   --memory-ratio 1.00 --max-prefill-length 8192 $R $Y"
export LONGCTX_CORPUS=/root/K/corpus-ft/freetoken
exec 9>/root/gpu.lock; flock 9
while pgrep -f "ft serve" >/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")" -gt 16 ]; do sleep 5; done
echo "=== reach $TAG ctx=$CTX numtok=$NUMTOK kvd=${KVD:-q8_0} grow=${GROW:-65536} yarn=${YARN:-} res=${RES:-mirror} envs=${ENVS:-} $(date -u +%FT%TZ) tree $W $(tail -1 $W/SOURCE) memavail_gib=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo) sm=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader) load=$(cut -d\" \" -f1 /proc/loadavg)" | tee $O/summary.txt
echo "$F" > $O/flags.txt
nvidia-smi dmon -s um -d 2 -o T > $O/dmon.txt 2>&1 & DM=$!
timeout 5400 /root/venv/bin/python /root/K/longctx/longctx_decode2.py $CTX 128 $O/passes.json -- $F > $O/run.log 2>&1
rc=$?; kill $DM 2>/dev/null
{ echo "rc=$rc captures=$(grep -c "Start capturing CUDA graphs" $O/run.log) tracebacks=$(grep -c Traceback $O/run.log) oom=$(grep -ci "out of memory" $O/run.log)"
  echo floor_returns=$(grep -c "at the arena floor" $O/run.log)
  grep -E "RuntimeError|refused" $O/run.log | tail -2 | cut -c1-400
  grep -E "^(pass1|pass2|pass4_natural): " $O/run.log | python3 -c "
import sys,json
for l in sys.stdin:
    k,v=l.split(\": \",1); d=json.loads(v); m=d.get(\"mirror_decode_delta\",{})
    print(k, \"prompt\", d[\"prompt_tokens\"], \"prefill\", round(d[\"prefill_tok_s\"]), \"decode\", round(d[\"decode_tok_s\"],1), \"hit\", round(d.get(\"decode_hit_rate\") or 0,3), \"cov_faults\", m.get(\"coverage_faults\"), \"starved\", m.get(\"starved_writebacks\"), \"psha\", d.get(\"prompt_sha1\"), \"osha\", d.get(\"out_sha1\"))"
} | tee -a $O/summary.txt
echo "=== reach $TAG done $(date -u +%FT%TZ)" | tee -a $O/summary.txt
