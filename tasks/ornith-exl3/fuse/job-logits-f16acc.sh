#!/bin/bash
# P2b gate by logits (box): Ornith EXL3 + saver, a ~80K-token prompt (FT_PROMPT_REPEAT filler, 10 chunks),
# 32 greedy tokens:
#   ref   = FREETOKEN_EXL3_F16ACC=0 (fp32 accumulation), free greedy
#   f16   = FREETOKEN_EXL3_F16ACC=1, teacher-forced on ref's ids
#   tile  = F16ACC=0 with another valid extend-attention tile (triton M64 N64 w8 s1), forced: kernel noise
# Usage: WT=<tree> [REPEAT=650] [RATIO=0.90] job-logits-f16acc.sh TAG
TAG=${1:?tag}; W=${WT:?tree}; O=/root/K/results/exl3logits-f16acc-$TAG; mkdir -p $O
cd $W; export PYTHONPATH=$W/python
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
export FREETOKEN_EXPERT_ARENA=1 FREETOKEN_ARENA_STEP_SLOTS=8 FREETOKEN_GROWABLE_OVERLAP=1 FREETOKEN_SCHEDULER_INVARIANT=warn
export FT_PROMPT_REPEAT=${REPEAT:-650}
F="--model /root/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq --text-model-only --max-running-requests 1 --kv-grow-step-tokens 65536 --num-tokens 262144 --max-seq-len-override 262144 --kv-cache-dtype q8_0 --attention-backend triton --moe-backend offload --moe-cache-auto --moe-cache-policy lfu --expert-residency mirror --moe-mirror-host-rows -1 --memory-ratio ${RATIO:-0.90} --max-prefill-length 8192"
SC=tasks/ornith-exl3/compare/ft_logits.py
echo "=== exl3 logits-f16acc $TAG repeat=$FT_PROMPT_REPEAT $(date -u +%FT%TZ) code $(git -C $W log --oneline -1) dirty=$(git -C $W status --porcelain python | wc -l)"
git -C $W diff > $O/tree.diff
FREETOKEN_EXL3_F16ACC=0 flock /root/gpu.lock timeout 2400 /root/venv/bin/python $SC $O/ref.pt 32 -- $F > $O/ref.log 2>&1
echo "ref rc=$? $(grep -h 'emitted window\|^saved\|^text' $O/ref.log | tr '\n' ' ' | cut -c1-200)"
st=$(grep -o "emitted window starts at [0-9]*" $O/ref.log | grep -o "[0-9]*$")
[ -n "$st" ] || { echo "ref failed, stop"; exit 1; }
FREETOKEN_EXL3_F16ACC=1 FT_FORCE=$O/ref.pt:$st flock /root/gpu.lock timeout 2400 /root/venv/bin/python $SC $O/f16.pt 32 -- $F > $O/f16.log 2>&1
echo "f16 rc=$?"
FREETOKEN_EXL3_F16ACC=0 FREETOKEN_EXTEND_BACKEND=triton FREETOKEN_EXTEND_BLOCK_M=64 FREETOKEN_EXTEND_BLOCK_N=64 FREETOKEN_EXTEND_NUM_WARPS=8 FREETOKEN_EXTEND_NUM_STAGES=1 \
  FT_FORCE=$O/ref.pt:$st flock /root/gpu.lock timeout 2400 /root/venv/bin/python $SC $O/tile.pt 32 -- $F > $O/tile.log 2>&1
echo "tile rc=$?"
for b in f16 tile; do echo "== ref vs $b"; /root/venv/bin/python tasks/ornith-exl3/compare/score.py $O/$b.pt $O/ref.pt; done > $O/score.txt 2>&1
cat $O/score.txt
echo "=== exl3 logits-f16acc $TAG done $(date -u +%FT%TZ)"
