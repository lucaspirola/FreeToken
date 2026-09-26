#!/bin/bash
# P4 correctness by logits (box): Ornith EXL3 + saver, a ~80K-token prompt (FT_PROMPT_REPEAT filler,
# 10 prefill chunks), 32 greedy tokens. Same design as exp/attn-prefill's attn3-box/logits.sh:
#   tri  = FREETOKEN_EXTEND_BACKEND=triton, free greedy (the reference)
#   fi   = flashinfer extend path, teacher-forced on tri's ids
#   tile = triton with another valid tile (M64 N64 w8 s1), teacher-forced: the kernel-noise yardstick
# Usage: WT=<tree> [REPEAT=600] [RATIO=0.90] job-logits-extend.sh TAG
TAG=${1:?tag}; W=${WT:?tree}; O=/root/K/results/exl3logits-extend-$TAG; mkdir -p $O
cd $W; export PYTHONPATH=$W/python
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
export FREETOKEN_EXPERT_ARENA=1 FREETOKEN_ARENA_STEP_SLOTS=8 FREETOKEN_GROWABLE_OVERLAP=1 FREETOKEN_SCHEDULER_INVARIANT=warn
export FT_PROMPT_REPEAT=${REPEAT:-600}
F="--model /root/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq --text-model-only --max-running-requests 1 --kv-grow-step-tokens 65536 --num-tokens 262144 --max-seq-len-override 262144 --kv-cache-dtype q8_0 --attention-backend triton --moe-backend offload --moe-cache-auto --moe-cache-policy lfu --expert-residency mirror --moe-mirror-host-rows -1 --memory-ratio ${RATIO:-0.90} --max-prefill-length 8192"
SC=tasks/ornith-exl3/compare/ft_logits.py
echo "=== exl3 logits-extend $TAG repeat=$FT_PROMPT_REPEAT $(date -u +%FT%TZ) code $(git -C $W log --oneline -1) dirty=$(git -C $W status --porcelain python | wc -l)"
git -C $W diff > $O/tree.diff
FREETOKEN_EXTEND_BACKEND=triton flock /root/gpu.lock timeout 2400 /root/venv/bin/python $SC $O/tri.pt 32 -- $F > $O/tri.log 2>&1
echo "tri rc=$? $(grep -h 'emitted window\|^saved\|^text' $O/tri.log | tr '\n' ' ' | cut -c1-240)"
st=$(grep -o "emitted window starts at [0-9]*" $O/tri.log | grep -o "[0-9]*$")
[ -n "$st" ] || { echo "tri failed, stop"; exit 1; }
FT_FORCE=$O/tri.pt:$st flock /root/gpu.lock timeout 2400 /root/venv/bin/python $SC $O/fi.pt 32 -- $F > $O/fi.log 2>&1
echo "fi rc=$? $(grep -h 'flashinfer' $O/fi.log | head -1 | cut -c1-120)"
FREETOKEN_EXTEND_BACKEND=triton FREETOKEN_EXTEND_BLOCK_M=64 FREETOKEN_EXTEND_BLOCK_N=64 FREETOKEN_EXTEND_NUM_WARPS=8 FREETOKEN_EXTEND_NUM_STAGES=1 \
  FT_FORCE=$O/tri.pt:$st flock /root/gpu.lock timeout 2400 /root/venv/bin/python $SC $O/tile.pt 32 -- $F > $O/tile.log 2>&1
echo "tile rc=$?"
for b in fi tile; do echo "== tri vs $b"; /root/venv/bin/python tasks/ornith-exl3/compare/score.py $O/$b.pt $O/tri.pt; done > $O/score.txt 2>&1
cat $O/score.txt
echo "=== exl3 logits-extend $TAG done $(date -u +%FT%TZ)"
