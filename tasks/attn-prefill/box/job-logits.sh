#!/bin/bash
# Teacher-forced logits, triton vs flashinfer extend (+ a second triton tile as the noise yardstick) on an
# NVFP4 model at >= 80K prompt tokens (FT_PROMPT_REPEAT filler, 8192-token chunks), 32 greedy tokens.
# Usage: R=<tree> [REPEAT=650] job-logits.sh nemo|ornith
WHICH=${1:?nemo|ornith}; R=${R:?tree}; O=/root/K/results/attnlogits-$WHICH; mkdir -p $O
cd $R; export PYTHONPATH=$R/python CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH FREETOKEN_EXPERT_ARENA=1
export FT_PROMPT_REPEAT=${REPEAT:-650}
if [ $WHICH = ornith ]; then
  F="--model /root/models/Ornith-1.5-35B-A3B-NVFP4 --text-model-only --max-running-requests 1 --kv-grow-step-tokens 65536 --num-tokens 131072 --max-seq-len-override 131072 --kv-cache-dtype q8_0 --attention-backend triton --moe-backend offload --moe-cache-auto --moe-cache-policy lfu --memory-ratio 0.85 --max-prefill-length 8192"
else
  F="--model /root/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --max-running-requests 1 --linear-state-slots 13 --kv-grow-step-tokens 65536 --num-tokens 131072 --max-seq-len-override 131072 --kv-cache-dtype q8_0 --attention-backend triton --moe-backend offload --moe-cache-auto --moe-cache-policy lfu --memory-ratio 0.85 --max-prefill-length 8192"
fi
SC=tasks/ornith-exl3/compare/ft_logits.py
echo "=== attn logits $WHICH repeat=$FT_PROMPT_REPEAT $(date -u +%FT%TZ) code $(git -C $R log --oneline -1) dirty=$(git -C $R status --porcelain python | wc -l)"
FREETOKEN_EXTEND_BACKEND=triton flock /root/gpu.lock timeout 2400 /root/venv/bin/python $SC $O/tri.pt 32 -- $F > $O/tri.log 2>&1
echo "tri rc=$? $(grep -h 'emitted window\|^saved' $O/tri.log | tr '\n' ' ' | cut -c1-200)"
st=$(grep -o "emitted window starts at [0-9]*" $O/tri.log | grep -o "[0-9]*$")
[ -n "$st" ] || { echo "tri failed, stop"; exit 1; }
FREETOKEN_EXTEND_BACKEND=flashinfer FT_FORCE=$O/tri.pt:$st flock /root/gpu.lock timeout 2400 /root/venv/bin/python $SC $O/fi.pt 32 -- $F > $O/fi.log 2>&1
echo "fi rc=$? $(grep -c 'extend attention: flashinfer' $O/fi.log) flashinfer log lines"
FREETOKEN_EXTEND_BACKEND=triton FREETOKEN_EXTEND_BLOCK_M=64 FREETOKEN_EXTEND_BLOCK_N=64 FREETOKEN_EXTEND_NUM_WARPS=8 FREETOKEN_EXTEND_NUM_STAGES=1 \
  FT_FORCE=$O/tri.pt:$st flock /root/gpu.lock timeout 2400 /root/venv/bin/python $SC $O/tile.pt 32 -- $F > $O/tile.log 2>&1
echo "tile rc=$?"
for b in fi tile; do echo "== tri vs $b"; /root/venv/bin/python tasks/ornith-exl3/compare/score.py $O/$b.pt $O/tri.pt; done > $O/score.txt 2>&1
cat $O/score.txt
echo "=== attn logits $WHICH done $(date -u +%FT%TZ)"
