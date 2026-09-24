#!/bin/bash
# Re-prove step 2 on the new EXL3 prefill paths, then the serving probe (box numbers).
cd /root/FreeToken
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0 FREETOKEN_EXPERT_ARENA=1
M=/root/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq
S=/root/v3-status; rm -f $S
COMMON="--model $M --text-model-only --max-running-requests 1 --moe-strategy offload --moe-cache-auto --moe-cache-policy lfu --attention-backend triton --memory-ratio 0.95 --max-seq-len-override 65536 --max-prefill-length 8192"
run() { local tag=$1 rep=$2; shift 2
  FT_PROMPT_REPEAT=$rep flock /root/gpu.lock timeout 1800 /root/venv/bin/python tasks/ornith-exl3/compare/ft_logits.py /root/v3-$tag.pt 32 -- $COMMON "$@" > /root/v3-$tag.log 2>&1
  echo "$tag rc=$?" >> $S; }
run short-mirror 1 --expert-residency mirror --moe-mirror-host-rows -1
run short-whole 1 --expert-residency whole
run long-mirror 40 --expert-residency mirror --moe-mirror-host-rows -1
run long-whole 40 --expert-residency whole
for t in short long; do
  EXL3_MOE_CPU_OFFLOAD=24 PYTHONPATH=/root/exllamav3-src flock /root/gpu.lock timeout 3600 /root/exl3-ref/bin/python tasks/ornith-exl3/compare/exl3_logits.py $M /root/v3-$t-mirror.pt /root/v3-exl3-$t.pt > /root/v3-exl3-$t.log 2>&1
  echo "exl3-$t rc=$?" >> $S
done
for t in short long; do
  { echo "== $t saver vs whole"; /root/venv/bin/python -c "
import torch; a=torch.load('/root/v3-$t-mirror.pt'); b=torch.load('/root/v3-$t-whole.pt')
print('ids equal', a['output_ids']==b['output_ids'], 'logits bit-equal', torch.equal(a['logits'], b['logits']), 'max|d|', (a['logits'].float()-b['logits'].float()).abs().max().item())
o=torch.load('/root/v2-$t-mirror.pt'); print('ids equal to pre-change run', a['output_ids']==o['output_ids'])"
    echo "== $t FT(new) vs exllamav3 off24"; /root/venv/bin/python tasks/ornith-exl3/compare/score.py /root/v3-$t-mirror.pt /root/v3-exl3-$t.pt
    echo "== $t FT(new) vs FT(v2, prefill-only change) [only if same ids]"; /root/venv/bin/python tasks/ornith-exl3/compare/score.py /root/v3-$t-mirror.pt /root/v2-$t-mirror.pt 2>&1 | tail -4; } >> /root/v3-scores.txt 2>&1
done
echo scored >> $S
SIZES="8000 32000 80000 128000" TAG=v3 RATIO=1.00 flock /root/gpu.lock /root/step5.sh
echo ALLDONE >> $S
