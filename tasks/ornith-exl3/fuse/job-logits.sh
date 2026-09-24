#!/bin/bash
# Teacher-forced logits (box): FT with the GEMV-prologue rotation (PREROT=1) and without (PREROT=0, = 0f782b2
# behaviour), both forced on the ids of the step-2/v3 FT run that exllamav3's reference logits were taken on
# (/root/v3-{short,long}-mirror.pt, window start 0), scored against exllamav3 (/root/v3-exl3-*.pt) and against
# each other; plus saver (mirror) vs whole with PREROT=1 (must stay bit-equal). Same flags as v3chain.sh.
# Usage: WT=<tree> job-logits.sh TAG   (each ft_logits run flocks /root/gpu.lock on its own, ~3 min each)
TAG=${1:?tag}; W=${WT:?tree}; O=/root/K/results/exl3logits-$TAG; mkdir -p $O
cd $W; export PYTHONPATH=$W/python
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0 FREETOKEN_EXPERT_ARENA=1
M=/root/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq
COMMON="--model $M --text-model-only --max-running-requests 1 --moe-strategy offload --moe-cache-auto --moe-cache-policy lfu --attention-backend triton --memory-ratio 0.95 --max-seq-len-override 65536 --max-prefill-length 8192"
echo "=== exl3 logits $TAG $(date -u +%FT%TZ) code $(git -C $W log --oneline -1) dirty=$(git -C $W status --porcelain python | wc -l)"
run() { local name=$1 rep=$2 pre=$3; shift 3
  FT_PROMPT_REPEAT=$rep FT_FORCE=/root/v3-${name%%-*}-mirror.pt:0 FREETOKEN_EXL3_GEMV_PREROT=$pre flock /root/gpu.lock timeout 1800 \
    /root/venv/bin/python tasks/ornith-exl3/compare/ft_logits.py $O/$name.pt 32 -- $COMMON "$@" > $O/$name.log 2>&1
  echo "$name rc=$?"; }
for t in short:1 long:40; do
  n=${t%%:*}; rep=${t##*:}
  run $n-pre1-mirror $rep 1 --expert-residency mirror --moe-mirror-host-rows -1
  run $n-pre0-mirror $rep 0 --expert-residency mirror --moe-mirror-host-rows -1
  run $n-pre1-whole $rep 1 --expert-residency whole
  { echo "== $n pre1 saver vs whole"; /root/venv/bin/python -c "
import torch; a=torch.load('$O/$n-pre1-mirror.pt'); b=torch.load('$O/$n-pre1-whole.pt')
print('ids equal', a['output_ids']==b['output_ids'], 'logits bit-equal', torch.equal(a['logits'], b['logits']), 'max|d|', (a['logits'].float()-b['logits'].float()).abs().max().item())"
    echo "== $n FT pre1 vs exllamav3"; /root/venv/bin/python tasks/ornith-exl3/compare/score.py $O/$n-pre1-mirror.pt /root/v3-exl3-$n.pt
    echo "== $n FT pre0 vs exllamav3"; /root/venv/bin/python tasks/ornith-exl3/compare/score.py $O/$n-pre0-mirror.pt /root/v3-exl3-$n.pt
    echo "== $n FT pre1 vs FT pre0"; /root/venv/bin/python tasks/ornith-exl3/compare/score.py $O/$n-pre1-mirror.pt $O/$n-pre0-mirror.pt; } >> $O/score.txt 2>&1
done
cat $O/score.txt
echo "=== exl3 logits $TAG done $(date -u +%FT%TZ)"
