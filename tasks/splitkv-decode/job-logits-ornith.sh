#!/bin/bash
# Teacher-forced decode logits, base vs v2, Ornith NVFP4 (whole model) on greedy_client.py's 80K
# haystack (thinking off): base greedy, then v2 forced on base's ids; scored with score.py.
# Usage: job-logits-ornith.sh SIZE
SIZE=${1:-80000}; . "$(dirname "$0")/box-env.sh"; S=$(cd "$(dirname "$0")" && pwd)
O=$K/results/logits-ornith-$SIZE; mkdir -p $O
$PY $S/greedy_client.py --dump $SIZE $O/prompt.txt
F="--model /root/models/Ornith-1.5-35B-A3B-NVFP4 --text-model-only --max-running-requests 1 --attention-backend triton --kv-cache-dtype q8_0 --moe-backend offload --moe-cache-auto --moe-cache-policy lfu --memory-ratio ${RATIO:-0.85} --max-seq-len-override 131072 --num-tokens 131072 --max-prefill-length 8192"
run() { local tag=$1 tree=$2; shift 2
  ( flock 9; gpu_idle || exit 1
    echo "=== logits-ornith $tag $(date -u +%FT%TZ) code $(git -C $tree log --oneline -1) dirty=$(git -C $tree status --porcelain -uno | wc -l)"
    cd $tree; PYTHONPATH=$tree/python FT_PROMPT_FILE=$O/prompt.txt FT_NO_THINK=1 FREETOKEN_EXPERT_ARENA=1 "$@" timeout 2400 \
      $PY $S/ft_logits.py $O/$tag.pt 32 -- $F > $O/$tag.log 2>&1; echo "$tag rc=$?"; grep -E "window|saved|text:" $O/$tag.log
  ) 9>/root/gpu.lock; }
run base /root/FT-splitkv-base
st=$(grep -o "window starts at [0-9]*" $O/base.log | grep -o "[0-9]*$")
[ -n "$st" ] || { echo "base run failed, no window start"; exit 1; }
run v2 /root/FT-splitkv env FT_FORCE=$O/base.pt:$st
run base-rerun /root/FT-splitkv-base env FT_FORCE=$O/base.pt:$st
SC=/root/FT-exl3p/tasks/ornith-exl3/compare/score.py
{ echo "== v2 vs base"; $PY $SC $O/v2.pt $O/base.pt; echo "== base rerun vs base"; $PY $SC $O/base-rerun.pt $O/base.pt
  $PY - <<PY
import torch
b = torch.load("$O/base.pt"); v = torch.load("$O/v2.pt")
lb, lv = b["logits"].float(), v["logits"].float()
t2 = lb.topk(2, -1)
for i in range(lb.shape[0]):
    gap = (t2.values[i, 0] - t2.values[i, 1]).item()
    d = (lb[i] - lv[i]).abs().max().item()
    flag = " <- argmax differs" if lb[i].argmax() != lv[i].argmax() else ""
    print(f"pos {i:2d} base top1-top2 gap {gap:7.4f}  max|dlogit| {d:7.4f}{flag}")
PY
} > $O/score.txt 2>&1
cat $O/score.txt
