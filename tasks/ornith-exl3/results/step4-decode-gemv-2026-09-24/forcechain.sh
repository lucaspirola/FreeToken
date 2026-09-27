#!/bin/bash
# new decode GEMV teacher-forced on the v2 run ids: isolates the kernel change (box)
cd /root/FreeToken
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0 FREETOKEN_EXPERT_ARENA=1
M=/root/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq
COMMON="--model $M --text-model-only --max-running-requests 1 --moe-strategy offload --moe-cache-auto --moe-cache-policy lfu --attention-backend triton --memory-ratio 0.95 --max-seq-len-override 65536 --max-prefill-length 8192 --expert-residency mirror --moe-mirror-host-rows -1"
S=/root/force-status; rm -f $S /root/force-scores.txt
for t in short long; do
  rep=1; [ $t = long ] && rep=40
  FT_FORCE=/root/v2-$t-mirror.pt:0 FT_PROMPT_REPEAT=$rep flock /root/gpu.lock timeout 1800 /root/venv/bin/python tasks/ornith-exl3/compare/ft_logits.py /root/force-$t.pt 32 -- $COMMON > /root/force-$t.log 2>&1
  echo "$t rc=$?" >> $S
  { echo "== $t new GEMV vs v2 GEMV, same ids"; /root/venv/bin/python tasks/ornith-exl3/compare/score.py /root/force-$t.pt /root/v2-$t-mirror.pt
    echo "== $t new GEMV vs exllamav3 off24 on the v2 ids"; /root/venv/bin/python tasks/ornith-exl3/compare/score.py /root/force-$t.pt /root/v2-exl3-$t.pt
    echo "== $t v2 GEMV vs exllamav3 off24 (same ids, for reference)"; /root/venv/bin/python tasks/ornith-exl3/compare/score.py /root/v2-$t-mirror.pt /root/v2-exl3-$t.pt; } >> /root/force-scores.txt 2>&1
done
echo ALLDONE >> $S
