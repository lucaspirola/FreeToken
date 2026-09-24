#!/bin/bash
# Task 3 logits (box): EXL3 decode with the split-K reduction in the GEMV + fused epilogues (new) vs the separate
# torch.sum / cast / act / had_rows / combine path (old, same code with both env switches 0), teacher-forced.
# Prompts as step 2: short (FT_PROMPT_REPEAT=1) and long (=40), 31 greedy tokens, RAM saver.
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
export FREETOKEN_EXPERT_ARENA=1 FREETOKEN_ARENA_STEP_SLOTS=8 FREETOKEN_GROWABLE_OVERLAP=1 FREETOKEN_SCHEDULER_INVARIANT=warn
S=/root/t3logits-status; rm -f $S
while ! grep -q ALLDONE /root/logits-status 2>/dev/null; do sleep 30; done
W=/root/FT-exl3-t3; cd $W; SCRIPT=$W/tasks/ornith-exl3/compare/ft_logits.py
O=/root/t3logits; mkdir -p $O
ORN="--model /root/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq --text-model-only --max-running-requests 1 --kv-grow-step-tokens 65536 --num-tokens 262144 --max-seq-len-override 262144 --kv-cache-dtype q8_0 --attention-backend triton --moe-backend offload --moe-cache-auto --moe-cache-policy lfu --expert-residency mirror --moe-mirror-host-rows -1 --memory-ratio 1.00 --max-prefill-length 8192"
for p in short:1 long:40; do
  name=${p%%:*}; rep=${p##*:}
  FT_PROMPT_REPEAT=$rep FREETOKEN_EXL3_SPLITK_INKERNEL=0 FREETOKEN_EXL3_FUSED_EPILOGUE=0 PYTHONPATH=$W/python flock /root/gpu.lock timeout 1800 /root/venv/bin/python $SCRIPT $O/$name-old.pt 31 -- $ORN > $O/$name-old.log 2>&1
  echo "$name old rc=$? $(grep -h '^text' $O/$name-old.log | cut -c1-160)" >> $S
  start=$(grep -o "emitted window starts at [0-9]*" $O/$name-old.log | grep -o "[0-9]*$")
  FT_PROMPT_REPEAT=$rep FT_FORCE=$O/$name-old.pt:$start PYTHONPATH=$W/python flock /root/gpu.lock timeout 1800 /root/venv/bin/python $SCRIPT $O/$name-new.pt 31 -- $ORN > $O/$name-new.log 2>&1
  echo "$name new rc=$?" >> $S
  FT_PROMPT_REPEAT=$rep PYTHONPATH=$W/python flock /root/gpu.lock timeout 1800 /root/venv/bin/python $SCRIPT $O/$name-newfree.pt 31 -- $ORN > $O/$name-newfree.log 2>&1
  echo "$name newfree rc=$? $(grep -h '^text' $O/$name-newfree.log | cut -c1-160)" >> $S
  echo "== $name new vs old (forced)" >> $O/score.txt
  /root/venv/bin/python $W/tasks/ornith-exl3/compare/score.py $O/$name-new.pt $O/$name-old.pt >> $O/score.txt 2>&1
done
echo ALLDONE >> $S
