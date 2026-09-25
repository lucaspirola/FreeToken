#!/bin/bash
# 1) ratio-default fixed-size KV arms (rerun: serve-default.sh's --kv-grow-step-tokens 65536
#    cannot be overridden by EXTRA_ARGS since 0 is rejected; use a copy without it).
# 2) "dma-wb on g5": ck4 whole / mirror-1m / mirror / whole-1m on exp/mirror-dma-wb 773d9f8.
until grep -q RATIO-EVIDENCE-DONE /root/ratio-evidence.log; do sleep 20; done
R=/root/FreeToken-ratio
OUTD=$R/tasks/ratio-default/results-g5
cd $R
sed 's/--kv-grow-step-tokens 65536 //' scripts/serve-default.sh > scripts/serve-fixed.sh; chmod +x scripts/serve-fixed.sh
grep -c kv-grow-step scripts/serve-fixed.sh
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
export UV_PROJECT_ENVIRONMENT=/root/venv UV_NO_SYNC=1 PYTHONPATH=$R/python
export VERIFY_LAUNCHER=nohup FREETOKEN_PORT=1920 VERIFY_OUT=$OUTD/verify-memory-ratio.tsv
exec 9>/root/gpu.lock; flock 9
for name in nemotron-fixed ornith-fixed; do
  case $name in nemotron*) M=/root/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4;; *) M=/root/models/Ornith-1.5-35B-A3B-NVFP4;; esac
  [ -f $OUTD/$name-verify.txt ] && mv $OUTD/$name-verify.txt $OUTD/$name-verify-grow0-rejected.txt
  [ -f $OUTD/$name-server.log ] && mv $OUTD/$name-server.log $OUTD/$name-server-grow0-rejected.log
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "=== $name $(date -u +%FT%TZ) GPU ${used} MiB code $(git log --oneline -1 | cut -c1-60)"
  VERIFY_SERVE=$R/scripts/serve-fixed.sh FREETOKEN_MODEL=$M \
    FREETOKEN_EXTRA_ARGS="--num-tokens 262144 --max-seq-len-override 262144" VERIFY_LOG=$OUTD/$name-server.log \
    scripts/verify-memory-ratio.sh > $OUTD/$name-verify.txt 2>&1; echo "exit $?"
  tail -n 3 $OUTD/$name-verify.txt
  sleep 15
done
echo FIXED-DONE
exec 9>&-
# dma-wb
D=/root/FreeToken-dma
if [ ! -d $D ]; then
  git -C /root/FreeToken-headroom worktree add --detach $D 9dac3b5
  cd $D && git apply /root/dmawb.patch && git add -A python scripts tests tasks/exclusive-expert-ram/*.sh tasks/exclusive-expert-ram/*.py \
    && git -c user.name=probe -c user.email=probe@local commit -qm "773d9f8 code (exp/mirror-dma-wb) applied on 9dac3b5" && git log --oneline -1
fi
cd $D
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "=== dma-wb on g5 $(date -u +%FT%TZ) GPU ${used} MiB"
ARMS="whole mirror-1m mirror whole-1m" tasks/exclusive-expert-ram/checkpoint-box.sh ck4dma
echo G5-NEXT-DONE
