#!/bin/bash
# exp/ratio-default GPU evidence on ft-g5: verify-memory-ratio.sh at 1.00 (nohup launcher),
# growable (serve-default profile) and fixed-size KV, Nemotron then Ornith NVFP4.
# Run detached:  setsid nohup /root/ratio-evidence.sh > /root/ratio-evidence.log 2>&1 &
R=/root/FreeToken-ratio
OUTD=$R/tasks/ratio-default/results-g5
mkdir -p $OUTD
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
export UV_PROJECT_ENVIRONMENT=/root/venv UV_NO_SYNC=1 PYTHONPATH=$R/python
export VERIFY_LAUNCHER=nohup FREETOKEN_PORT=1920 VERIFY_OUT=$OUTD/verify-memory-ratio.tsv
exec 9>/root/gpu.lock; flock 9
cd $R
run() {  # name model extra
  local name=$1
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "=== $name $(date -u +%FT%TZ) GPU ${used} MiB code $(git log --oneline -1 | cut -c1-60)"
  FREETOKEN_MODEL=$2 FREETOKEN_EXTRA_ARGS="$3" VERIFY_LOG=$OUTD/$name-server.log \
    scripts/verify-memory-ratio.sh > $OUTD/$name-verify.txt 2>&1; echo "exit $?"
  tail -n 3 $OUTD/$name-verify.txt
  sleep 15
}
NEM=/root/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
ORN=/root/models/Ornith-1.5-35B-A3B-NVFP4
run nemotron-growable $NEM ""
run nemotron-fixed $NEM "--kv-grow-step-tokens 0 --num-tokens 262144 --max-seq-len-override 262144"
for _ in $(seq 120); do [ -f /root/ornith-download.done ] && break; sleep 30; done
if [ -f /root/ornith-download.done ]; then
  run ornith-growable $ORN "--num-tokens 262144 --max-seq-len-override 262144"
  run ornith-fixed $ORN "--kv-grow-step-tokens 0 --num-tokens 262144 --max-seq-len-override 262144"
fi
echo RATIO-EVIDENCE-DONE
