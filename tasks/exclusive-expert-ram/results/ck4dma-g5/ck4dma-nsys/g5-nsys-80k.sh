#!/bin/bash
# nsys trace of one 8K request (prefill + 128 decode steps), whole vs pool (mirror, reserve 256),
# on the dma-wb code (/root/FreeToken-dma), after the ck4dma arms. Diagnosis of the 8K decode gap.

# nsys in cuda-13.0/bin is a stub; install the matching Nsight Systems (after the arms, not during)
command -v /opt/nvidia/nsight-systems/2025.3.2/bin/nsys >/dev/null || \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nsight-systems-2025.3.2 > /root/nsys-install.log 2>&1
NSYS=$(ls /opt/nvidia/nsight-systems/*/bin/nsys | head -1); echo "nsys: $NSYS $($NSYS --version)"
nsys() { "$NSYS" "$@"; }
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
R=/root/FreeToken-dma; O=$R/tasks/exclusive-expert-ram/results/ck4dma-nsys; mkdir -p $O
exec 9>/root/gpu.lock; flock 9
for a in whole mirror; do
  E=$O/$a.env
  { echo "export FREETOKEN_MODEL=/root/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
    echo "export FREETOKEN_MEMORY_RATIO=1.00"
    if [ $a = whole ]; then echo "export FREETOKEN_MIRROR_EXPERT_RAM=0"; echo "export FREETOKEN_MIRROR_HOST_ROWS=0"
    else echo "export FREETOKEN_MIRROR_EXPERT_RAM=1"; echo "export FREETOKEN_MIRROR_HOST_ROWS=-1"; echo "export FREETOKEN_MIRROR_RESERVE_ROWS=256"; fi
    echo "export UV_PROJECT_ENVIRONMENT=/root/venv"; echo "export UV_NO_SYNC=1"; echo "export PYTHONPATH=$R/python"; } > $E
  echo "=== nsys $a $(date -u +%FT%TZ) GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  LOG=$O/$a-80k-server.log; : > $LOG
  FREETOKEN_HOST_ENV=$E FREETOKEN_PORT=1920 setsid nohup "$NSYS" launch --session-new=ft$a --trace=cuda,nvtx \
     --cuda-graph-trace=graph $R/scripts/serve-default.sh >> $LOG 2>&1 < /dev/null &
  t0=$(date +%s)
  until grep -q "API server is ready" $LOG || [ $(( $(date +%s) - t0 )) -gt 900 ]; do sleep 5; done
  echo "ready after $(( $(date +%s) - t0 ))s"
  export FREETOKEN_URL=http://127.0.0.1:1920
  for i in 1 2 3; do PROBE_GEN_TOKENS=64 /root/venv/bin/python $R/scripts/probe_decode.py 8000 > /dev/null 2>&1; sleep 10; done
  nsys start --session=ft$a --output=$O/$a-80k --force-overwrite=true
  PROBE_GEN_TOKENS=128 PROBE_PASSES=2 /root/venv/bin/python $R/scripts/probe_decode.py 80000 | tee $O/$a-80k-probe.jsonl
  nsys stop --session=ft$a
  nsys shutdown --session=ft$a --kill=sigterm 2>&1 | tail -2
  sleep 10; pkill -f "ft serve" ; sleep 10
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
done
echo G5-NSYS-80K-DONE
