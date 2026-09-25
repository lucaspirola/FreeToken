#!/bin/bash
# ft-dev: caching-allocator snapshots of the startup prefill-transient measurement (FT-ts = exp/dt-dma
# 583afc8 + FREETOKEN_TRANSIENT_SNAPSHOT). Startup only: ready -> stop. Lock per start.
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
R=/root/FT-ts; O=/root/ts-out; mkdir -p $O; S=/root/snapdev-status; rm -f $S
run() {  # tag model rows [extra env lines]
  local tag=$1 model=$2 rows=$3; shift 3
  local E=$O/$tag.env LOG=$O/$tag-server.log
  { echo "export FREETOKEN_MODEL=/root/models/$model"
    echo "export FREETOKEN_MEMORY_RATIO=1.00"
    [ $rows = -1 ] && { echo "export FREETOKEN_MIRROR_EXPERT_RAM=1"; echo "export FREETOKEN_MIRROR_HOST_ROWS=-1"; echo "export FREETOKEN_MIRROR_RESERVE_ROWS=256"; }
    echo "export FREETOKEN_TRANSIENT_SNAPSHOT=$O/$tag"
    for l in "$@"; do echo "export $l"; done
    echo "export UV_PROJECT_ENVIRONMENT=/root/venv"; echo "export UV_NO_SYNC=1"; echo "export PYTHONPATH=$R/python"; } > $E
  : > $LOG
  exec 9>/root/gpu.lock; flock 9
  nvidia-smi --query-gpu=memory.used --format=csv,noheader > $O/$tag-smi-before.txt
  FREETOKEN_HOST_ENV=$E FREETOKEN_PORT=1920 setsid nohup $R/scripts/serve-default.sh >> $LOG 2>&1 < /dev/null &
  t0=$(date +%s)
  until grep -q "API server is ready" $LOG || [ $(( $(date +%s) - t0 )) -gt 900 ]; do sleep 5; done
  sleep 3; pkill -f "[f]t serve"; sleep 10; pkill -9 -f "[f]t serve"; sleep 2
  flock -u 9; exec 9>&-
  echo "$tag $(ls $O/$tag 2>/dev/null | wc -l) pickles $(date -u +%T): $(grep -h 'Prefill headroom' $LOG | sed 's/.*Prefill headroom/Prefill headroom/' | cut -c1-260)" >> $S
}
run nemo-mirror NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 -1
run ornith Ornith-1.5-35B-A3B-NVFP4 0
run nemo-whole NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 0
echo ALLDONE >> $S
