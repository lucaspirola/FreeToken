#!/bin/bash
# ft-dev: FREETOKEN_PREFILL_CAP_TEST with the native caching allocator (WSL's) and with expandable (native default).
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
while pgrep -f "[c]aptest.sh" > /dev/null; do sleep 20; done
R=/root/FT-ts; O=/root/ts-out; mkdir -p $O; S=/root/captest2-status; rm -f $S
run() {  # tag model rows [extra env lines]
  local tag=$1 model=$2 rows=$3; shift 3
  local E=$O/$tag.env LOG=$O/$tag-server.log
  { echo "export FREETOKEN_MODEL=/root/models/$model"
    echo "export FREETOKEN_MEMORY_RATIO=1.00"
    [ $rows = -1 ] && { echo "export FREETOKEN_MIRROR_EXPERT_RAM=1"; echo "export FREETOKEN_MIRROR_HOST_ROWS=-1"; echo "export FREETOKEN_MIRROR_RESERVE_ROWS=256"; }
    for l in "$@"; do echo "export $l"; done
    echo "export UV_PROJECT_ENVIRONMENT=/root/venv"; echo "export UV_NO_SYNC=1"; echo "export PYTHONPATH=$R/python"; } > $E
  : > $LOG
  exec 9>/root/gpu.lock; flock 9
  FREETOKEN_HOST_ENV=$E FREETOKEN_PORT=1920 setsid nohup $R/scripts/${SERVE:-serve-default.sh} >> $LOG 2>&1 < /dev/null &
  t0=$(date +%s)
  until grep -q "API server is ready\|Traceback" $LOG || [ $(( $(date +%s) - t0 )) -gt 900 ]; do sleep 5; done
  sleep 3; pkill -f "[f]t serve"; sleep 10; pkill -9 -f "[f]t serve"; sleep 2
  flock -u 9; exec 9>&-
  echo "== $tag $(date -u +%T)" >> $S
  grep -h "cap test\|Prefill headroom:" $LOG | sed 's/.*INFO *//' | cut -c1-230 >> $S
}
NE=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
CAP=FREETOKEN_PREFILL_CAP_TEST=96,128,192,256
run nemo-mirror-cap2-ne NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 -1 $NE $CAP
echo ALLDONE >> $S
