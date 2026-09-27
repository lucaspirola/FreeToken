#!/usr/bin/env bash
# ncu of the routed-expert DECODE kernels (ncu_decode.py), GPU idle, host GPU lock.
export PATH=/usr/lib/wsl/lib:/usr/local/cuda/bin:$PATH
F=$(dirname "$(readlink -f "$0")"); WT=$(cd "$F/../../.." && pwd); P=$F/profile; TAG=${1:-base}
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock; flock 9
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -le 32 ]; do sleep 5; done
echo "ncu start $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg) tree $(git -C $WT log --oneline -1 | cut -c1-60)" > $P/ncu-$TAG.log
cd $WT && PYTHONPATH=$WT/python TVM_FFI_CUDA_ARCH_LIST=12.0 /usr/local/cuda/bin/ncu --nvtx --nvtx-include "moe/" \
  -k regex:"$4" \
  --section SpeedOfLight --section WarpStateStats --section Occupancy --section LaunchStats --section MemoryWorkloadAnalysis \
  --section ComputeWorkloadAnalysis --section InstructionStats \
  -f -o $P/ncu-$TAG /home/lucas/ai/FreeToken/.venv/bin/python $F/ncu_moe_m.py $2 $3 >> $P/ncu-$TAG.log 2>&1
/usr/local/cuda/bin/ncu -i $P/ncu-$TAG.ncu-rep --page details > $P/ncu-$TAG-details.txt 2>&1
echo "NCUDONE $(date -Is)" >> $P/ncu-$TAG.log
