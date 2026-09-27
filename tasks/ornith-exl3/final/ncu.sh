#!/usr/bin/env bash
# ncu of the routed-expert prefill kernels (ncu_moe.py), after chain2, GPU idle, host GPU lock.
F=$(dirname "$(readlink -f "$0")"); WT=$(cd "$F/../../.." && pwd); P=$F/profile
while systemctl --user is-active --quiet ft-final-chain2; do sleep 30; done
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock; flock 9
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -le 32 ]; do sleep 5; done
echo "ncu start $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg) $(/usr/local/cuda/bin/ncu --version | tail -1)" > $P/ncu.log
cd $WT && PYTHONPATH=$WT/python /usr/local/cuda/bin/ncu --nvtx --nvtx-include "moe/" \
  -k regex:"_exl3_gemm_kernel|_reconstruct_experts_kernel|_had_rows_kernel|_splitk_combine_kernel" -c 8 \
  --section SpeedOfLight --section WarpStateStats --section Occupancy --section LaunchStats --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis \
  -f -o $P/ncu-moe /home/lucas/ai/FreeToken/.venv/bin/python $F/ncu_moe.py >> $P/ncu.log 2>&1
/usr/local/cuda/bin/ncu -i $P/ncu-moe.ncu-rep --page details > $P/ncu-moe-details.txt 2>&1
echo "NCUDONE $(date -Is)" >> $P/ncu.log
