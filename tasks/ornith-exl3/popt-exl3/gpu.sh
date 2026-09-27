#!/usr/bin/env bash
# Run one GPU job ($2...) under the host GPU lock with the popt-exl3 tree on PYTHONPATH, output to $1.
# Launched as a systemd-run --user unit by go.sh (never from an agent shell directly).
export PATH=/usr/lib/wsl/lib:/usr/local/cuda/bin:$PATH
F=$(dirname "$(readlink -f "$0")"); WT=$(cd "$F/../../.." && pwd); OUT=$1; shift
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock; flock 9
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -le 32 ]; do sleep 5; done
{ echo "start $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg) tree ${TREE:-$WT} $(git -C ${TREE:-$WT} log --oneline -1 | cut -c1-50)"
  cd $F && PYTHONPATH=${TREE:-$WT}/python TVM_FFI_CUDA_ARCH_LIST=12.0 "$@"; echo "rc=$? end $(date -Is)"; } > $OUT 2>&1
