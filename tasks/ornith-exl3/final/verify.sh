#!/usr/bin/env bash
# Step 3: scripts/tune-memory-ratio.sh (retired -> verify-memory-ratio.sh) for Ornith EXL3 with the
# new serve-default.sh, on :1920 with the nohup launcher, inside a systemd --user unit, under the
# GPU host lock with the GPU empty and MemAvailable >= 23 GiB [agent practice]. Never :1919.
F=$(dirname "$(readlink -f "$0")"); WT=$(cd "$F/../../.." && pwd)
export PATH=/usr/lib/wsl/lib:/usr/local/cuda/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock; flock 9
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -le 32 ]; do sleep 5; done
until [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -ge 23 ]; do sleep 20; done
echo "verify start $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg) code $(git -C $WT rev-parse --short HEAD)+serve-default" > $F/results/verify.log
cd $WT
VERIFY_LAUNCHER=nohup FREETOKEN_PORT=1920 FREETOKEN_MODEL_NAME=ornith \
UV_PROJECT_ENVIRONMENT=/home/lucas/ai/FreeToken/.venv UV_NO_SYNC=1 PYTHONPATH=$WT/python \
VERIFY_OUT=$F/results/verify-memory-ratio.tsv VERIFY_LOG=$F/results/verify-server.log \
  scripts/tune-memory-ratio.sh >> $F/results/verify.log 2>&1
echo "VERIFYDONE rc=$? $(date -Is)" >> $F/results/verify.log
