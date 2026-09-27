#!/usr/bin/env bash
# Run a command under the GPU host lock with no model loaded (GPU idle), in the popt-sched tree.
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:/usr/lib/wsl/lib"
cd "$(dirname "$(readlink -f "$0")")/../../.."
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock; flock 9
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -le 32 ]; do sleep 5; done
"$@"
