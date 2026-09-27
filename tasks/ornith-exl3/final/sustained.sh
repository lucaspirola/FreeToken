#!/usr/bin/env bash
# One sustained-GEMM sample under the GPU host lock (between chain arms, no model loaded).
F=$(dirname "$(readlink -f "$0")")
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock; flock 9
export PATH=/usr/lib/wsl/lib:$PATH
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ "${used:-99999}" -le 32 ] || { echo "$(date -Is) gpu busy: ${used} MiB" >> "$F/profile/sustained_gemm.err"; exit 1; }
/home/lucas/ai/FreeToken/.venv/bin/python "$F/sustained_gemm.py" >> "$F/profile/sustained_gemm.jsonl" 2>> "$F/profile/sustained_gemm.err"
