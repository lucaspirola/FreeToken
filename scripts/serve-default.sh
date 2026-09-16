#!/usr/bin/env bash
# Default serving profile for the persistent `freetoken-serve` system unit
# (scripts/systemd/freetoken-serve.service). Change the profile HERE, not in the unit.
#
# Static 16-way profile: KV pre-committed (no --kv-grow-step-tokens), so no decode-graph
# recapture ever happens and overlap scheduling stays on. FREETOKEN_PIN_BUDGET_GB=17 keeps
# every MoE expert bank pinned (banks are 15.41 GiB; the WSL auto budget of 0.4xRAM is too
# small and silently moves 7 layers to CPU decode). Elastic promotion is tracked in
# tasks/kv-rollout/.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

export FREETOKEN_PIN_BUDGET_GB="${FREETOKEN_PIN_BUDGET_GB:-17}"
export FREETOKEN_SCHEDULER_INVARIANT="${FREETOKEN_SCHEDULER_INVARIANT:-warn}"
export TVM_FFI_CUDA_ARCH_LIST="${TVM_FFI_CUDA_ARCH_LIST:-12.0}"

CACHE=/home/lucas/.cache/freetoken
mkdir -p "$CACHE"/{hidden-states,pooled-sink,spill,trace,logs}

exec uv run ft serve \
  --model /home/lucas/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --host 127.0.0.1 --port "${FREETOKEN_PORT:-1919}" \
  --max-running-requests 16 \
  --num-tokens 262144 --max-seq-len-override 131072 --kv-cache-dtype q8_0 \
  --attention-backend triton --moe-backend offload --moe-cache-auto --moe-cache-policy lfu \
  --memory-ratio 0.85 --max-prefill-length 8192 \
  --host-ram-reserve-gb "${FREETOKEN_HOST_RAM_RESERVE_GB:-0}" \
  --session-spill-ram-gb 0 --session-spill-dir "$CACHE/spill" \
  --enable-cache-report \
  --served-model-name nemotron-3.5-lightning \
  --served-model-alias nemotron-3.5-lightning-judge \
  --served-model-alias nemotron-3.5-lightning-collect \
  --reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder \
  --force-nonempty-content --max-output-tokens 16384 \
  --trace-dir "$CACHE/trace" \
  --hidden-states-dir "$CACHE/hidden-states" --hidden-states-max-tokens 4096 \
  --pooled-sink-dir "$CACHE/pooled-sink" --pin-prefix-min-tokens 1024 \
  ${FREETOKEN_EXTRA_ARGS:-}
