#!/usr/bin/env bash
# Default serving profile for the persistent `freetoken-serve` system unit
# (scripts/systemd/freetoken-serve.service). Change the profile HERE, not in the unit.
#
# Single-lane elastic profile: ONE session resident on the GPU at a time (max experts,
# 6 GDN state slots), every other session checkpointed to RAM/disk and swapped back in on
# its turn. KV starts at one 64K step and grows on demand up to 1M tokens, funded from
# the on-GPU expert cache only when VRAM actually runs out. The expert cache is a
# fixed-capacity VMM arena (FREETOKEN_EXPERT_ARENA=1), so growing or shrinking it never
# reallocates buffers and decode CUDA graphs are never recaptured.
# FREETOKEN_PIN_BUDGET_GB=17 keeps every MoE expert bank pinned (banks are 15.41 GiB; the
# WSL auto budget of 0.4xRAM is too small and silently moves 7 layers to CPU decode).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

export FREETOKEN_PIN_BUDGET_GB="${FREETOKEN_PIN_BUDGET_GB:-17}"
export FREETOKEN_SCHEDULER_INVARIANT="${FREETOKEN_SCHEDULER_INVARIANT:-warn}"
export FREETOKEN_EXPERT_ARENA="${FREETOKEN_EXPERT_ARENA:-1}"
export FREETOKEN_ARENA_STEP_SLOTS="${FREETOKEN_ARENA_STEP_SLOTS:-8}"
export FREETOKEN_GROWABLE_OVERLAP="${FREETOKEN_GROWABLE_OVERLAP:-1}"
export TVM_FFI_CUDA_ARCH_LIST="${TVM_FFI_CUDA_ARCH_LIST:-12.0}"

CACHE=/home/lucas/.cache/freetoken
mkdir -p "$CACHE"/{hidden-states,pooled-sink,spill,trace,logs}

exec uv run ft serve \
  --model /home/lucas/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --host 127.0.0.1 --port "${FREETOKEN_PORT:-1919}" \
  --max-running-requests 1 --linear-state-slots 6 --kv-grow-step-tokens 65536 \
  --num-tokens 1048576 --max-seq-len-override 1048576 --kv-cache-dtype q8_0 \
  --attention-backend triton --moe-backend offload --moe-cache-auto --moe-cache-policy lfu \
  --memory-ratio 0.91 --max-prefill-length 8192 \
  --host-ram-reserve-gb "${FREETOKEN_HOST_RAM_RESERVE_GB:-0}" \
  --session-spill-ram-gb 1 --session-spill-disk-gb 50 --session-spill-limit-gb 50 \
  --session-spill-dir "$CACHE/spill" \
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
