#!/usr/bin/env bash
# Default serving profile for the persistent `freetoken-serve` system unit
# (scripts/systemd/freetoken-serve.service.in, installed by scripts/systemd/install.sh).
# Change the profile HERE, not in the unit, and never hand-type flags on another host:
# this file IS the Ornith Q6_K / RTX 2000 Ada reference configuration (CLAUDE.md).
#
# Single-lane elastic profile: ONE session resident on the GPU at a time (max experts),
# every other session checkpointed to RAM/disk and swapped back in on its turn.
# Thirteen recurrent-state slots leave room for the active lane and session snapshots.
# KV starts at one 64K step and grows on demand up to 524288 tokens, funded from the on-GPU
# expert cache only when VRAM actually runs out. The expert cache is a fixed-capacity VMM
# arena (FREETOKEN_EXPERT_ARENA=1), so growing or shrinking it never reallocates buffers
# and decode CUDA graphs are never recaptured; FREETOKEN_GROWABLE_OVERLAP=1 keeps overlap
# scheduling on while KV is growable.
#
# WHOLE MODEL IN RAM: the 32 GiB pin budget covers Ornith Q6_K's ~24.61 GiB expert banks.
# cudaHostRegister page-locks the banks for direct GPU transfers, avoiding CPU decode.
# Unlimited memlock is installed by scripts/systemd/install.sh. Allow process overhead
# plus the 4 GiB host reserve beyond the banks; this profile was tested with 64 GB WSL RAM.
#
# Per-host knobs (environment, or $HOME/.config/freetoken/serve.env which is sourced):
#   FREETOKEN_MODEL               GGUF file (default Ornith-1.5-35B-Q6_K.gguf)
#   FREETOKEN_PORT                default 8080
#   FREETOKEN_MODEL_NAME          default ornith1.5-35b
#   FREETOKEN_REASONING_PARSER    default qwen3
#   FREETOKEN_PIN_BUDGET_GB       default 32 (>= expert banks)
#   FREETOKEN_HOST_RAM_RESERVE_GB default 4
#   FREETOKEN_MEMORY_RATIO        default 1.00; qualify each host with the tuner
#   FREETOKEN_CACHE_DIR           default $HOME/.cache/freetoken (spill, traces, logs)
#   FREETOKEN_EXTRA_ARGS          appended verbatim (last flag wins for repeated options)
#   TVM_FFI_CUDA_ARCH_LIST        auto-detected from nvidia-smi (12.0 Blackwell, 8.9 Ada)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

HOST_ENV="${FREETOKEN_HOST_ENV:-$HOME/.config/freetoken/serve.env}"
if [ -f "$HOST_ENV" ]; then
  # shellcheck disable=SC1090
  . "$HOST_ENV"
fi

MODEL="${FREETOKEN_MODEL:-$HOME/ai/models/Ornith-1.5-35B-Q6_K.gguf}"
CACHE="${FREETOKEN_CACHE_DIR:-$HOME/.cache/freetoken}"
if [ -z "${TVM_FFI_CUDA_ARCH_LIST:-}" ]; then
  TVM_FFI_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)
  export TVM_FFI_CUDA_ARCH_LIST="${TVM_FFI_CUDA_ARCH_LIST:-12.0}"
fi

export FREETOKEN_PIN_BUDGET_GB="${FREETOKEN_PIN_BUDGET_GB:-32}"
export FREETOKEN_SCHEDULER_INVARIANT="${FREETOKEN_SCHEDULER_INVARIANT:-warn}"
export FREETOKEN_EXPERT_ARENA="${FREETOKEN_EXPERT_ARENA:-1}"
export FREETOKEN_ARENA_STEP_SLOTS="${FREETOKEN_ARENA_STEP_SLOTS:-8}"
export FREETOKEN_GROWABLE_OVERLAP="${FREETOKEN_GROWABLE_OVERLAP:-1}"

mkdir -p "$CACHE"/{hidden-states,pooled-sink,spill,trace,logs}

exec uv run --no-sync ft serve \
  --model "$MODEL" \
  --host 127.0.0.1 --port "${FREETOKEN_PORT:-8080}" \
  --max-running-requests 1 --linear-state-slots 13 --kv-grow-step-tokens 65536 \
  --num-tokens 524288 --max-seq-len-override 524288 --kv-cache-dtype q8_0 \
  --rope-yarn-factor 2 --rope-yarn-original-context 262144 \
  --sampling-defaults model --enable-special-token-ckpt --decode-log-interval 10 \
  --attention-backend triton --moe-backend offload --moe-cache-auto --moe-cache-policy lfu \
  --memory-ratio "${FREETOKEN_MEMORY_RATIO:-1.00}" --max-prefill-length 4096 \
  --host-ram-reserve-gb "${FREETOKEN_HOST_RAM_RESERVE_GB:-4}" \
  --session-spill-ram-gb 1 --session-spill-disk-gb 50 --session-spill-limit-gb 50 \
  --session-spill-dir "$CACHE/spill" \
  --enable-cache-report \
  --served-model-name "${FREETOKEN_MODEL_NAME:-ornith1.5-35b}" \
  --served-model-alias "${FREETOKEN_MODEL_NAME:-ornith1.5-35b}-judge" \
  --served-model-alias "${FREETOKEN_MODEL_NAME:-ornith1.5-35b}-collect" \
  --reasoning-parser "${FREETOKEN_REASONING_PARSER:-qwen3}" --tool-call-parser qwen3_coder \
  --force-nonempty-content --max-output-tokens 16384 \
  --trace-dir "$CACHE/trace" \
  --hidden-states-dir "$CACHE/hidden-states" --hidden-states-max-tokens 4096 \
  --pooled-sink-dir "$CACHE/pooled-sink" --pin-prefix-min-tokens 1024 \
  ${FREETOKEN_EXTRA_ARGS:-}
