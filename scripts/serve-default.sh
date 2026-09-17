#!/usr/bin/env bash
# Default serving profile for the persistent `freetoken-serve` system unit
# (scripts/systemd/freetoken-serve.service.in, installed by scripts/systemd/install.sh).
# Change the profile HERE, not in the unit, and never hand-type flags on another host:
# this file IS the reference configuration (docs/nemotron.md "Default profile").
#
# Single-lane elastic profile: ONE session resident on the GPU at a time (max experts),
# every other session checkpointed to RAM/disk and swapped back in on its turn.
#   --linear-state-slots 13 = the 4-slot working set of that one lane + padding + 8 GDN
#   snapshot slots (~80 MB each) so cold-session restores and prefix snapshots have
#   somewhere to land; with 6 slots the soak logged "no GDN snapshot slot available for
#   cold session restore" on every swap-in and fell back to a full re-prefill.
# KV starts at one 64K step and grows on demand up to 1M tokens, funded from the on-GPU
# expert cache only when VRAM actually runs out. The expert cache is a fixed-capacity VMM
# arena (FREETOKEN_EXPERT_ARENA=1), so growing or shrinking it never reallocates buffers
# and decode CUDA graphs are never recaptured; FREETOKEN_GROWABLE_OVERLAP=1 keeps overlap
# scheduling on while KV is growable.
#
# WHOLE MODEL IN RAM: FREETOKEN_PIN_BUDGET_GB must be >= the MoE expert banks (15.41 GiB
# for Nemotron 3.5 Lightning NVFP4) so every bank is cudaHostRegister'd + mlock'd and no
# layer is silently moved to CPU decode (the WSL auto budget of 0.4 x RAM is too small).
# Pinning that much needs RLIMIT_MEMLOCK=infinity, which only the system unit / the
# user@UID memlock drop-in grant (see scripts/systemd/install.sh). The host needs about
# banks + 4 GiB of free RAM at start (28 GiB machines run it with ~6-7 GiB headroom).
#
# Per-host knobs (environment, or $HOME/.config/freetoken/serve.env which is sourced):
#   FREETOKEN_MODEL               model directory
#   FREETOKEN_PORT                default 1919
#   FREETOKEN_PIN_BUDGET_GB       default 17 (>= expert banks; lower only if RAM is short)
#   FREETOKEN_HOST_RAM_RESERVE_GB default 0 (RAM the preflight keeps free; owner's choice)
#   FREETOKEN_MEMORY_RATIO        default 0.91 of VRAM (KV ceiling + expert cache)
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

MODEL="${FREETOKEN_MODEL:-$HOME/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
CACHE="${FREETOKEN_CACHE_DIR:-$HOME/.cache/freetoken}"
if [ -z "${TVM_FFI_CUDA_ARCH_LIST:-}" ]; then
  TVM_FFI_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)
  export TVM_FFI_CUDA_ARCH_LIST="${TVM_FFI_CUDA_ARCH_LIST:-12.0}"
fi

export FREETOKEN_PIN_BUDGET_GB="${FREETOKEN_PIN_BUDGET_GB:-17}"
export FREETOKEN_SCHEDULER_INVARIANT="${FREETOKEN_SCHEDULER_INVARIANT:-warn}"
export FREETOKEN_EXPERT_ARENA="${FREETOKEN_EXPERT_ARENA:-1}"
export FREETOKEN_ARENA_STEP_SLOTS="${FREETOKEN_ARENA_STEP_SLOTS:-8}"
export FREETOKEN_GROWABLE_OVERLAP="${FREETOKEN_GROWABLE_OVERLAP:-1}"

mkdir -p "$CACHE"/{hidden-states,pooled-sink,spill,trace,logs}

exec uv run ft serve \
  --model "$MODEL" \
  --host 127.0.0.1 --port "${FREETOKEN_PORT:-1919}" \
  --max-running-requests 1 --linear-state-slots 13 --kv-grow-step-tokens 65536 \
  --num-tokens 1048576 --max-seq-len-override 1048576 --kv-cache-dtype q8_0 \
  --attention-backend triton --moe-backend offload --moe-cache-auto --moe-cache-policy lfu \
  --memory-ratio "${FREETOKEN_MEMORY_RATIO:-0.91}" --max-prefill-length 8192 \
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
