#!/usr/bin/env bash
# Serving profile for LFM2.5-24B-A2B (Lfm2MoeForCausalLM) as an EXL3 5.0 bpw -hq export.
# A second profile beside scripts/serve-default.sh (Ornith, port 1919), which it does not
# change. Same structure and the same single-lane choices; read serve-default.sh for the
# reasons behind each shared flag. Only the LFM2.5 differences are noted here.
#
# Model: 40 layers = 30 short-conv + 10 full-attention (GQA 32/8, head_dim 64), MoE with
#   64 experts top-4 on layers 2-39, hidden 2048, vocab 128000.
#   - The conv layers keep only a conv state (state_layout "short_conv"): 2 x 2048 values per
#     layer per slot, about 240 KiB per slot for all 30 layers, and no recurrent state.
#   - Routed experts are 38 x 64 x 9.44M weights; at 5 bits that is about 13.4 GiB of expert
#     banks (an estimate from the config, not measured). FREETOKEN_PIN_BUDGET_GB=20 covers them
#     when the whole model is in RAM (RAM saver off).
#   - The exllamav3 export stores an EXL3 lm_head, so the embeddings are untied on load.
# Context: 131072 tokens, the model's native maximum (rope theta 1e7, no YaRN in the config).
# KV: q8_0 keys and values, as in the default profile. Only the 10 attention layers hold KV.
# Parsers: lfm2 reasoning (the chat template always opens <think>; there is no thinking-off
#   switch) and lfm2 tool calls (pythonic lists between <|tool_call_start|> and
#   <|tool_call_end|>).
# Cache directory: a separate default ($HOME/.cache/freetoken/lfm25), so the session spill,
#   traces and hidden states of this server never mix with the Ornith server's.
# Not supported for this model: n-gram speculative decoding (spec_ngram refuses the
#   short_conv state layout).
#
# Per-host knobs (environment, or $HOME/.config/freetoken/serve.env which is sourced):
#   FREETOKEN_LFM25_MODEL         model directory
#   FREETOKEN_PORT                default 1921
#   FREETOKEN_LFM25_CACHE_DIR     default $HOME/.cache/freetoken/lfm25
#   FREETOKEN_LFM25_MIRROR_EXPERT_RAM, FREETOKEN_LFM25_MIRROR_HOST_ROWS
#                                 default 1 and -1 = RAM saver on, auto pool rows (735 rows,
#                                 4.05 GiB pinned instead of the ~13.4 GiB of expert banks).
#                                 The pool keeps a GPU coverage floor of 1896 arena slots; at the
#                                 131072 KV ceiling the arena is 1896-1920 slots (measured
#                                 2026-11-02). Set 0 and 0 for the whole model in RAM. Own names
#                                 because serve.env sets the mirror variables for Ornith.
#   FREETOKEN_PIN_BUDGET_GB,
#   FREETOKEN_HOST_RAM_RESERVE_GB, FREETOKEN_MEMORY_RATIO, FREETOKEN_SESSION_SPILL_RAM_GB,
#   FREETOKEN_MAX_OUTPUT_TOKENS, FREETOKEN_EXTRA_ARGS, TVM_FFI_CUDA_ARCH_LIST
#                                 as in scripts/serve-default.sh
# The model directory has its own variable because serve.env may set FREETOKEN_MODEL for Ornith.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

HOST_ENV="${FREETOKEN_HOST_ENV:-$HOME/.config/freetoken/serve.env}"
if [ -f "$HOST_ENV" ]; then
  # shellcheck disable=SC1090
  . "$HOST_ENV"
fi

MODEL="${FREETOKEN_LFM25_MODEL:-$HOME/ai/models/LFM2.5-24B-A2B-Uncensored-Heretic-exl3-5.0bpw-hq}"
CACHE="${FREETOKEN_LFM25_CACHE_DIR:-$HOME/.cache/freetoken/lfm25}"
if [ -z "${TVM_FFI_CUDA_ARCH_LIST:-}" ]; then
  TVM_FFI_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)
  export TVM_FFI_CUDA_ARCH_LIST="${TVM_FFI_CUDA_ARCH_LIST:-12.0}"
fi

export FREETOKEN_PIN_BUDGET_GB="${FREETOKEN_PIN_BUDGET_GB:-20}"
export FREETOKEN_MIRROR_EXPERT_RAM="${FREETOKEN_LFM25_MIRROR_EXPERT_RAM:-1}"
export FREETOKEN_MIRROR_HOST_ROWS="${FREETOKEN_LFM25_MIRROR_HOST_ROWS:--1}"
export FREETOKEN_SCHEDULER_INVARIANT="${FREETOKEN_SCHEDULER_INVARIANT:-warn}"
export FREETOKEN_EXPERT_ARENA="${FREETOKEN_EXPERT_ARENA:-1}"
export FREETOKEN_ARENA_STEP_SLOTS="${FREETOKEN_ARENA_STEP_SLOTS:-8}"
export FREETOKEN_GROWABLE_OVERLAP="${FREETOKEN_GROWABLE_OVERLAP:-1}"

CTX_TOKENS=131072

mkdir -p "$CACHE"/{hidden-states,pooled-sink,spill,trace,logs}

# float16: with the fp32 residual its KL vs exllamav3 is at the control level (.work/lfm25/router).
exec uv run ft serve \
  --model "$MODEL" --dtype float16 \
  --host 127.0.0.1 --port "${FREETOKEN_PORT:-1921}" \
  --max-running-requests 1 --linear-state-slots 13 --kv-grow-step-tokens 65536 \
  --num-tokens "$CTX_TOKENS" --max-seq-len-override "$CTX_TOKENS" --kv-cache-dtype q8_0 \
  --attention-backend triton --moe-backend offload --moe-cache-auto --moe-cache-policy lfu \
  --memory-ratio "${FREETOKEN_MEMORY_RATIO:-1.00}" --max-prefill-length 8192 \
  --host-ram-reserve-gb "${FREETOKEN_HOST_RAM_RESERVE_GB:-2}" \
  --session-spill-ram-gb "${FREETOKEN_SESSION_SPILL_RAM_GB:-1}" \
  --session-spill-disk-gb 15 --session-spill-limit-gb 15 \
  --session-spill-dir "$CACHE/spill" \
  --enable-cache-report \
  --served-model-name lfm25 \
  --reasoning-parser lfm2 --tool-call-parser lfm2 \
  --force-nonempty-content \
  --max-output-tokens "${FREETOKEN_MAX_OUTPUT_TOKENS:-65536}" \
  --trace-dir "$CACHE/trace" \
  --hidden-states-dir "$CACHE/hidden-states" --hidden-states-max-tokens 4096 \
  --pooled-sink-dir "$CACHE/pooled-sink" --pin-prefix-min-tokens 1024 \
  ${FREETOKEN_EXTRA_ARGS:-}
