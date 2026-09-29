#!/usr/bin/env bash
# Default serving profile for the persistent `freetoken-serve` system unit
# (scripts/systemd/freetoken-serve.service.in, installed by scripts/systemd/install.sh).
# Change the profile HERE, not in the unit, and never hand-type flags on another host:
# this file IS the reference configuration.
#
# Model (2026-09-27, exp/final-numbers): Ornith-1.5-35B-A3B EXL3 5.0 bpw
# (~/ai/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq), text only, served as `ornith` with the
# qwen3 reasoning parser. Nemotron 3.5 Lightning is no longer served (owner, 2026-09-26).
# Numbers behind every choice below: tasks/ornith-exl3/final/README.md.
#
# Single-lane profile: ONE session resident on the GPU at a time (max experts),
# every other session checkpointed to RAM/disk and swapped back in on its turn.
#   --linear-state-slots 13 = the 4-slot working set of that one lane + padding + 8 GDN
#   snapshot slots so cold-session restores and prefix snapshots have somewhere to land
#   (measured on Nemotron, 2026-09: with 6 slots the soak logged "no GDN snapshot slot
#   available for cold session restore" on every swap-in and fell back to a full re-prefill;
#   Ornith is a GDN hybrid too: 30 linear-attention + 10 full-attention layers).
# Prefix pins stay at the default scope (shared: only a prefix two sessions share is pinned).
#   A session's own haystack survives between its questions through its session lease and
#   the RAM/disk spill (measured on Nemotron 2026-09-23: a 105K haystack hit 104,832 cached
#   tokens across a 30K and a 961K handoff with nothing pinned). The --pin-prefix-scope
#   session option (plan S13) is NOT used here: when the pinned prefix plus the next session
#   exceeded the KV pool, it blocked that session's admission indefinitely
#   (tasks/exclusive-expert-ram/results/s13b-session-stuck.txt). A request without a session
#   key gets no cache across requests: send session-id / x-session-id (or prompt_cache_key).
# Context: 262144 tokens, the model's native maximum, is the default. FREETOKEN_LONG_CONTEXT=1
#   switches to 393216 (1.5x, YaRN factor 2 over the native 262144). Not the default because static
#   YaRN rescales RoPE at EVERY position, so short-prompt outputs differ byte-for-byte from the
#   262144 profile even when the session never approaches 393216 (measured: all 5 natural-text md5
#   change; the text itself stays coherent and correct). It also costs RAM/VRAM unconditionally:
#     - saver pins ~1.39 GiB more host RAM (the pool is sized for the ceiling, paid even if a
#       session never passes 262144);
#     - once a session's KV actually grows past 262144, each 65536-token step funds itself from the
#       GPU expert cache: full 393216 KV costs 768 fewer resident expert slots than full 262144
#       (saver 3856 -> 3088, whole 3920 -> 3168 slots, -20%), only past the old ceiling.
#   Correctness measured clean: 15/15 depth needles (single + multi, 10-98% depth, up to 390K
#   tokens) pass in both saver and whole, character-identical between them; no recall degradation
#   toward the ceiling. Speed: 380K saver prefill 2702 tok/s (TTFT 141s), decode 94 tok/s; <=262144
#   numbers are unchanged from the 262144 profile (same-day control, within normal arm spread).
#   Full tables: tasks/ornith-exl3/ctx393/README.md.
# KV: q8_0 keys and values. The lower lanes (K q8_0/V q6_0, K q6_0/V q5_0, q4_0) free
#   0.3-1.25 GiB of KV at 256K for expert slots, but their prefill runs on the triton extend
#   kernel (kernel/extend_flashinfer.py eligible() takes q8_0/fp8/unquantized only): 256K
#   prefill 855-1441 tok/s vs 3569 on q8_0, and 256K decode 107-114 vs 123 tok/s.
#
# RAM SAVER ON (G3, a switch): the expert residency is the bounded host mirror
# (FREETOKEN_MIRROR_EXPERT_RAM=1, FREETOKEN_MIRROR_HOST_ROWS=-1 = auto rows; the aliases
# server/args.py resolves into --expert-residency mirror --moe-mirror-host-rows -1). Ornith's
# routed experts are 19.28 GiB; the saver pins a ~13.4 GiB pool instead (ram_gib 16.6-16.8 vs
# 19.9-22.2 for the whole model) at 91-92% of the whole model's natural-text decode, 91-97% of
# its probe decode and 96-100% of its prefill (8K-256K, saver/whole in the same period). Set FREETOKEN_MIRROR_EXPERT_RAM=0 FREETOKEN_MIRROR_HOST_ROWS=0
# (serve.env or the environment) for the whole model in RAM; then FREETOKEN_PIN_BUDGET_GB must be
# >= the expert banks (19.28 GiB for Ornith EXL3) so every bank is cudaHostRegister'd + mlock'd
# and no layer is silently moved to CPU decode. Pinning needs RLIMIT_MEMLOCK=infinity, which only
# the system unit / the user@UID memlock drop-in grant (see scripts/systemd/install.sh). The host
# needs the pinned set + ~4 GiB of free RAM at start.
#
# KV starts at one 64K step and grows on demand up to the ceiling, funded from the on-GPU
# expert cache only when VRAM actually runs out. The expert cache is a fixed-capacity VMM
# arena (FREETOKEN_EXPERT_ARENA=1, the alias server/args.py resolves into --expert-arena),
# so growing or shrinking it never reallocates buffers
# and decode CUDA graphs are never recaptured; FREETOKEN_GROWABLE_OVERLAP=1 keeps overlap
# scheduling on while KV is growable.
#
# Per-host knobs (environment, or $HOME/.config/freetoken/serve.env which is sourced):
#   FREETOKEN_MODEL               model directory
#   FREETOKEN_PORT                default 1919
#   FREETOKEN_MIRROR_EXPERT_RAM   default 1 (RAM saver on); 0 with FREETOKEN_MIRROR_HOST_ROWS=0
#                                 = whole model in RAM
#   FREETOKEN_MIRROR_HOST_ROWS    default -1 (auto pool rows)
#   FREETOKEN_PIN_BUDGET_GB       default 20 (>= Ornith EXL3's 19.28 GiB of expert banks, for
#                                 the whole-model residency; lower only if RAM is short)
#   FREETOKEN_HOST_RAM_RESERVE_GB default 0 (RAM the preflight keeps free; owner's choice)
#   FREETOKEN_MEMORY_RATIO        default 1.00 of FREE VRAM (KV ceiling + expert cache),
#                                 the engine's default too (2026-09-24, owner decision):
#                                 an override, not a tuning knob. The engine reserves its
#                                 own runtime headroom (predicted from the config, measured
#                                 at startup); scripts/verify-memory-ratio.sh checks a host.
#                                 It is a fraction of FREE VRAM, so it sizes itself to
#                                 whatever else holds the card -- which is also why a
#                                 measurement taken beside another GPU process is not
#                                 comparable.
#   FREETOKEN_CACHE_DIR           default $HOME/.cache/freetoken (spill, traces, logs)
#   FREETOKEN_LONG_CONTEXT        default 0 (262144, native). 1 = 393216 via YaRN factor 2; see the
#                                 Context note above and tasks/ornith-exl3/ctx393/README.md before
#                                 enabling it -- it changes short-prompt output determinism.
#   FREETOKEN_EXTRA_ARGS          appended verbatim (last flag wins for repeated options)
#   TVM_FFI_CUDA_ARCH_LIST        auto-detected from nvidia-smi (12.0 Blackwell, 8.9 Ada)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

HOST_ENV="${FREETOKEN_HOST_ENV:-$HOME/.config/freetoken/serve.env}"
if [ -f "$HOST_ENV" ]; then
  # shellcheck disable=SC1090
  . "$HOST_ENV"
fi

MODEL="${FREETOKEN_MODEL:-$HOME/ai/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq}"
CACHE="${FREETOKEN_CACHE_DIR:-$HOME/.cache/freetoken}"
if [ -z "${TVM_FFI_CUDA_ARCH_LIST:-}" ]; then
  TVM_FFI_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)
  export TVM_FFI_CUDA_ARCH_LIST="${TVM_FFI_CUDA_ARCH_LIST:-12.0}"
fi

export FREETOKEN_PIN_BUDGET_GB="${FREETOKEN_PIN_BUDGET_GB:-20}"
export FREETOKEN_MIRROR_EXPERT_RAM="${FREETOKEN_MIRROR_EXPERT_RAM:-1}"
export FREETOKEN_MIRROR_HOST_ROWS="${FREETOKEN_MIRROR_HOST_ROWS:--1}"
export FREETOKEN_SCHEDULER_INVARIANT="${FREETOKEN_SCHEDULER_INVARIANT:-warn}"
export FREETOKEN_EXPERT_ARENA="${FREETOKEN_EXPERT_ARENA:-1}"
export FREETOKEN_ARENA_STEP_SLOTS="${FREETOKEN_ARENA_STEP_SLOTS:-8}"
export FREETOKEN_GROWABLE_OVERLAP="${FREETOKEN_GROWABLE_OVERLAP:-1}"

CTX_TOKENS=262144
YARN_ARGS=()
if [ "${FREETOKEN_LONG_CONTEXT:-0}" = "1" ]; then
  CTX_TOKENS=393216
  YARN_ARGS=(--rope-yarn-factor 2 --rope-yarn-original-context 262144)
fi

mkdir -p "$CACHE"/{hidden-states,pooled-sink,spill,trace,logs}

# Output cap: 65536, raised from 16384 on 2026-09-22. Measured on Nemotron's needle
# battery: a multi-hop question over a 20K haystack emitted 125119 characters of
# reasoning and hit finish_reason=length at exactly 16383 completion tokens, having
# already FOUND the fact but never reaching the answer. Ornith also serves thinking ON
# by default, and reasoning is billed from the same budget as the answer, so a cap that
# merely fits the answer truncates the thought that produces it (the ck6o-ck8o needle runs
# on Ornith used 65536 and reached it on "ordering" at 21K/120K). 65536 against a 262144
# context; FREETOKEN_MAX_OUTPUT_TOKENS overrides it per host.
exec uv run ft serve \
  --model "$MODEL" \
  --host 127.0.0.1 --port "${FREETOKEN_PORT:-1919}" \
  --max-running-requests 1 --linear-state-slots 13 --kv-grow-step-tokens 65536 \
  --text-model-only \
  --num-tokens "$CTX_TOKENS" --max-seq-len-override "$CTX_TOKENS" --kv-cache-dtype q8_0 \
  --attention-backend triton --moe-backend offload --moe-cache-auto --moe-cache-policy lfu \
  --memory-ratio "${FREETOKEN_MEMORY_RATIO:-1.00}" --max-prefill-length 8192 \
  --host-ram-reserve-gb "${FREETOKEN_HOST_RAM_RESERVE_GB:-0}" \
  --session-spill-ram-gb 1 --session-spill-disk-gb 50 --session-spill-limit-gb 50 \
  --session-spill-dir "$CACHE/spill" \
  --enable-cache-report \
  --served-model-name ornith \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --force-nonempty-content \
  --max-output-tokens "${FREETOKEN_MAX_OUTPUT_TOKENS:-65536}" \
  --trace-dir "$CACHE/trace" \
  --hidden-states-dir "$CACHE/hidden-states" --hidden-states-max-tokens 4096 \
  --pooled-sink-dir "$CACHE/pooled-sink" --pin-prefix-min-tokens 1024 \
  "${YARN_ARGS[@]}" \
  ${FREETOKEN_EXTRA_ARGS:-}
