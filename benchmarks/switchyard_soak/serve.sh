#!/usr/bin/env bash
# P2 serving profile (docs/nemotron.md) + Switchyard compliance flags (docs/switchyard.md §1).
# This is the line the passing soaks (§U onward) were run with; keep it verbatim so a
# result is comparable across runs. Invoke through scripts/gpu_lock.sh, never piped.
#
# SOAK_EXTRA_ARGS appends flags for an A/B (e.g. "--speculative ngram"). Leave it unset for
# a comparable baseline -- a run with it set is NOT the reference profile.
#
# SOAK_HOST_RAM_RESERVE_GB overrides --host-ram-reserve-gb (default 6, the reference value:
# soak AA9.5 asked for the knob because the flag was hard-coded). It raises the *static*
# pre-load bank preflight only -- the load-phase transient is bounded by run.sh's
# SOAK_RAM_LOAD_ABORT_GIB watchdog, not by this.
#
# FREETOKEN_HIDDEN_STATES_DIR overrides the Switchyard hidden-state probe root
# (docs/switchyard.md §6). The server refuses a missing directory at parse time, so it is
# created here; FreeToken never cleans it -- the consumer deletes each artifact it has scored.
set -euo pipefail
cd /home/lucas/ai/FreeToken
export FREETOKEN_HIDDEN_STATES_DIR="${FREETOKEN_HIDDEN_STATES_DIR:-/home/lucas/.cache/freetoken/hidden-states}"
mkdir -p "$FREETOKEN_HIDDEN_STATES_DIR"
export FREETOKEN_PIN_BUDGET_GB="${FREETOKEN_PIN_BUDGET_GB:-17}"
export FREETOKEN_SCHEDULER_INVARIANT="${FREETOKEN_SCHEDULER_INVARIANT:-warn}"
exec uv run ft serve \
  --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --host 127.0.0.1 --port "${SOAK_PORT:-1919}" \
  --max-running-requests 16 --elastic-initial-requests 4 --kv-grow-step-tokens 65536 \
  --num-tokens 262144 --max-seq-len-override 131072 --kv-cache-dtype q8_0 \
  --attention-backend triton --moe-backend offload --moe-cache-auto \
  --moe-cache-policy lfu \
  --memory-ratio 0.85 --max-prefill-length 8192 \
  --host-ram-reserve-gb "${SOAK_HOST_RAM_RESERVE_GB:-6}" \
  --enable-cache-report --served-model-name nemotron-3.5-lightning \
  --reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder \
  --force-nonempty-content --max-output-tokens 16384 \
  --hidden-states-dir "$FREETOKEN_HIDDEN_STATES_DIR" --hidden-states-max-tokens 4096 \
  ${SOAK_EXTRA_ARGS:-}
