#!/usr/bin/env bash
# Decide --moe-extend-cache-tokens against the scheduler's real chunk sizes.
#
# One model load; m = 64 / 128 / 256 / 512 on both movement paths (cached decode-slot
# gather vs the full-layer prefill stream), plus a decode burst after each cell so the
# cached arm's damage to the following decode's working set lands on the same row.
# Answers ticket 2 of
# benchmarks/results/nemotron35_lightning_5080_extend_moe_2026-09-05.md §7.
#
# Invoke through scripts/gpu_lock.sh, NEVER piped -- gpu_lock's exit trap runs
# `pkill -9 -g $$` and would kill the reader of a pipe before the last flush:
#
#   FREETOKEN_GPU_LOCK_WAIT=7200 scripts/gpu_lock.sh benchmarks/extend_moe/run_threshold.sh
#
# Optional: $1 is an output directory (default benchmarks/extend_moe/runs/threshold);
# anything after it is passed through to the driver.
set -uo pipefail
cd /home/lucas/ai/FreeToken
OUT="${1:-benchmarks/extend_moe/runs/threshold}"
[ $# -gt 0 ] && shift
mkdir -p "$OUT"
exec >>"$OUT/run_threshold.log" 2>&1     # redirect INSIDE the wrapped script; gpu_lock kills before a buffered flush
export PYTHONPATH=/home/lucas/ai/FreeToken/python
export CUDA_VISIBLE_DEVICES=0
export FREETOKEN_PIN_BUDGET_GB=17
# The gate is forced per arm from inside the driver; an inherited env override would
# silently pin every arm to one threshold.
unset FREETOKEN_MOE_EXTEND_CACHE_TOKENS
MODEL=/home/lucas/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
JSON="$OUT/extend_moe_threshold.json"

echo "=== $(date +%T) bench_extend_moe_threshold start ==="
.venv/bin/python -u benchmarks/bench_extend_moe_threshold.py \
    --model "$MODEL" \
    --m 64 128 256 512 \
    --repeats 7 \
    --base-tokens 4096 \
    --decode-steps 32 \
    --moe-cache-auto \
    --json "$JSON" "$@"
echo "=== $(date +%T) rc=$? ==="
# The JSON stays a run artifact (benchmarks/*/runs/ is gitignored); the numbers that
# survive belong in a benchmarks/results/*.md write-up, not in a copied dump.
echo "DONE $(date +%T)"
