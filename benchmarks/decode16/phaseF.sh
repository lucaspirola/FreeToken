#!/usr/bin/env bash
# Phase F of the 16-lane decode study: the graph-capture A/B at a clean 16-lane batch.
#
# The term under test: on the P2 Switchyard profile (--max-running-requests 16
# --elastic-initial-requests 4) the elastic tier's decode-graph set stopped at 8, so every
# decode batch of 9-16 lanes ran EAGER. FREETOKEN_ELASTIC_GRAPH_MAX_BS=8 reproduces that
# exactly, so before/after are two runs of the SAME binary.
#
# Phase F: the DENSE capture set (the fix Phase E forced). Phase E showed a sparse ladder
# is worse than no graph for any size that has to pad, so the set is now dense to 16.
#   F1 before  12 lanes at elastic capacity 15, graphs [1..8]  -> EAGER at 12
#   F2 after   12 lanes at elastic capacity 15, graphs [1..15] -> EXACT bs-12 graph
#   F3 after   16 lanes on the P2 profile, graphs [1..16] -- confirms the dense set still
#              fits in VRAM at capacity 16 and re-measures the full-width batch
# Capacity is pinned with --elastic-initial-requests 15 (elastic only ever grows, and the
# 13af13d soak took 421 of its 427 decode batches at capacity 16), so F1/F2 reproduce the
# soak's real shape: a wide pool running a narrower batch.
#
# Invoke through scripts/gpu_lock.sh, NEVER piped:
#   scripts/gpu_lock.sh benchmarks/decode16/phaseF.sh <outdir>
set -uo pipefail
cd /home/lucas/ai/FreeToken
OUT="${1:-benchmarks/decode16/runs/phaseF}"
mkdir -p "$OUT"
exec >>"$OUT/phaseF.log" 2>&1        # redirect INSIDE the wrapped script (gpu_lock kills before a flush)
export PYTHONPATH=/home/lucas/ai/FreeToken/python
export CUDA_VISIBLE_DEVICES=0
export FREETOKEN_PIN_BUDGET_GB=17
MODEL=/home/lucas/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4

# The P2 serve line (benchmarks/switchyard_soak/serve.sh) expressed as bench flags.
P2=(--model "$MODEL" --kv-cache-dtype q8_0 --prefill-chunk 8192 --mem-ratio 0.85
    --cache-policy lfu --nvfp4-backend triton --moe-collect-stats
    --max-context 131072
    --server-arg "--num-tokens 262144 --kv-grow-step-tokens 65536 --host-ram-reserve-gb 6")

arm() {  # arm <name> <extra bench args...>
  local name="$1"; shift
  echo "=== $(date +%T) ARM $name : $* (ELASTIC_GRAPH_MAX_BS=${FREETOKEN_ELASTIC_GRAPH_MAX_BS:-unset}) ==="
  uv run python -u benchmarks/bench_decode_moe.py "${P2[@]}" "$@" \
      --json "$OUT/${name}.json" > "$OUT/${name}.stdout" 2>&1
  echo "=== $(date +%T) ARM $name rc=$? ==="
  # The bench spawns the server with a tempfile log; keep it -- the graph-capture and
  # elastic-capacity lines are the path proof for which arm actually ran.
  local slog
  slog=$(grep -o "server log: .*" "$OUT/${name}.stdout" | head -1 | sed "s/server log: //")
  [ -n "$slog" ] && [ -f "$slog" ] && cp "$slog" "$OUT/${name}.server.log"
  grep -E "aggregate|stream_median|stream_min|decode_tok_s|ms_per_token" "$OUT/${name}.stdout" | head -20
  grep -E "capturing CUDA graphs with sizes|Elastic capacity|GPU memory (before|after) capturing" \
      "$OUT/${name}.server.log" 2>/dev/null | tail -12
}

WIDE=(--server-arg "--max-running-requests 16 --elastic-initial-requests 15")

export FREETOKEN_ELASTIC_GRAPH_MAX_BS=8
arm f1_before_eager12 --concurrency 12 --decode 256 "${WIDE[@]}"
unset FREETOKEN_ELASTIC_GRAPH_MAX_BS
arm f2_after_exact12  --concurrency 12 --decode 256 "${WIDE[@]}"
arm f3_after_dense16  --concurrency 16 --decode 256 \
    --server-arg "--max-running-requests 16 --elastic-initial-requests 4"
echo "PHASE F DONE $(date +%T)"
