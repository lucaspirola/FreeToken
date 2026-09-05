#!/usr/bin/env bash
# Phase D of the 16-lane decode study: the graph-capture A/B at a clean 16-lane batch.
#
# The term under test: on the P2 Switchyard profile (--max-running-requests 16
# --elastic-initial-requests 4) the elastic tier's decode-graph set stopped at 8, so every
# decode batch of 9-16 lanes ran EAGER. FREETOKEN_ELASTIC_GRAPH_MAX_BS=8 reproduces that
# exactly, so before/after are two runs of the SAME binary.
#
# Phase D: the MIDDLE of the 9-15 band, where the fix changes the batch's shape as well
# as its launch path. Before, a 12-lane decode batch ran eager and UNPADDED. After, it
# replays the bs-16 graph with 4 dummy rows -- which route experts of their own. 165 of
# the 13af13d soak's 427 decode batches live in 9-15, so a regression here would matter
# more than the 16-lane win.
#   D1 before  12 lanes, uniform prompts   graphs [1,2,3,4,8]  -> eager at 12, unpadded
#   D2 after   12 lanes, uniform prompts   graphs [1,2,3,4,8,16] -> bs-16 graph, 4 dummies
#
# Invoke through scripts/gpu_lock.sh, NEVER piped:
#   scripts/gpu_lock.sh benchmarks/decode16/phaseD.sh <outdir>
set -uo pipefail
cd /home/lucas/ai/FreeToken
OUT="${1:-benchmarks/decode16/runs/phaseD}"
mkdir -p "$OUT"
exec >>"$OUT/phaseD.log" 2>&1        # redirect INSIDE the wrapped script (gpu_lock kills before a flush)
export PYTHONPATH=/home/lucas/ai/FreeToken/python
export CUDA_VISIBLE_DEVICES=0
export FREETOKEN_PIN_BUDGET_GB=17
MODEL=/home/lucas/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4

# The P2 serve line (benchmarks/switchyard_soak/serve.sh) expressed as bench flags.
P2=(--model "$MODEL" --kv-cache-dtype q8_0 --prefill-chunk 8192 --mem-ratio 0.85
    --cache-policy lfu --nvfp4-backend triton --moe-collect-stats
    --max-context 131072
    --server-arg "--num-tokens 262144 --elastic-initial-requests 4 --kv-grow-step-tokens 65536 --host-ram-reserve-gb 6")

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
  grep -E "capturing CUDA graphs with sizes|Elastic capacity|MoE decode miss stats: " \
      "$OUT/${name}.server.log" 2>/dev/null | tail -12
}

TWELVE=(--concurrency 12 --decode 256)

export FREETOKEN_ELASTIC_GRAPH_MAX_BS=8
arm d1_before_uniform12 "${TWELVE[@]}"
unset FREETOKEN_ELASTIC_GRAPH_MAX_BS
arm d2_after_uniform12 "${TWELVE[@]}"
echo "PHASE D DONE $(date +%T)"
