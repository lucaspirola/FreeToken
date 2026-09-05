#!/usr/bin/env bash
# Phase B of the 16-lane decode study: the graph-capture A/B on the real serving path.
#
# The term under test: on the P2 Switchyard profile (--max-running-requests 16
# --elastic-initial-requests 4) the elastic tier's decode-graph set stopped at 8, so every
# decode batch of 9-16 lanes ran EAGER. FREETOKEN_ELASTIC_GRAPH_MAX_BS=8 reproduces that
# exactly, so before/after are two runs of the SAME binary.
#
# Four arms, one GPU lock, one server each:
#   B1 before  16 lanes, mixed contexts (8 long / 8 short)   graphs [1,2,3,4,8]
#   B2 after   16 lanes, mixed contexts                      graphs [1,2,3,4,8,16]
#   B3 after    1 lane   (single-stream regression check)
#   B4 after    131K needle + single-stream decode           (bench_long_context)
#
# Invoke through scripts/gpu_lock.sh, NEVER piped:
#   scripts/gpu_lock.sh benchmarks/decode16/phaseB.sh <outdir>
set -uo pipefail
cd /home/lucas/ai/FreeToken
OUT="${1:-benchmarks/decode16/runs/phaseB}"
mkdir -p "$OUT"
exec >>"$OUT/phaseB.log" 2>&1        # redirect INSIDE the wrapped script (gpu_lock kills before a flush)
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

MIX=(--concurrency 16 --pad-lanes 8 --pad-tokens 16384 --decode 256)

export FREETOKEN_ELASTIC_GRAPH_MAX_BS=8
arm b1_before_mixed16 "${MIX[@]}"
unset FREETOKEN_ELASTIC_GRAPH_MAX_BS
arm b2_after_mixed16 "${MIX[@]}"
arm b3_after_single --concurrency 1 --decode 256

echo "=== $(date +%T) ARM b4_needle131k ==="
uv run python -u benchmarks/bench_long_context.py --model "$MODEL" \
    --synthetic-needle --needle-depth 0.5 --target-prompt-tokens 131072 \
    --max-context 131072 --decode 128 --kv-cache-dtype q8_0 --prefill-chunk 8192 \
    --mem-ratio 0.85 --cache-policy lfu --nvfp4-backend triton \
    --host-ram-reserve-gb 6 --kv-grow-step-tokens 65536 \
    --json "$OUT/b4_needle131k.json" > "$OUT/b4_needle131k.stdout" 2>&1
echo "=== $(date +%T) ARM b4 rc=$? ==="
tail -25 "$OUT/b4_needle131k.stdout"
echo "PHASE B DONE $(date +%T)"
