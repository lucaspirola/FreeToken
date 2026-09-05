#!/usr/bin/env bash
# Phase C of the 16-lane decode study: the graph-capture A/B at a clean 16-lane batch.
#
# The term under test: on the P2 Switchyard profile (--max-running-requests 16
# --elastic-initial-requests 4) the elastic tier's decode-graph set stopped at 8, so every
# decode batch of 9-16 lanes ran EAGER. FREETOKEN_ELASTIC_GRAPH_MAX_BS=8 reproduces that
# exactly, so before/after are two runs of the SAME binary.
#
# Phase C: the same A/B with UNPADDED (uniform, short) prompts. Phase B's padded lanes
# prefill first and finish first, so the 16-lane decode window is short and staggered;
# with 16 identical-shape prompts every lane starts together and the whole measured
# window is a true 16-lane decode batch -- directly comparable to the bs=16 LFU row of
# the 2026-09-04 cache study (168.2 tok/s aggregate).
#   C1 before  16 lanes, uniform prompts   graphs [1,2,3,4,8]
#   C2 after   16 lanes, uniform prompts   graphs [1,2,3,4,8,16]
#   C3 after    1 lane  (single-stream regression; --elastic-initial-requests cannot be
#               combined with --max-running-requests 1, so this arm drops it)
#   C4 after    131K synthetic needle + single-stream long-context decode
#
# Invoke through scripts/gpu_lock.sh, NEVER piped:
#   scripts/gpu_lock.sh benchmarks/decode16/phaseC.sh <outdir>
set -uo pipefail
cd /home/lucas/ai/FreeToken
OUT="${1:-benchmarks/decode16/runs/phaseC}"
mkdir -p "$OUT"
exec >>"$OUT/phaseC.log" 2>&1        # redirect INSIDE the wrapped script (gpu_lock kills before a flush)
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

UNIFORM=(--concurrency 16 --decode 256)

export FREETOKEN_ELASTIC_GRAPH_MAX_BS=8
arm c1_before_uniform16 "${UNIFORM[@]}"
unset FREETOKEN_ELASTIC_GRAPH_MAX_BS
arm c2_after_uniform16 "${UNIFORM[@]}"

# Single stream: max_running_req is forced to 1 by --concurrency 1, and
# --elastic-initial-requests must be strictly smaller than it, so this arm uses the
# same profile minus elastic. The fix cannot touch bs=1 (captured in both arms); this
# is the regression check.
echo "=== $(date +%T) ARM c3_after_single ==="
uv run python -u benchmarks/bench_decode_moe.py --model "$MODEL" --kv-cache-dtype q8_0 \
    --prefill-chunk 8192 --mem-ratio 0.85 --cache-policy lfu --nvfp4-backend triton \
    --moe-collect-stats --max-context 131072 --concurrency 1 --decode 256 \
    --server-arg "--num-tokens 262144 --kv-grow-step-tokens 65536 --host-ram-reserve-gb 6" \
    --json "$OUT/c3_after_single.json" > "$OUT/c3_after_single.stdout" 2>&1
echo "=== $(date +%T) ARM c3 rc=$? ==="
grep -E "decode_tok_s|decode |ttft" "$OUT/c3_after_single.stdout" | tail -10

# 131K needle. prompt + decode + chat template must stay under --max-context, hence
# 130000 rather than 131072 (the Phase B attempt died on exactly that arithmetic).
echo "=== $(date +%T) ARM c4_needle131k ==="
uv run python -u benchmarks/bench_long_context.py --model "$MODEL" \
    --synthetic-needle --needle-depth 0.5 --target-prompt-tokens 130000 \
    --max-context 131072 --decode 128 --kv-cache-dtype q8_0 --prefill-chunk 8192 \
    --mem-ratio 0.85 --cache-policy lfu --nvfp4-backend triton \
    --host-ram-reserve-gb 6 --kv-grow-step-tokens 65536 \
    --json "$OUT/c4_needle131k.json" > "$OUT/c4_needle131k.stdout" 2>&1
echo "=== $(date +%T) ARM c4 rc=$? ==="
tail -25 "$OUT/c4_needle131k.stdout"
echo "PHASE C DONE $(date +%T)"
