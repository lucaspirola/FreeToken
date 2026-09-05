#!/usr/bin/env bash
# Phase E2: the NON-elastic graph ladder, dense-to-16 vs the shipped sparse set.
#
# Phase E (2026-09-05) priced PADDING on the non-elastic path: [1,2,4,8] (a 12-lane batch
# runs eager) vs [1,2,4,8,16] (it pads to 16). It did not test the fix, which is the same
# rule the elastic tier took in 14c1bd8: make the small end DENSE so a 12-lane batch gets
# a bs-12 graph and never pads.
#
# Arms are two invocations of ONE binary, selected by FREETOKEN_GRAPH_DENSE_BS:
#   s16  =0 -> [1,2,4,8,16]         (shipped: 12 pads to 16)
#   d16  =1 -> [1..16]              (dense: 12 replays a bs-12 graph)
# Alternated three times each, because phase E's two repeats of the SAME arm spanned
# 139.2-145.0 tok/s: a single pair cannot resolve a few percent.
#
# Invoke through scripts/gpu_lock.sh, NEVER piped:
#   scripts/gpu_lock.sh benchmarks/decode16/phaseE2.sh <outdir>
set -uo pipefail
cd /home/lucas/ai/FreeToken
OUT="${1:-benchmarks/decode16/runs/phaseE2}"
mkdir -p "$OUT"
exec >>"$OUT/phaseE2.log" 2>&1        # redirect INSIDE the wrapped script
export PYTHONPATH=/home/lucas/ai/FreeToken/python
export CUDA_VISIBLE_DEVICES=0
export FREETOKEN_PIN_BUDGET_GB=17
MODEL=/home/lucas/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4

P2=(--model "$MODEL" --kv-cache-dtype q8_0 --prefill-chunk 8192 --mem-ratio 0.85
    --cache-policy lfu --nvfp4-backend triton --moe-collect-stats
    --max-context 131072
    --server-arg "--num-tokens 262144 --kv-grow-step-tokens 65536 --host-ram-reserve-gb 6 --max-running-requests 16 --cuda-graph-max-bs 16")

arm() {  # arm <name>
  local name="$1"
  echo "=== $(date +%T) ARM $name : GRAPH_DENSE_BS=${FREETOKEN_GRAPH_DENSE_BS} ==="
  uv run python -u benchmarks/bench_decode_moe.py "${P2[@]}" --concurrency 12 --decode 256 \
      --json "$OUT/${name}.json" > "$OUT/${name}.stdout" 2>&1
  echo "=== $(date +%T) ARM $name rc=$? ==="
  local slog
  slog=$(grep -o "server log: .*" "$OUT/${name}.stdout" | head -1 | sed "s/server log: //")
  [ -n "$slog" ] && [ -f "$slog" ] && cp "$slog" "$OUT/${name}.server.log"
  grep -E "aggregate decode|per-stream decode|event gaps" "$OUT/${name}.stdout" | head -5
  grep -E "capturing CUDA graphs with sizes" "$OUT/${name}.server.log" 2>/dev/null | tail -2
}

for rep in 1 2 3; do
  export FREETOKEN_GRAPH_DENSE_BS=0; arm "s16_r${rep}"
  export FREETOKEN_GRAPH_DENSE_BS=1; arm "d16_r${rep}"
done
echo "PHASE E2 DONE $(date +%T)"
