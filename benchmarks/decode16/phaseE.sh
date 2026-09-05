#!/usr/bin/env bash
# Phase E of the 16-lane decode study: the graph-capture A/B at a clean 16-lane batch.
#
# The term under test: on the P2 Switchyard profile (--max-running-requests 16
# --elastic-initial-requests 4) the elastic tier's decode-graph set stopped at 8, so every
# decode batch of 9-16 lanes ran EAGER. FREETOKEN_ELASTIC_GRAPH_MAX_BS=8 reproduces that
# exactly, so before/after are two runs of the SAME binary.
#
# Phase E: the risk check the fix creates. A soak sits at elastic capacity 16 while its
# decode batches drift through 9-15 (165 of the 13af13d soak's 427). Before the fix those
# ran EAGER at their true width; after, they replay the bs-16 graph with dummy rows, and a
# dummy row routes 6 experts of its own. If padding costs more than eager saves, the fix is
# a net loss on the majority of the band.
#
# Both arms are NON-elastic at --max-running-requests 16 (identical GDN/MoE pools, the soak
# steady state) and drive 12 clients. --cuda-graph-max-bs picks the arm:
#   E1  --cuda-graph-max-bs 8   -> graphs [1,2,4,8],    a 12-lane batch runs EAGER at 12
#   E2  --cuda-graph-max-bs 16  -> graphs [1,2,4,8,16], a 12-lane batch replays bs-16
#
# Invoke through scripts/gpu_lock.sh, NEVER piped:
#   scripts/gpu_lock.sh benchmarks/decode16/phaseE.sh <outdir>
set -uo pipefail
cd /home/lucas/ai/FreeToken
OUT="${1:-benchmarks/decode16/runs/phaseE}"
mkdir -p "$OUT"
exec >>"$OUT/phaseE.log" 2>&1        # redirect INSIDE the wrapped script (gpu_lock kills before a flush)
export PYTHONPATH=/home/lucas/ai/FreeToken/python
export CUDA_VISIBLE_DEVICES=0
export FREETOKEN_PIN_BUDGET_GB=17
MODEL=/home/lucas/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4

# The P2 serve line (benchmarks/switchyard_soak/serve.sh) expressed as bench flags.
P2=(--model "$MODEL" --kv-cache-dtype q8_0 --prefill-chunk 8192 --mem-ratio 0.85
    --cache-policy lfu --nvfp4-backend triton --moe-collect-stats
    --max-context 131072
    --server-arg "--num-tokens 262144 --kv-grow-step-tokens 65536 --host-ram-reserve-gb 6 --max-running-requests 16")

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

arm e1_eager12  "${TWELVE[@]}" --server-arg "--cuda-graph-max-bs 8"
arm e2_padded16 "${TWELVE[@]}" --server-arg "--cuda-graph-max-bs 16"
echo "PHASE E DONE $(date +%T)"
