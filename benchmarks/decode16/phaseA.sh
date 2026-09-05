#!/usr/bin/env bash
# Phase A of the 16-lane decode attribution study: weightless kernel microbenchmarks.
# One GPU lock, four sweeps, batch 1 vs 16 at the Nemotron 3.5 Lightning geometry.
#   (b) MoE expert GEMV/GEMM at m rows       -> bench_nvfp4_moe_kernels
#   (c) decode attention at 16 lanes x long  -> bench_decode_launch
#   (d) Mamba-2 decode at batch 16           -> bench_mamba2_decode
#   (a) expert movement per step             -> bench_offload_cache_copy
# Invoke through scripts/gpu_lock.sh, NEVER piped:
#   scripts/gpu_lock.sh benchmarks/decode16/phaseA.sh <outdir>
set -uo pipefail
cd /home/lucas/ai/FreeToken
OUT="${1:-benchmarks/decode16/runs/phaseA}"
mkdir -p "$OUT"
exec >>"$OUT/phaseA.log" 2>&1            # redirect INSIDE the wrapped script; gpu_lock kills before a buffered flush
export PYTHONPATH=/home/lucas/ai/FreeToken/python
export CUDA_VISIBLE_DEVICES=0
run() { echo "=== $(date +%T) $* ==="; uv run python -u "$@"; echo "=== $(date +%T) rc=$? ==="; }

run benchmarks/bench_mamba2_decode.py --batch 1 2 4 8 16 \
    --json "$OUT/mamba2_decode.jsonl"

run benchmarks/bench_nvfp4_moe_kernels.py --backend triton b12x \
    --decode-m 1 2 4 8 16 32 --prefill-m 64 \
    --json "$OUT/moe_kernels.jsonl"

run benchmarks/bench_offload_cache_copy.py --models nemotron35-lightning \
    --batch-sizes 1 4 16 --miss-rates 0.0 0.1 0.25 0.5 1.0

run benchmarks/bench_decode_launch.py --q-heads 32 --kv-heads 2 --head-dim 128 \
    --quant q8_0 --ctx-lens 4096 32768 131072 --batch-sizes 1 8 16 \
    --splits 1 2 4 8 16 32 64 --block-n 32 --warps 4 \
    --oracle-max-ctx 0 --json "$OUT/decode_launch.jsonl"
echo "PHASE A DONE $(date +%T)"
