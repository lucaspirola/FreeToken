#!/bin/bash
# Decode-attention kernel microbench (no server): production launch breakdown + launch sweep.
# Usage: job-microbench.sh TAG   (runs under /root/gpu.lock)
TAG=${1:-base}; . "$(dirname "$0")/box-env.sh"
exec 9>/root/gpu.lock; flock 9
echo "=== microbench $TAG $(date -u +%FT%TZ) code $(git -C $R log --oneline -1)"; gpu_idle || exit 1
cd $R; export PYTHONPATH=$R/python
O=$K/results/mb-$TAG; mkdir -p $O
$PY benchmarks/bench_decode_attn_breakdown.py --q-heads 32 --kv-heads 2 --head-dim 128 --quant q8_0 \
   --ctx-lens 8192 81920 262144 1048576 --layers 6 --json $O/nemotron-breakdown.jsonl
$PY benchmarks/bench_decode_attn_breakdown.py --q-heads 16 --kv-heads 2 --head-dim 256 --quant q8_0 \
   --ctx-lens 8192 81920 262144 --layers 10 --json $O/ornith-breakdown.jsonl
if [ "${SWEEP:-1}" = 1 ]; then
$PY benchmarks/bench_decode_launch.py --q-heads 32 --kv-heads 2 --head-dim 128 --quant q8_0 \
   --ctx-lens 81920 262144 1048576 --splits 32 42 64 84 126 168 256 --block-n 32 64 128 --warps 4 8 \
   --oracle-max-ctx 81920 --json $O/nemotron-sweep.jsonl > $O/nemotron-sweep.txt 2>&1
fi
echo "=== microbench $TAG done $(date -u +%FT%TZ)"
