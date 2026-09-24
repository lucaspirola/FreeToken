#!/bin/bash
# v2 kernel check: tests, per-layer breakdown with the new defaults, and the Ornith/bf16 split question.
TAG=${1:-v2}; . "$(dirname "$0")/box-env.sh"
exec 9>/root/gpu.lock; flock 9
echo "=== v2 $TAG $(date -u +%FT%TZ) code $(git -C $R log --oneline -1) dirty=$(git -C $R status --porcelain python | wc -l)"; gpu_idle || exit 1
cd $R; export PYTHONPATH=$R/python
O=$K/results/mb-$TAG; mkdir -p $O
timeout 900 $PY -m pytest -q -p no:cacheprovider tests/kernels/test_kv_quant.py tests/kernels/test_triton_attention.py > $O/tests.txt 2>&1
echo "tests rc=$? $(tail -1 $O/tests.txt)"
$PY benchmarks/bench_decode_attn_breakdown.py --q-heads 32 --kv-heads 2 --head-dim 128 --quant q8_0 \
   --ctx-lens 8192 81920 262144 1048576 --layers 6 --json $O/nemotron-breakdown.jsonl 2>&1 | grep -v Warn
$PY benchmarks/bench_decode_attn_breakdown.py --q-heads 16 --kv-heads 2 --head-dim 256 --quant q8_0 \
   --ctx-lens 8192 81920 262144 --layers 10 --json $O/ornith-breakdown.jsonl 2>&1 | grep -v Warn
for q in q8_0 bf16; do
$PY benchmarks/bench_decode_launch.py --q-heads 16 --kv-heads 2 --head-dim 256 --quant $q \
   --ctx-lens 8192 81920 262144 --splits 42 64 84 128 --block-n 32 64 --warps 4 8 \
   --oracle-max-ctx 8192 --json $O/ornith-$q-splits.jsonl > $O/ornith-$q-splits.txt 2>&1
echo "ornith $q rc=$?"; grep -A4 "^best" $O/ornith-$q-splits.txt
done
echo "=== v2 $TAG done $(date -u +%FT%TZ)"
