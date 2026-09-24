#!/bin/bash
# Stage-1 variant sweep (block-dot on/off x stages x tile x splits) + kernel tests. Usage: job-sweep.sh TAG
TAG=${1:-v1}; . "$(dirname "$0")/box-env.sh"
exec 9>/root/gpu.lock; flock 9
echo "=== sweep $TAG $(date -u +%FT%TZ) code $(git -C $R log --oneline -1) dirty=$(git -C $R status --porcelain python | wc -l)"; gpu_idle || exit 1
cd $R; export PYTHONPATH=$R/python
O=$K/results/sweep-$TAG; mkdir -p $O
timeout 900 $PY -m pytest -q -p no:cacheprovider tests/kernels/test_kv_quant.py tests/kernels/test_triton_attention.py > $O/tests.txt 2>&1
echo "tests rc=$? $(tail -1 $O/tests.txt)"
$PY benchmarks/bench_decode_launch.py --q-heads 32 --kv-heads 2 --head-dim 128 --quant q8_0 \
   --ctx-lens 8192 81920 262144 1048576 ${SWEEP_ARGS:---splits 64 84 128 168 256 --block-n 32 64 --warps 4 8 --stages 2 3 --block-dot 0 1} \
   --oracle-max-ctx 81920 --json $O/nemotron.jsonl > $O/nemotron.txt 2>&1
echo "nemotron rc=$?"; tail -6 $O/nemotron.txt
$PY benchmarks/bench_decode_launch.py --q-heads 32 --kv-heads 2 --head-dim 128 --quant q8_0 \
   --ctx-lens 8192 81920 262144 1048576 --splits 128 256 --block-n 64 --warps 8 --stages 2 --block-dot 0 1 \
   --split-min 0 1024 4096 --oracle-max-ctx 8192 --json $O/nemotron-splitmin.jsonl > $O/nemotron-splitmin.txt 2>&1
echo "splitmin rc=$?"
$PY benchmarks/bench_decode_launch.py --q-heads 16 --kv-heads 2 --head-dim 256 --quant q8_0 \
   --ctx-lens 8192 81920 262144 ${SWEEP_ARGS_ORNITH:---splits 64 128 --block-n 32 64 --warps 4 8 --stages 2 3 --block-dot 0 1} \
   --oracle-max-ctx 81920 --json $O/ornith.jsonl > $O/ornith.txt 2>&1
echo "ornith rc=$?"; tail -5 $O/ornith.txt
echo "=== sweep $TAG done $(date -u +%FT%TZ)"
