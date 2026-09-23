#!/usr/bin/env bash
# Plan S12a: NVFP4 Ornith through the reorganised path, single lane, arena, tuned prefill
# tables (bc6f815), whole model in RAM (the mirror refuses Ornith: 2 experts span shards,
# check_experts). Prefill at 32K/80K (plan) and up to 250K (owner, 2026-09-23: "and why prefil is not being measured, let's say, up to 250k tokens?"), chunk 8192, pass 2, empty GPU.
# Then the roofline inputs the plan asks for, on the same empty GPU after the arm stops:
#   bench_moe_prefill_gemm.py --model ornith --m 8192, and ft bench bw.
# Run as a systemd transient unit (--setenv=PATH), never from an agent shell.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
HERE="$PWD/tasks/exclusive-expert-ram"
OUT="$HERE/results/ornith-s12a"
mkdir -p "$OUT"
PY=/home/lucas/ai/FreeToken/.venv/bin/python
export FT_VENV=/home/lucas/ai/FreeToken/.venv FT_RATIO=1.00
export FT_MODEL="$HOME/ai/models/Ornith-1.5-35B-A3B-NVFP4" FT_NAME=ornith
export FT_EXTRA="--num-tokens 262144 --max-seq-len-override 262144 --served-model-name ornith"
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock
flock 9
empty_gpu() {
  [ -z "$(git status --porcelain -- python)" ] || { echo "python/ dirty"; exit 1; }
  systemctl is-active --quiet freetoken-serve && { echo "freetoken-serve is up"; exit 1; }
  systemctl --user list-units --state=active --no-legend 'ft-measure-*' | grep -q . && { echo "ft-measure unit up"; exit 1; }
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$used" = 0 ] || { echo "GPU holds $used MiB"; exit 1; }
  avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  [ "$avail" -ge 22 ] || { echo "MemAvailable $avail GiB"; exit 1; }
  echo "preflight ok: GPU 0 MiB, MemAvailable $avail GiB, code $(git log --oneline -1)"
}
empty_gpu
PYTHONPATH=$PWD/python "$PY" -m freetoken.models.check_experts "$FT_MODEL" > "$OUT/check_experts.txt" 2>&1 || true
FT_ROWS=0 FT_SIZES="8000 32000 80000 128000 250000" "$HERE/measure.sh" ornith-s12a
journalctl --user -u ft-measure-ornith-s12a -o cat --no-pager | sed 's/\x1b\[[0-9;]*m//g' > "$OUT/ornith-s12a-journal.txt" || true
mv "$HERE"/results/ornith-s12a-* "$OUT/" 2>/dev/null || true
empty_gpu
PYTHONPATH=$PWD/python:$PWD/benchmarks timeout 1800 "$PY" benchmarks/bench_moe_prefill_gemm.py --model ornith --m 8192 \
  > "$OUT/roofline-gemm.txt" 2>&1 || echo "gemm bench exit $?" >> "$OUT/roofline-gemm.txt"
PYTHONPATH=$PWD/python timeout 1800 "$PY" -m freetoken.cli bench bw --dtype nvfp4 -o "$OUT/roofline-bw.json" \
  > "$OUT/roofline-bw.txt" 2>&1 || echo "bw bench exit $?" >> "$OUT/roofline-bw.txt"
echo "s12a done"
