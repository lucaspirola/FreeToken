#!/usr/bin/env bash
# Ornith-1.5-35B-A3B KV-cache quantization study, ONE Hugging Face Job (rtx-pro-6000, 96 GB):
# BF16 weights resident on the GPU (~69 GB), bf16 activations, FreeToken's KV formats emulated
# bit-exactly at FreeToken's write/read points (see kv_eval.py docstring), exact full-vocab
# KL(bf16-KV || lane) per context position on PG19 books (256K + 3 x 32K), teacher-forced
# needles at 32K/128K/254K, YaRN 1.5 vs unscaled at <=32K and a 352K needle.
# Order: setup -> download -> smoke (small slice of every stage + VRAM/time probe at 352K depth
# + projection) -> all -> report. Every section is uploaded as soon as it exists to
# $RES under kv-validation/ (summary.md/json, raw/*.pt, logs).
# Knobs (env): SKIP="smoke" skips the smoke stage; LANES=comma list overrides the lanes;
# BUDGET_MIN (default 150) soft wall budget for the main stage, lower-priority work is skipped
# and reported; EXTRA="..." extra kv_eval.py args for the main stage.
set -euo pipefail

SRC_REPO=ornith-ai/Ornith-1.5-35B-A3B
RES=pirola/Ornith-1.5-35B-A3B-exl3-4.0bpw-hq
PREFIX=kv-validation
SKIP="${SKIP:-}"
BUDGET_MIN="${BUDGET_MIN:-150}"
skip() { [[ " $SKIP " == *" $1 "* ]]; }

mkdir -p /work && cd /work
LOG=/work/kv-job.log
exec > >(tee -a "$LOG") 2>&1
upload_log() { hf upload "$RES" "$LOG" "$PREFIX/job-$1.log" --commit-message "kv job log ($1)" >/dev/null 2>&1 || true; }
trap 'upload_log failed' ERR
stage() { echo; echo "=== $(date -u +%FT%TZ) $* ==="; df -h /work | tail -1; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true; }

stage setup; nvidia-smi; free -g; cat /sys/fs/cgroup/memory.max 2>/dev/null || true
pip install -q "huggingface_hub[cli,hf_xet]>=1.0" "torch==2.11.0" \
  --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
pip install -q "transformers==5.15.1" accelerate safetensors pyarrow
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability(), '| transformers', transformers.__version__)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stage "download full precision ($SRC_REPO)"
hf download "$SRC_REPO" --local-dir /work/in
du -sh /work/in

K="python /work/kv_eval.py --model-dir /work/in --repo $RES --upload-prefix $PREFIX"
LANE_ARG=()
[ -n "${LANES:-}" ] && LANE_ARG=(--lanes "$LANES")

if ! skip smoke; then
  stage "smoke: every stage on a small slice, 3 lanes; VRAM + time probe at 352K depth; projection"
  $K --stage smoke --out-dir /work/smoke --lanes bf16,q4_0,q8_0K_q6_0V \
     --long-len 24576 --short-len 8192 --n-short 1 --bucket-edges 0,8192,16384,24576 \
     --needle-sizes 16384 --yarn-short-len 8192 --yarn-needle-size 16384
  upload_log smoke-ok
fi

stage "main: docs (prefill) -> needles -> yarn -> docs (decode) -> report (budget ${BUDGET_MIN} min)"
# shellcheck disable=SC2086
$K --stage all --out-dir /work/kv-validation --budget-min "$BUDGET_MIN" "${LANE_ARG[@]}" ${EXTRA:-}
upload_log main-ok

stage report
$K --stage report --out-dir /work/kv-validation
upload_log done
echo DONE
