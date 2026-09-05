#!/usr/bin/env bash
set -uo pipefail
cd /home/lucas/ai/FreeToken
exec > scratchpad/spec_opt/session5.log 2>&1
export FREETOKEN_PIN_BUDGET_GB=17
export PYTHONPATH=python
export FREETOKEN_SPEC_CHECK_COMMIT=25
.venv/bin/python -u benchmarks/probe_spec_ngram_impl.py \
  --model /home/lucas/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --out scratchpad/spec_opt/spec5.json \
  --moe-cache-auto --max-tokens 1024 --skip-needle \
  --only copy --variants v1
echo "EXIT=$?"
