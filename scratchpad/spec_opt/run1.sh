#!/usr/bin/env bash
set -uo pipefail
cd /home/lucas/ai/FreeToken
exec > scratchpad/spec_opt/session1.log 2>&1
export FREETOKEN_PIN_BUDGET_GB=17
export PYTHONPATH=python
.venv/bin/python -u benchmarks/probe_spec_ngram_impl.py \
  --model /home/lucas/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --out scratchpad/spec_opt/spec1.json \
  --moe-cache-auto --max-tokens 1024 --needle-max-tokens 256 \
  --variants v0 v1
echo "EXIT=$?"
