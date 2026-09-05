#!/usr/bin/env bash
set -uo pipefail
cd /home/lucas/ai/FreeToken
export FREETOKEN_PIN_BUDGET_GB=17
export PYTHONPATH=python
.venv/bin/python benchmarks/probe_spec_ngram_impl.py \
  --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --out scratchpad/spec_ngram/spec8.json \
  --moe-cache-auto --max-tokens 1024 --needle-max-tokens 256
echo "EXIT=$?"
