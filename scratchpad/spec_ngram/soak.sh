#!/usr/bin/env bash
# 10-minute 16-way passthrough soak, spec OFF then spec ON, one arm per server boot.
set -uo pipefail
cd /home/lucas/ai/FreeToken
SOAK_PHASES=passthrough SOAK_PROBE=0 \
  benchmarks/switchyard_soak/run.sh spec_off 10m
sleep 20
SOAK_PHASES=passthrough SOAK_PROBE=0 SOAK_EXTRA_ARGS="--speculative ngram" \
  benchmarks/switchyard_soak/run.sh spec_on 10m
