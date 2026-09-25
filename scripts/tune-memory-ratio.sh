#!/usr/bin/env bash
# Retired 2026-09-24 (owner decision): --memory-ratio is 1.00 by default and is not tuned
# per host any more. The engine predicts its runtime headroom from the config, measures it
# at startup, and reserves or leaves it free itself. This name now runs the verify:
# scripts/verify-memory-ratio.sh (one start at 1.00, 8K/80K/256K, PASS/FAIL, TSV).
# It no longer bisects and no longer writes FREETOKEN_MEMORY_RATIO to serve.env.
echo "tune-memory-ratio.sh is retired: running scripts/verify-memory-ratio.sh (ratio 1.00, no bisection)" >&2
exec "$(dirname "$(readlink -f "$0")")/verify-memory-ratio.sh" "$@"
