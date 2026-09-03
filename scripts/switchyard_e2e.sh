#!/usr/bin/env bash
# Thin wrapper around scripts/switchyard_e2e.py: runs it inside the FreeToken venv
# so the checks reach the same httpx/openai the server was built against.
#
#   scripts/switchyard_e2e.sh contract --base-url http://127.0.0.1:30000
#   scripts/switchyard_e2e.sh soak --duration 20m
#   scripts/switchyard_e2e.sh agents
#
# See docs/switchyard.md for the FreeToken launch line these expect to be running.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if command -v uv >/dev/null 2>&1; then
  exec uv run --project "$repo_root" python "$repo_root/scripts/switchyard_e2e.py" "$@"
fi
exec python3 "$repo_root/scripts/switchyard_e2e.py" "$@"
