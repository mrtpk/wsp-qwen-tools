#!/usr/bin/env bash
# Usage: gen.sh <registry-name> <port> <categories>
# The BFCL venv of v1 is used read only (bfcl-eval==2026.3.23). Results go to v2/eval.
set -euo pipefail
cd "$(dirname "$0")"
export BFCL_PROJECT_ROOT="$PWD" OPENAI_BASE_URL="http://localhost:$2/v1" OPENAI_API_KEY=EMPTY
exec @ROOT@/v1/bfcl/.venv/bin/python run_bfcl.py generate --model "$1" --test-category "$3" --num-threads 48
