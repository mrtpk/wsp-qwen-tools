#!/usr/bin/env bash
# Usage: score.sh <registry-name>[,<registry-name>...] <categories>
set -euo pipefail
cd "$(dirname "$0")"
export BFCL_PROJECT_ROOT="$PWD" OPENAI_API_KEY=EMPTY
exec @ROOT@/v1/bfcl/.venv/bin/python run_bfcl.py evaluate --model "$1" --test-category "$2"
