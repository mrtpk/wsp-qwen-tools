#!/usr/bin/env bash
# Usage: gen_retry.sh <registry-name> <port> <categories>   (retry diagnostic; see run_bfcl_retry.py)
set -euo pipefail
cd "$(dirname "$0")"
export BFCL_PROJECT_ROOT="$PWD" OPENAI_BASE_URL="http://localhost:$2/v1" OPENAI_API_KEY=EMPTY RETRY_LOG="$PWD/retry-log.jsonl"
exec @ROOT@/v1/bfcl/.venv/bin/python run_bfcl_retry.py generate --model "$1" --test-category "$3" --num-threads 48
