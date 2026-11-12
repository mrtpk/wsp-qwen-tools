#!/usr/bin/env bash
# Unpack the saved BFCL-V4 generations and score them again (CPU only, no model needed).
# Needs: the release assets in release/ (release/ASSETS.md), the BFCL venv at v1/bfcl/.venv (env/requirements-bfcl.txt), zstd.
# Run from the repository root after tools/relocate.sh. Output: v1/bfcl/score, v2/eval/score.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/v1/bfcl/.venv/bin/python"
CATS=single_turn,multi_turn,memory
export OPENAI_API_KEY=EMPTY
zstd -dc "$ROOT/release/v1-bfcl-result.tar.zst" | tar -xf - -C "$ROOT/v1/bfcl"
zstd -dc "$ROOT/release/v2-bfcl-result.tar.zst" | tar -xf - -C "$ROOT/v2/eval"
names() { ls "$1" | tr '\n' ',' | sed 's/,$//'; }
served() { ls "$1" | sed -E 's/-(think|nothink)-FC$//' | sort -u | grep -v -x -e qwen35-base -e qwen35-wsp -e qwen35-base-retry | tr '\n' ',' | sed 's/,$//'; }
cd "$ROOT/v1/bfcl"
BFCL_PROJECT_ROOT="$PWD" QWEN35_EXTRA_MODELS="$(served result)" "$PY" run_bfcl.py evaluate --model "$(names result)" --test-category "$CATS"
cd "$ROOT/v2/eval"
BFCL_PROJECT_ROOT="$PWD" RETRY_LOG="$PWD/retry-log.jsonl" QWEN35_EXTRA_MODELS="$(served result)" "$PY" run_bfcl_retry.py evaluate --model "$(names result)" --test-category "$CATS"
echo "Done. Tables: python3 analysis/report_tables.py v1/bfcl/score v2/eval/score"
