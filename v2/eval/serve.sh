#!/usr/bin/env bash
# Usage: serve.sh <model-dir> <served-name> <port> <gpu-mem-fraction> [max-model-len] [seed]
set -euo pipefail
ROOT=@ROOT@
cd "$ROOT"
export CUDA_HOME="$ROOT/.venv/lib/python3.12/site-packages/nvidia/cu13"
export FLASHINFER_EXTRA_LDFLAGS="-L$ROOT/.cudart"
export PATH="$ROOT/.venv/bin:$PATH"
exec "$ROOT/.venv/bin/vllm" serve "$1" --served-model-name "$2" --port "$3" \
  --gpu-memory-utilization "$4" --max-model-len "${5:-65536}" --language-model-only \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --enable-prefix-caching --seed "${6:-0}"
