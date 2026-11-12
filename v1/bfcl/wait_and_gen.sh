#!/usr/bin/env bash
# Usage: wait_and_gen.sh <port> <registry-name> <categories>
set -euo pipefail
cd "$(dirname "$0")"
until curl -sf "localhost:$1/v1/models" > /dev/null; do sleep 10; done
exec ./gen.sh "$2" "$1" "$3"
