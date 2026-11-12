#!/usr/bin/env bash
# The scripts and ledgers hold the placeholders @ROOT@ (this repository) and @BASE@ (the base model). This script rewrites them in place.
# Usage: tools/relocate.sh BASE_MODEL_DIR
#   BASE_MODEL_DIR: a local snapshot of Qwen/Qwen3.5-0.8B at revision 2fc06364715b967f1860aea9cf38778875588b17.
# The new root is the folder that holds this tools/ folder. Run it once after you clone the repository.
set -euo pipefail
NEW_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="$(cd "$1" && pwd)"
OLD_ROOT=@ROOT@
OLD_BASE=@BASE@
cd "$NEW_ROOT"
grep -rIl --exclude-dir=results --exclude=relocate.sh -e "$OLD_ROOT" -e "$OLD_BASE" v1 v2 analysis tools 2>/dev/null \
  | while read -r f; do sed -i -e "s|$OLD_BASE|$BASE|g" -e "s|$OLD_ROOT|$NEW_ROOT|g" "$f"; echo "rewrote $f"; done
