#!/usr/bin/env bash
# Fetch OpenWSP (MIT) at the commit that the v2 study used, into v2/openwsp.
# Upstream: https://github.com/openwsp/openwsp, commit 8d09276736034f6874bb40e7019247c9541a5551.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/v2/openwsp"
if [ ! -d "$DEST/.git" ]; then git clone https://github.com/openwsp/openwsp "$DEST"; fi
git -C "$DEST" checkout --detach 8d09276736034f6874bb40e7019247c9541a5551
