#!/usr/bin/env bash
set -e
python3 -c "from add import add; assert add(2, 3) == 5 and add(-1, 1) == 0"
