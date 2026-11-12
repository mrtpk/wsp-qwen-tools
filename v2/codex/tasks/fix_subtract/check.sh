#!/usr/bin/env bash
set -e
python3 -c "from calc import add, subtract; assert subtract(5, 3) == 2 and add(1, 2) == 3"
