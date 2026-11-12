#!/usr/bin/env bash
set -e
python3 -c "from strings import reverse_words as r; assert r('x y z') == 'z y x'" && python3 test_strings.py
