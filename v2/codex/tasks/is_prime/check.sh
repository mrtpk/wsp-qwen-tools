#!/usr/bin/env bash
set -e
python3 -c "from mathx import is_prime, square; assert [n for n in range(30) if is_prime(n)] == [2,3,5,7,11,13,17,19,23,29] and square(3) == 9"
