#!/usr/bin/env bash
set -e
! grep -q 3.14159 geometry.py && python3 -c "import math; from geometry import circle_area as a, circumference as c; assert a(1) == math.pi and c(1) == 2 * math.pi"
