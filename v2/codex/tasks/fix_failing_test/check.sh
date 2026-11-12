#!/usr/bin/env bash
set -e
python3 test_app.py && python3 -c "from app import clamp; assert clamp(7, 0, 9) == 7"
