#!/usr/bin/env bash
set -e
python3 -c "import subprocess; assert abs(float(subprocess.check_output(['python3', 'total.py']).decode().strip()) - 20.0) < 1e-9"
