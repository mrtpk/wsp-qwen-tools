#!/usr/bin/env bash
set -e
grep -q "def compute_total" utils.py && ! grep -qw foo utils.py main.py && [ "$(python3 main.py)" = "10" ]
