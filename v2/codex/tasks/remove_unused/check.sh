#!/usr/bin/env bash
set -e
[ ! -e old_utils.py ] && [ "$(python3 main.py)" = "run" ]
