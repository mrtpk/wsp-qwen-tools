#!/usr/bin/env bash
set -e
[ "$(python3 greet.py)" = "Hello, World" ] && [ "$(python3 greet.py --name Ann)" = "Hello, Ann" ]
