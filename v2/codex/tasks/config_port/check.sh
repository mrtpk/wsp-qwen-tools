#!/usr/bin/env bash
set -e
python3 -c "import json; c = json.load(open('config.json')); assert c == {'host': '0.0.0.0', 'port': 9000, 'debug': False}"
