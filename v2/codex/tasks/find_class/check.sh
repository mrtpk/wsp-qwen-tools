#!/usr/bin/env bash
set -e
[ "$(tr -d "[:space:]" < answer.txt | sed "s#^\./##")" = "pkg/c.py" ]
