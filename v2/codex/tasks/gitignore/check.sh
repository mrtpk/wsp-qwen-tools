#!/usr/bin/env bash
set -e
grep -q "__pycache__" .gitignore && grep -q "\*\.log" .gitignore
