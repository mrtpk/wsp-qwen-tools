#!/usr/bin/env bash
# Run a command in a new window of tmux session "wsp". The command is written to
# v1/tmuxlogs/<name>.sh and its output goes to v1/tmuxlogs/<name>.log.
# Usage: tmux_run.sh <name> <wait-seconds> <command>
#   wait-seconds = 0: start the command and return at once (long jobs).
#   wait-seconds > 0: wait up to that time for the end marker, then print the log.
set -euo pipefail
name="$1"
wait_s="$2"
cmd="$3"
dir="$(cd "$(dirname "$0")" && pwd)/tmuxlogs"
mkdir -p "$dir"
log="$dir/$name.log"
script="$dir/$name.sh"
printf '%s\n' "$cmd" > "$script"
: > "$log"
tmux has-session -t wsp 2>/dev/null || tmux new-session -d -s wsp -n idle
tmux new-window -d -t wsp -n "$name" "bash $script > $log 2>&1; echo __EXIT__ \$? >> $log"
if [ "$wait_s" -gt 0 ]; then
  for _ in $(seq "$wait_s"); do
    grep -q "^__EXIT__" "$log" && break
    sleep 1
  done
  cat "$log"
fi
