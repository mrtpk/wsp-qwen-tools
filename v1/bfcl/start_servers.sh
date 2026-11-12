#!/usr/bin/env bash
# Start vLLM servers one after the other in tmux, each after the previous one answers.
# Usage: start_servers.sh <seed> <gpu-fraction> <model-dir>:<served-name>:<port> [...]
set -euo pipefail
seed="$1"
frac="$2"
shift 2
T=@ROOT@/v1/tmux_run.sh
B=@ROOT@/v1/bfcl
for spec in "$@"; do
  IFS=: read -r dir name port <<< "$spec"
  $T "srv-$name" 0 "cd $B && ./serve.sh $dir $name $port $frac 262144 $seed"
  until curl -sf "localhost:$port/v1/models" > /dev/null; do sleep 10; done
  echo "server $name up on $port"
done
