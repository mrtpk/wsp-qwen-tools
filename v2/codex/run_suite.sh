#!/usr/bin/env bash
# Usage: run_suite.sh <served-name> <port> <out-dir> [timeout-s]
# Runs every task once with codex exec in a container (image codex-eval:0.167.0), against
# http://localhost:<port>/v1. The container is the sandbox: it mounts only the work dir and CODEX_HOME.
# Writes <out-dir>/<task>/{work/, events.jsonl, stderr.log, codex_rc, result}.
set -u
C=$(cd "$(dirname "$0")" && pwd)
for t in $(ls "$C/tasks"); do
  d=$3/$t; mkdir -p "$d"; rm -rf "$d/work"; cp -r "$C/tasks/$t/repo" "$d/work"
  timeout "${4:-300}" docker run --rm --network host --user "$(id -u):$(id -g)" -e HOME=/tmp -e CODEX_HOME=/codexhome \
      -v "$d/work:/work" -v "$C/home:/codexhome" -w /work codex-eval:0.167.0 \
      codex exec --json -s danger-full-access --skip-git-repo-check \
      -c model="\"$1\"" -c model_providers.vllm.base_url="\"http://localhost:$2/v1\"" \
      "$(cat "$C/tasks/$t/prompt.txt")" < /dev/null > "$d/events.jsonl" 2> "$d/stderr.log"
  echo $? > "$d/codex_rc"
  ( cd "$d/work" && bash "$C/tasks/$t/check.sh" > ../check.log 2>&1 ) && echo pass > "$d/result" || echo fail > "$d/result"
  echo "$t $(cat "$d/result") rc=$(cat "$d/codex_rc")"
done
