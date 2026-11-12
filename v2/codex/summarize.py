"""Summarize Codex suite results. Usage: summarize.py <results-dir>  (e.g. codex/results/think)

Per model: task pass rate (over all tasks and repeats), passes per task, and agent-loop measures:
  cmds        shell commands per run (command_execution items)
  cmd_fail    share of shell commands with a non-zero exit code
  tool_err    tool calls Codex rejected per run (unknown tool, bad arguments, ...; from stderr)
  no_action   share of runs with no shell command and no file change
  timeout     share of runs stopped by the time limit (exit 124)
"""
import json, sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
for model in sorted(p.name for p in root.iterdir()):
    runs = sorted((root / model).glob("rep*/*/result"))
    per_task = defaultdict(list); cmds = fails = errs = noact = tout = 0
    for r in runs:
        d = r.parent; ok = r.read_text().strip() == "pass"; per_task[d.name].append(ok)
        items = [json.loads(l)["item"] for l in open(d / "events.jsonl") if '"item.completed"' in l]
        c = [i for i in items if i["type"] == "command_execution"]
        cmds += len(c); fails += sum(i.get("exit_code") not in (0, None) or i.get("status") == "failed" for i in c)
        errs += sum("ERROR codex_core::tools::router" in l for l in open(d / "stderr.log"))
        noact += not (c or any(i["type"] == "file_change" for i in items))
        tout += (d / "codex_rc").read_text().strip() == "124"
    n = len(runs)
    print(f"{model}: pass {sum(map(sum, per_task.values()))}/{n} = {sum(map(sum, per_task.values())) / n:.2f}"
          f" | cmds {cmds / n:.1f} cmd_fail {fails / max(cmds, 1):.2f} tool_err {errs / n:.1f}"
          f" no_action {noact / n:.2f} timeout {tout / n:.2f}")
    print("   " + " ".join(f"{t}={sum(v)}/{len(v)}" for t, v in sorted(per_task.items())))
