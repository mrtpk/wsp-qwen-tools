"""Editor entry point: the openwsp SDK with the local tool-use backend.

Usage (from v2, through ./wsp):
  wsp run CELL.py         run a kernel cell (counts against the code-run budget)
  wsp forward REQ.json    one forward request (counts against the pass budget)
  wsp diff                submitted edits against the base
  wsp export DIR          write the versioned edit (manifest.json, edit.npz)
  wsp status              passes, code runs, edited tensors

Every cell and its output are appended to <work>/kernel-log.jsonl.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from openwsp.sdk import OpenWSP
from tooluse_backend import create

CODE_RUN_BUDGET = int(os.environ.get("OPENWSP_CODE_RUNS", "300"))


def main():
    base, work = Path(os.environ["OPENWSP_BASE"]), Path(os.environ["OPENWSP_WORK"])
    practice = Path(os.environ["OPENWSP_PRACTICE"])
    command, args = sys.argv[1], sys.argv[2:]
    tokenizer = None
    if command == "run":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base)
    backend = create() if command == "forward" else None
    api = OpenWSP(base, work_dir=work, practice_dir=practice, backend=backend, tokenizer=tokenizer)
    log = work / "kernel-log.jsonl"
    runs = sum(1 for line in log.read_text().splitlines() if json.loads(line).get("counted")) if log.exists() else 0
    if command == "run":
        code = Path(args[0]).read_text()
        if runs >= CODE_RUN_BUDGET:
            print(json.dumps({"ok": False, "error": f"Code-run budget exhausted ({runs})"}))
            sys.exit(1)
        result = api.run_code(code)
        counted = result.get("runs", 0) > 0
        with log.open("a") as handle:
            handle.write(json.dumps({"time": time.time(), "cell": args[0], "code": code, "counted": counted,
                                     "run": runs + counted, "ok": result["ok"], "output": result["output"],
                                     "error": result["error"]}) + "\n")
        print(result["output"], end="")
        if result["error"]:
            print("ERROR:", result["error"])
        print(f"[code run {runs + counted}/{CODE_RUN_BUDGET}]")
    elif command == "forward":
        request = json.loads(Path(args[0]).read_text())
        print(json.dumps(api.forward(request), indent=1))
    elif command == "diff":
        print(json.dumps(api.diff(), indent=1))
    elif command == "export":
        print(json.dumps(api.export_edit(args[0], base_model="Qwen/Qwen3.5-0.8B",
                                         revision="2fc06364715b967f1860aea9cf38778875588b17"), indent=1))
    elif command == "status":
        passes = backend_passes(work)
        print(json.dumps({"passes": passes, "code_runs": runs, "edited": api.store.edited()}, indent=1))
    else:
        raise SystemExit(f"unknown command {command}")


def backend_passes(work):
    path = work / "passes.jsonl"
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if json.loads(line)["status"] == "done")


if __name__ == "__main__":
    main()
