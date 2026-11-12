"""Write the BFCL single-turn prompts and the base non-thinking outcomes to JSONL.

Run with the BFCL venv: v1/bfcl/.venv/bin/python dump_prompts.py <model> <out.jsonl>
"""

import json
import os
import sys
from pathlib import Path

BFCL = Path(__file__).resolve().parents[1] / "bfcl"
os.environ.setdefault("BFCL_PROJECT_ROOT", str(BFCL))

from bfcl_eval.constants.enums import ModelStyle  # noqa: E402
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI  # noqa: E402
from bfcl_eval.model_handler.utils import convert_to_tool  # noqa: E402
from bfcl_eval.utils import load_dataset_entry  # noqa: E402

CALL_CATS = ["simple_python", "simple_java", "simple_javascript", "multiple", "parallel",
             "parallel_multiple", "live_simple", "live_multiple", "live_parallel",
             "live_parallel_multiple"]
NO_CALL_CATS = ["irrelevance", "live_irrelevance"]


def read_jsonl(path):
    return [json.loads(line) for line in open(path)]


def main(model, out):
    rows = []
    for cat in CALL_CATS + NO_CALL_CATS:
        results = {r["id"]: r for r in read_jsonl(next((BFCL / "result" / model).rglob(f"BFCL_v4_{cat}_result.json")))}
        score = read_jsonl(next((BFCL / "score" / model).rglob(f"BFCL_v4_{cat}_score.json")))
        failed = {r["id"] for r in score[1:]}
        for entry in load_dataset_entry(cat):
            res = results[entry["id"]]
            called = isinstance(res["result"], list) and len(res["result"]) > 0
            rows.append({
                "id": entry["id"],
                "category": cat,
                "needs_call": cat in CALL_CATS,
                "messages": entry["question"][0],
                "tools": convert_to_tool(entry["function"], GORILLA_TO_OPENAPI,
                                         ModelStyle.OPENAI_COMPLETIONS),
                "called": called,
                "passed": entry["id"] not in failed,
                "input_token_count": res["input_token_count"],
                "output_token_count": res["output_token_count"],
                "reasoning_content": res.get("reasoning_content"),
                "result": res["result"],
            })
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
