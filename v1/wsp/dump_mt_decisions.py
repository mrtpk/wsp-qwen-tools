"""Rebuild the step-level decision points of BFCL multi-turn traces (thinking mode).

Run with the BFCL venv: ../bfcl/.venv/bin/python dump_mt_decisions.py <model> <category> <out.jsonl>
Each output row is one assistant step: the messages before it (as the handler sent them: no
reasoning in the history), the tools, the reasoning of this step, the label (called or stopped),
the turn and step index, and the vLLM input token count for the check.
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


def parse_args(args):
    """The Qwen3.5 template needs the arguments as a mapping. vLLM parses the JSON string the same way."""
    try:
        v = json.loads(args)
    except (json.JSONDecodeError, TypeError):
        return {}
    return v if isinstance(v, dict) else {}


def tool_calls(content, turn, step):
    return [{"id": f"call_{turn}_{step}_{i}", "type": "function",
             "function": {"name": name, "arguments": parse_args(args)}}
            for i, call in enumerate(content) for name, args in call.items()]


def main(model, category, out):
    entries = {e["id"]: e for e in load_dataset_entry(category)}
    res_file = next((BFCL / "result" / model).rglob(f"BFCL_v4_{category}_result.json"))
    rows = []
    for line in open(res_file):
        r = json.loads(line)
        e = entries[r["id"]]
        tools = convert_to_tool(e["function"], GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)
        # vLLM parses each tool into its request model, which keeps only these fields (it drops "response").
        tools = [{"type": "function", "function": {k: t["function"][k] for k in ("name", "description", "parameters")
                                                   if k in t["function"]}} for t in tools]
        messages = []
        turns = [x for x in r["inference_log"] if isinstance(x, dict) and "begin_of_turn_query" in x]
        assert len(turns) == len(r["result"]), (r["id"], len(turns), len(r["result"]))
        for t, turn in enumerate(turns):
            messages = messages + turn["begin_of_turn_query"]
            s = 0
            while f"step_{s}" in turn:
                items = turn[f"step_{s}"]
                asst = next(x for x in items if x["role"] == "assistant")
                called = isinstance(asst["content"], list) and len(asst["content"]) > 0
                rows.append({"id": r["id"], "turn": t, "step": s, "messages": messages, "tools": tools,
                             "reasoning_content": asst.get("reasoning_content") or "", "called": called,
                             "input_token_count": r["input_token_count"][t][s]})
                if called:
                    messages = messages + [{"role": "assistant", "content": "",
                                            "tool_calls": tool_calls(asst["content"], t, s)}]
                    tool_msgs = [x for x in items if x["role"] == "tool"]
                    ids = [c["id"] for c in messages[-1]["tool_calls"]]
                    messages = messages + [{"role": "tool", "content": x["content"], "tool_call_id": ids[i]}
                                           for i, x in enumerate(tool_msgs)]
                else:
                    messages = messages + [{"role": "assistant", "content": asst["content"]}]
                s += 1
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} decision rows from {res_file.name}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
