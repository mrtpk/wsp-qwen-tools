"""Convert the raw practice datasets to one OpenAI chat format. No BFCL data.

Output rows: {"id", "source", "messages", "tools", "label", "detail"}
  messages: the conversation up to (not including) the labelled assistant turn
  label:    "call" or "no_call" for that assistant turn
  detail:   source-specific note (When2Call subtype, called function name, step index)

Usage: python -I convert.py <raw-dir> <out-dir> <bfcl-user-messages.json>
"""
import json
import re
import sys
from pathlib import Path

raw, out = Path(sys.argv[1]), Path(sys.argv[2])
bfcl = set(json.loads(Path(sys.argv[3]).read_text()))


def norm(text):
    return re.sub(r"\W+", " ", text.lower()).strip()


def tool(spec):
    spec = json.loads(spec) if isinstance(spec, str) else spec
    params = dict(spec.get("parameters") or {"type": "object", "properties": {}})
    if params.get("type") == "dict":
        params["type"] = "object"
    return {"type": "function", "function": {"name": spec["name"], "description": spec.get("description", ""),
                                             "parameters": params}}


def leaks(messages):
    return any(m["role"] == "user" and norm(m["content"]) in bfcl for m in messages)


def write(name, rows):
    kept = [r for r in rows if not leaks(r["messages"])]
    with (out / f"{name}.jsonl").open("w") as handle:
        for row in kept:
            handle.write(json.dumps(row) + "\n")
    print(name, "rows", len(rows), "kept", len(kept), "dropped (BFCL user message)", len(rows) - len(kept))


# When2Call train_sft: every target turn is text (no call). Subtype from the opening words.
rows = []
for i, line in enumerate((raw / "When2Call__train_when2call_train_sft.jsonl").open()):
    x = json.loads(line)
    answer = x["messages"][-1]["content"]
    ask = answer.startswith(("To proceed", "To provide", "To assist", "Could you", "Please provide", "I need"))
    rows.append({"id": f"w2c-{i}", "source": "when2call", "messages": x["messages"][:-1],
                 "tools": [tool(t) for t in x["tools"]], "label": "no_call",
                 "detail": {"subtype": ("no_tools" if not x["tools"] else "ask" if ask else "cannot"),
                            "reference": answer}})
write("when2call", rows)

# ToolACE: tools in the system prompt; the first assistant turn is "[Name(args)]" or text.
rows = []
for i, x in enumerate(json.loads((raw / "ToolACE__data.json").read_text())):
    marker = "Here is a list of functions in JSON format that you can invoke:\n"
    if marker not in x["system"]:
        continue
    specs = json.JSONDecoder().raw_decode(x["system"].split(marker, 1)[1].strip())[0]
    turns = x["conversations"]
    first = next(j for j, t in enumerate(turns) if t["from"] == "assistant")
    if any(t["from"] != "user" for t in turns[:first]):
        continue
    answer = turns[first]["value"].strip()
    call = answer.startswith("[") and answer.endswith("]") and "(" in answer
    rows.append({"id": f"toolace-{i}", "source": "toolace",
                 "messages": [{"role": "user", "content": t["value"]} for t in turns[:first]],
                 "tools": [tool(s) for s in specs], "label": "call" if call else "no_call",
                 "detail": {"reference": answer[:500]}})
write("toolace", rows)

# APIGen-MT-5k: multi-turn agent conversations. One row per assistant step (call or text).
rows = []
for i, x in enumerate(json.loads((raw / "APIGen-MT-5k__apigen-mt_5k.json").read_text())):
    tools = [tool(s) for s in (json.loads(x["tools"]) if isinstance(x["tools"], str) else x["tools"])]
    messages, step, pending = [{"role": "system", "content": x["system"]}], 0, None
    for turn in x["conversations"]:
        kind, value = turn["from"], turn["value"]
        if kind == "human":
            messages.append({"role": "user", "content": value})
        elif kind in ("gpt", "function_call"):
            label = "call" if kind == "function_call" else "no_call"
            detail = {"step": step, "reference": value[:500]}
            if label == "call":
                detail["name"] = json.loads(value)["name"]
            rows.append({"id": f"apigen-{i}-{step}", "source": "apigen_mt", "messages": list(messages),
                         "tools": tools, "label": label, "detail": detail})
            step += 1
            if label == "call":
                call = json.loads(value)
                pending = f"call_{i}_{step}"
                messages.append({"role": "assistant", "content": "", "tool_calls": [
                    {"id": pending, "type": "function",
                     "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}]})
            else:
                messages.append({"role": "assistant", "content": value})
        elif kind == "observation":
            messages.append({"role": "tool", "tool_call_id": pending, "content": value})
write("apigen_mt", rows)
