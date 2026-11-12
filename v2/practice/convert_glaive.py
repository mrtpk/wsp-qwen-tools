"""Convert glaive-function-calling-v2 (revision e7f4b645) to the practice row format. No BFCL data.

Only the first user turn and the first assistant turn. Rows with no tools are dropped.
  label call:    the first assistant turn is a <functioncall>.
  label no_call: the first assistant turn is text although tools exist (a refusal or a question).
                 Dropped when the next assistant turn is a call (a preamble; label not clear).
Leak filter: as in convert.py (user message equal to a BFCL user message after normalization).

Usage: python -I convert_glaive.py <raw-json> <out-jsonl> <bfcl-user-messages.json>
"""
import json
import re
import sys
from pathlib import Path

raw, out = Path(sys.argv[1]), Path(sys.argv[2])
bfcl = set(json.loads(Path(sys.argv[3]).read_text()))


def norm(text):
    return re.sub(r"\W+", " ", text.lower()).strip()


def tools_of(system):
    text = system.split("Use them if required -", 1)[1] if "Use them if required -" in system else ""
    dec, i, specs = json.JSONDecoder(), 0, []
    while True:
        j = text.find("{", i)
        if j < 0:
            return specs
        spec, i = dec.raw_decode(text, j)
        params = dict(spec.get("parameters") or {"type": "object", "properties": {}})
        specs.append({"type": "function", "function": {"name": spec["name"], "description": spec.get("description", ""),
                                                      "parameters": params}})


rows, dropped, leaked = [], 0, 0
for k, x in enumerate(json.loads(raw.read_text())):
    m = re.match(r"USER: (.*?)\n\n\nASSISTANT: (.*?) <\|endoftext\|>", x["chat"], re.S)
    try:
        tools = tools_of(x["system"])
    except (json.JSONDecodeError, KeyError):
        dropped += 1
        continue
    if not m or not tools:
        dropped += 1
        continue
    user, answer = m.group(1).strip(), m.group(2).strip()
    if norm(user) in bfcl:
        leaked += 1
        continue
    if answer.startswith("<functioncall>"):
        try:
            call = json.loads(answer[len("<functioncall>"):].strip().replace("'{", "{").replace("}'", "}"))
        except json.JSONDecodeError:
            dropped += 1
            continue
        label, detail = "call", {"name": call["name"], "reference": json.dumps(call)[:500]}
    elif x["chat"][m.end():].lstrip().startswith("ASSISTANT: <functioncall>"):
        dropped += 1  # a text preamble directly before a call: the label is not clear
        continue
    else:
        label, detail = "no_call", {"reference": answer[:500]}
    rows.append({"id": f"glaive-{k}", "source": "glaive", "messages": [{"role": "user", "content": user}],
                 "tools": tools, "label": label, "detail": detail})
with out.open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("rows", len(rows), "call", sum(r["label"] == "call" for r in rows), "dropped", dropped, "leaked", leaked)
