# Run 2. Rebuild the mt_B generation requests: the chat API needs tool-call arguments as a JSON string
# (c54 wrote a dict, which only the chat template accepts). Double-encoded arguments are decoded first.
import json
V2 = "@ROOT@/v2/"
def args_map(a):
    while isinstance(a, str):
        try: a = json.loads(a)
        except Exception: return {"value": a}
    return a if isinstance(a, dict) else {"value": a}
def fix(ms):
    return [{**m, "tool_calls": [{**c, "function": {**c["function"], "arguments": json.dumps(args_map(c["function"]["arguments"]))}} for c in m["tool_calls"]]} if m.get("tool_calls") else m for m in ms]
R = [json.loads(l) for l in open(V2 + "cells/run1/mt_B.jsonl")]
for think in (True, False):
    f = f"req-gen-mt_B-{'think' if think else 'nothink'}-v3.json"; r = json.load(open(f))
    r["items"] = [{"messages": fix(x["messages"]), "tools": x["tools"]} for x in R]; json.dump(r, open(f, "w")); print(f, len(r["items"]))
