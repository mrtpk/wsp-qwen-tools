# Run 2. Held-out probes, non-thinking, decision position only: cB (single-turn) and mt_B (multi-turn).
import json
V2 = "@ROOT@/v2/"
TC = tok.convert_tokens_to_ids("<tool_call>")
def args_map(a):
    while isinstance(a, str):
        try: a = json.loads(a)
        except Exception: return {"value": a}
    return a if isinstance(a, dict) else {"value": a}
def fix(ms):
    return [{**m, "tool_calls": [{**c, "function": {**c["function"], "arguments": args_map(c["function"]["arguments"])}} for c in m["tool_calls"]]} if m.get("tool_calls") else m for m in ms]
meta = {}
for name, path in (("cB", "cells/run2/cB.jsonl"), ("mt_B", "cells/run1/mt_B.jsonl")):
    R = [json.loads(l) for l in open(V2 + path)]
    texts, positions, info = [], [], []
    for r in R:
        t = tok.apply_chat_template(fix(r["messages"]), tools=r["tools"], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        texts.append(t); positions.append([len(tok(t, add_special_tokens=False)["input_ids"]) - 1])
        info.append({"id": r["id"], "label": r["label"], "group": r["detail"].get("group", r["label"])})
    json.dump({"type": "probe", "model": "original", "texts": texts, "positions": positions,
               "mlp_in": [23], "hidden": [16], "token_ids": [TC]}, open(f"req-probe-{name}-nothink.json", "w"))
    meta[name] = info; print(name, len(texts))
json.dump(meta, open("probe-meta-heldout.json", "w"))
