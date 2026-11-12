# Run 2, step 2. Multi-turn decision on mt_A (APIGen-MT only: call step vs text step), non-thinking.
# Probe at the decision position (end of prompt): residual before layers 8, 12, 16, 20 and layer-23 MLP input.
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
R = [json.loads(l) for l in open(V2 + "cells/run1/mt_A.jsonl")]
texts, positions, info, total = [], [], [], 0
for r in R:
    t = tok.apply_chat_template(fix(r["messages"]), tools=r["tools"], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    n = len(tok(t, add_special_tokens=False)["input_ids"])
    texts.append(t); positions.append([n - 1]); info.append({"id": r["id"], "label": r["label"]}); total += n
json.dump({"type": "probe", "model": "original", "texts": texts, "positions": positions,
           "mlp_in": [23], "hidden": [8, 12, 16, 20], "token_ids": [TC]}, open("req-probe-mtA-nothink-layers.json", "w"))
json.dump(info, open("probe-meta-mtA-layers.json", "w"))
print(len(texts), "texts,", total, "tokens,", sum(i["label"] == "call" for i in info), "call")
