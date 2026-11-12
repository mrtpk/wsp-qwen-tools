# Run 3, cell 2. Thinking-mode probes on tA and mA (base reasoning from passes 1, 2; original arm).
# Positions: the decision position (after "</think>\n\n"); and, when the base wrote text and then called,
# the position before that late "<tool_call>". Reads: layer-23 MLP input, residual before layer 16.
import json
V2 = "@ROOT@/v2/"
P = {p["pass"]: p["result"].get("file") for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
TC = tok.convert_tokens_to_ids("<tool_call>")
def args_map(a):
    while isinstance(a, str):
        try: a = json.loads(a)
        except Exception: return {"value": a}
    return a if isinstance(a, dict) else {"value": a}
def fix(ms):
    return [{**m, "tool_calls": [{**c, "function": {**c["function"], "arguments": args_map(c["function"]["arguments"])}} for c in m["tool_calls"]]} if m.get("tool_calls") else m for m in ms]
meta = {}
for p, s in ((1, "tA"), (2, "mA")):
    R = [json.loads(l) for l in open(V2 + f"cells/run3/{s}.jsonl")]; out = json.load(open(P[p]))["original"]
    texts, pos, info = [], [], []
    for r, o in zip(R, out):
        smp = o["samples"][0]
        t = tok.apply_chat_template(fix(r["messages"]), tools=r["tools"], add_generation_prompt=True, tokenize=False, enable_thinking=True)
        t += (smp["reasoning"] or "").strip("\n") + "\n</think>\n\n"
        n = len(tok(t, add_special_tokens=False)["input_ids"]); ps = [n - 1]; late = False
        content = (smp["content"] or "").strip()
        if smp["tool_calls"] and content:
            t2 = t + content + "\n\n"; ps.append(len(tok(t2, add_special_tokens=False)["input_ids"]) - 1); t = t2; late = True
        texts.append(t); pos.append(ps)
        info.append({"id": r["id"], "label": r["label"], "group": r["detail"].get("group", r["label"]),
                     "called": bool(smp["tool_calls"]), "late": late})
    json.dump({"type": "probe", "model": "original", "texts": texts, "positions": pos, "mlp_in": [23], "hidden": [16], "token_ids": [TC]},
              open(f"req-probe-{s}-think.json", "w"))
    meta[s] = info; print(s, len(texts), "late-call rows", sum(i["late"] for i in info), "max chars", max(map(len, texts)))
json.dump(meta, open("probe-meta-think.json", "w"))
