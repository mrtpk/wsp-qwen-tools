# Run 3, cell 17. Nothink-mode probes on tA and mA at the decision position (prompt end, after the empty think block).
# Original model. Reads: layer-23 MLP input; logp of "<tool_call>". For neuron N (nothink, V2-D11).
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
for s in ("tA", "mA"):
    R = [json.loads(l) for l in open(V2 + f"cells/run3/{s}.jsonl")]
    texts = [tok.apply_chat_template(fix(r["messages"]), tools=r["tools"], add_generation_prompt=True, tokenize=False, enable_thinking=False) for r in R]
    pos = [[len(tok(t, add_special_tokens=False)["input_ids"]) - 1] for t in texts]
    json.dump({"type": "probe", "model": "original", "texts": texts, "positions": pos, "mlp_in": [23], "token_ids": [TC]},
              open(f"req-probe-{s}-nothink.json", "w"))
    meta[s] = [{"id": r["id"], "label": r["label"], "group": r["detail"].get("group", r["label"])} for r in R]
    print(s, len(texts), "ends with", repr(texts[0][-30:]), "max chars", max(map(len, texts)))
json.dump(meta, open("probe-meta-nothink.json", "w"))
