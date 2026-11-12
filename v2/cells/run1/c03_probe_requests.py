# Build probe requests: decision position + 8 background positions (+ up to 4 positions inside the
# base's own first tool call, when it called). Arguments of history tool calls become mappings (template).
import json, random
V2 = "@ROOT@/v2/"
passes = {p["pass"]: p["result"]["file"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
TC = tok.convert_tokens_to_ids("<tool_call>")
def args_map(a):
    while isinstance(a, str):
        try: a = json.loads(a)
        except Exception: return {"value": a}
    return a if isinstance(a, dict) else {"value": a}
def fix(messages):
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{**c, "function": {**c["function"], "arguments": args_map(c["function"]["arguments"])}} for c in m["tool_calls"]]
        out.append(m)
    return out
def xml_call(call):
    try: args = json.loads(call["arguments"])
    except Exception: args = {}
    s = f"<tool_call>\n<function={call['name']}>\n"
    for k, v in args.items():
        s += f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
    return s + "</function>\n</tool_call>"
rng = random.Random(1)
meta = {}
for p, setname, think in [(1, "st_A", True), (2, "st_A", False), (3, "mt_A", True), (4, "mt_A", False)]:
    R = [json.loads(l) for l in open(V2 + f"cells/run1/{setname}.jsonl")]
    out = json.load(open(passes[p]))["original"]
    texts, positions, info = [], [], []
    for r, o in zip(R, out):
        s = o["samples"][0]
        prompt = tok.apply_chat_template(fix(r["messages"]), tools=r["tools"], add_generation_prompt=True,
                                         tokenize=False, enable_thinking=think)
        text = prompt + ((s["reasoning"] or "").strip("\n") + "\n</think>\n\n" if think else "")
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        call_pos = []
        if s["tool_calls"]:
            full = text + xml_call(s["tool_calls"][0])
            m = len(tok(full, add_special_tokens=False)["input_ids"])
            call_pos = sorted(rng.sample(range(n, m), min(4, m - n)))
            text = full
        bg = sorted(rng.sample(range(0, n - 1), 8))
        texts.append(text); positions.append([n - 1] + bg + call_pos)
        info.append({"id": r["id"], "label": r["label"], "group": r["detail"].get("subtype", r["label"]),
                     "called": bool(s["tool_calls"]), "n_bg": 8, "n_call": len(call_pos)})
    req = {"type": "probe", "model": "original", "texts": texts, "positions": positions,
           "mlp_in": [23], "mlp_act": [23], "hidden": [23], "token_ids": [TC]}
    name = f"probe-{setname}-{'think' if think else 'nothink'}"
    json.dump(req, open(f"req-{name}.json", "w")); meta[name] = info
    print(name, len(texts), max(len(t) for t in texts))
json.dump(meta, open("probe-meta.json", "w"))
