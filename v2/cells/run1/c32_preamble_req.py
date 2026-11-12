# Natural preamble-then-call positions in base outputs (negatives for the veto gate).
import json, random
V2 = "@ROOT@/v2/"
passes = {p["pass"]: p["result"]["file"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
TC = tok.convert_tokens_to_ids("<tool_call>")
def args_map(a):
    while isinstance(a, str):
        try: a = json.loads(a)
        except Exception: return {"value": a}
    return a if isinstance(a, dict) else {"value": a}
def fix(ms):
    return [{**m, "tool_calls": [{**c, "function": {**c["function"], "arguments": args_map(c["function"]["arguments"])}} for c in m["tool_calls"]]} if m.get("tool_calls") else m for m in ms]
def xml_call(call):
    a = args_map(call["arguments"]); s = f"<tool_call>\n<function={call['name']}>\n"
    for k, v in a.items(): s += f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
    return s + "</function>\n</tool_call>"
texts, positions, meta = [], [], []
rng = random.Random(3)
for p, setname, think in [(1, "st_A", True), (2, "st_A", False), (4, "mt_A", False), (25, "mt_B", True), (26, "mt_B", False), (3, "mt_A", True)]:
    R = [json.loads(l) for l in open(V2 + f"cells/run1/{setname}.jsonl")]; out = json.load(open(passes[p]))["original"]
    for r, o in zip(R, out):
        s = o["samples"][0]
        if not (s["tool_calls"] and (s["content"] or "").strip()): continue
        prompt = tok.apply_chat_template(fix(r["messages"]), tools=r["tools"], add_generation_prompt=True, tokenize=False, enable_thinking=think)
        head = prompt + ((s["reasoning"] or "").strip("\n") + "\n</think>\n\n" if think else "")
        n0 = len(tok(head, add_special_tokens=False)["input_ids"])
        text = head + s["content"].strip() + "\n\n" + xml_call(s["tool_calls"][0])
        ids = tok(text, add_special_tokens=False)["input_ids"]
        k = [j for j, t in enumerate(ids) if t == TC and j >= n0][0]
        mid = sorted(rng.sample(range(n0, k - 1), min(6, k - 1 - n0))) if k - 1 > n0 else []
        texts.append(text); positions.append([k - 1] + mid)
        meta.append({"set": setname, "think": think, "label": r["label"]})
json.dump({"type": "probe", "model": "original", "texts": texts, "positions": positions, "mlp_in": [23], "token_ids": [TC]}, open("req-preamble.json", "w"))
json.dump(meta, open("preamble-meta.json", "w"))
print(len(texts), {s: sum(m["set"] + str(m["think"]) == s for m in meta) for s in set(m["set"] + str(m["think"]) for m in meta)})
