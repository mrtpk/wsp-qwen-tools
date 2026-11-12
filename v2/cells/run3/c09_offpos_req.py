# Run 3, cell 9. Off-decision check. Texts: tB rows with the base's own full answer (passes 6 think, 8 nothink,
# original arm), call rows only. Positions: decision, 6 positions inside the tool call (function name and
# arguments), 6 random positions in the reasoning or prompt. Read: layer-23 MLP input.
import json, random
V2 = "@ROOT@/v2/"
P = {p["pass"]: p["result"].get("file") for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
TC = tok.convert_tokens_to_ids("<tool_call>"); rng = random.Random(9)
def xml_call(call):
    try: args = json.loads(call["arguments"])
    except Exception: args = {}
    s = f"<tool_call>\n<function={call['name']}>\n"
    for k, v in args.items(): s += f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
    return s + "</function>\n</tool_call>"
R = [json.loads(l) for l in open(V2 + "cells/run3/tB.jsonl")]
texts, pos, info = [], [], []
for p, think in ((6, True), (8, False)):
    out = json.load(open(P[p]))["original"]
    for i, (r, o) in enumerate(zip(R, out)):
        smp = o["samples"][0]
        if not smp["tool_calls"] or (smp["content"] or "").strip(): continue
        t = tok.apply_chat_template(r["messages"], tools=r["tools"], add_generation_prompt=True, tokenize=False, enable_thinking=think)
        n_prompt = len(tok(t, add_special_tokens=False)["input_ids"])
        if think: t += (smp["reasoning"] or "").strip("\n") + "\n</think>\n\n"
        n = len(tok(t, add_special_tokens=False)["input_ids"]); t2 = t + xml_call(smp["tool_calls"][0])
        m = len(tok(t2, add_special_tokens=False)["input_ids"])
        inside = sorted(rng.sample(range(n, m - 1), min(6, m - 1 - n)))
        lo_ = n_prompt if think and n - n_prompt > 10 else 0
        other = sorted(rng.sample(range(lo_, n - 1), 6))
        texts.append(t2); pos.append([n - 1] + inside + other)
        info.append({"think": think, "n_inside": len(inside), "row": i})
        if sum(1 for x in info if x["think"] == think) >= 96: break
json.dump({"type": "probe", "model": "original", "texts": texts, "positions": pos, "mlp_in": [23], "token_ids": [TC]}, open("req-probe-offpos.json", "w"))
json.dump(info, open("probe-meta-offpos.json", "w")); print(len(texts), "texts;", sum(i["think"] for i in info), "think")
