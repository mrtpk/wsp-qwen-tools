# Run 2. Probe requests on the clean set cA: decision position + 8 background positions.
# Thinking: prompt + the base's own reasoning from pass 30. Non-thinking: the prompt only.
# Reads: residual stream before layers 8, 12, 16, 20 and the layer-23 MLP input.
import json, random
V2 = "@ROOT@/v2/"
passes = {p["pass"]: p["result"].get("file") for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
TC = tok.convert_tokens_to_ids("<tool_call>")
R = [json.loads(l) for l in open(V2 + "cells/run2/cA.jsonl")]
gen = json.load(open(passes[30]))["original"]
rng = random.Random(3)
meta = {}
for think in (True, False):
    texts, positions, info = [], [], []
    for r, o in zip(R, gen):
        s = o["samples"][0]
        text = tok.apply_chat_template(r["messages"], tools=r["tools"], add_generation_prompt=True,
                                       tokenize=False, enable_thinking=think)
        if think:
            text += (s["reasoning"] or "").strip("\n") + "\n</think>\n\n"
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        texts.append(text); positions.append([n - 1] + sorted(rng.sample(range(0, n - 1), 8)))
        info.append({"id": r["id"], "label": r["label"], "group": r["detail"]["group"],
                     "called": bool(s["tool_calls"]) if think else None})
    req = {"type": "probe", "model": "original", "texts": texts, "positions": positions,
           "mlp_in": [23], "hidden": [8, 12, 16, 20], "token_ids": [TC]}
    name = f"probe-cA-{'think' if think else 'nothink'}"
    json.dump(req, open(f"req-{name}.json", "w")); meta[name] = info
    print(name, len(texts), max(len(t) for t in texts))
json.dump(meta, open("probe-meta-r2.json", "w"))
