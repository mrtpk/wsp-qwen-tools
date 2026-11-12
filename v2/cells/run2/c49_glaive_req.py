# Run 2. Probe request for the third-source test set gT (glaive), non-thinking, decision position.
import json
V2 = "@ROOT@/v2/"
TC = tok.convert_tokens_to_ids("<tool_call>")
R = [json.loads(l) for l in open(V2 + "cells/run2/gT.jsonl")]
texts, positions, info = [], [], []
for r in R:
    t = tok.apply_chat_template(r["messages"], tools=r["tools"], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    texts.append(t); positions.append([len(tok(t, add_special_tokens=False)["input_ids"]) - 1])
    info.append({"id": r["id"], "label": r["label"], "group": r["detail"]["group"]})
json.dump({"type": "probe", "model": "original", "texts": texts, "positions": positions,
           "mlp_in": [23], "hidden": [16], "token_ids": [TC]}, open("req-probe-gT-nothink.json", "w"))
json.dump(info, open("probe-meta-gT.json", "w")); print(len(texts))
