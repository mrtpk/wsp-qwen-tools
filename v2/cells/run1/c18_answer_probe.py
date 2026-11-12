# Probe request: answer-segment positions of the pass-16 (st_B think) outputs, original weights.
import json
V2 = "@ROOT@/v2/"
passes = {p["pass"]: p["result"]["file"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
R = [json.loads(l) for l in open(V2 + "cells/run1/st_B.jsonl")]; out = json.load(open(passes[16]))
TC = tok.convert_tokens_to_ids("<tool_call>")
def xml_call(call):
    try: args = json.loads(call["arguments"])
    except Exception: args = {}
    s = f"<tool_call>\n<function={call['name']}>\n"
    for k, v in args.items(): s += f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
    return s + "</function>\n</tool_call>"
texts, positions, info = [], [], []
for arm in ("edited", "original"):
    for i, (r, o) in enumerate(zip(R, out[arm])):
        s = o["samples"][0]
        content = (s["content"] or "")
        if not content.strip():
            continue                                   # no answer text: nothing to read
        prompt = tok.apply_chat_template(r["messages"], tools=r["tools"], add_generation_prompt=True, tokenize=False, enable_thinking=True)
        head = prompt + (s["reasoning"] or "").strip("\n") + "\n</think>\n\n"
        n0 = len(tok(head, add_special_tokens=False)["input_ids"])
        body = content.lstrip("\n")
        text = head + body + ("\n\n" + xml_call(s["tool_calls"][0]) if s["tool_calls"] else "")
        ids = tok(text, add_special_tokens=False)["input_ids"]
        end = ids.index(TC, n0) if s["tool_calls"] and TC in ids[n0:] else min(len(ids), n0 + 64)
        pos = [n0 - 1] + list(range(n0, end))[:64]
        texts.append(text); positions.append(pos)
        info.append({"arm": arm, "row": i, "label": r["label"], "group": r["detail"].get("subtype", r["label"]), "called": bool(s["tool_calls"]),
                     "n": len(pos)})
json.dump({"type": "probe", "model": "original", "texts": texts, "positions": positions, "mlp_in": [23], "token_ids": [TC]}, open("req-answer-probe.json", "w"))
json.dump(info, open("answer-probe-meta.json", "w"))
print(len(texts), sum(x["called"] for x in info), sum(x["n"] for x in info))
