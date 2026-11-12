# Run 2, step 2 (multi-turn). Teacher-forced gold call steps from APIGen-MT (not in mt_A or mt_B),
# non-thinking. Text = prompt + gold call. Later: per-token top-1 agreement by span kind.
import json, random
V2 = "@ROOT@/v2/"
used = {json.loads(l)["id"] for s in ("mt_A", "mt_B") for l in open(V2 + f"cells/run1/{s}.jsonl")}
rows = [r for r in map(json.loads, open(V2 + "practice/apigen_mt.jsonl"))
        if r["label"] == "call" and r["detail"]["step"] >= 1 and r["id"] not in used]
random.Random(4).shuffle(rows)
def args_map(a):
    while isinstance(a, str):
        try: a = json.loads(a)
        except Exception: return {"value": a}
    return a if isinstance(a, dict) else {"value": a}
def fix(ms):
    return [{**m, "tool_calls": [{**c, "function": {**c["function"], "arguments": args_map(c["function"]["arguments"])}} for c in m["tool_calls"]]} if m.get("tool_calls") else m for m in ms]
texts, spans, meta = [], [], []
total = 0  # openwsp limit: 524,288 tokens per teacher-forced pass
for r in rows:
    if len(texts) == 256: break
    prompt = tok.apply_chat_template(fix(r["messages"]), tools=r["tools"], add_generation_prompt=True,
                                     tokenize=False, enable_thinking=False)
    try: call = json.loads(r["detail"]["reference"])
    except json.JSONDecodeError: continue  # reference cut short in convert.py
    parts = [("decision", "<tool_call>"), ("fixed", "\n<function="), ("fname", call["name"]), ("fixed", ">\n")]
    for k, v in args_map(call["arguments"]).items():
        parts += [("fixed", "<parameter="), ("pname", k), ("fixed", ">\n"),
                  ("value", v if isinstance(v, str) else json.dumps(v)), ("fixed", "\n</parameter>\n")]
    parts += [("fixed", "</function>\n</tool_call>")]
    n0 = len(tok(prompt, add_special_tokens=False)["input_ids"])
    if n0 > 12000: continue
    text, sp = prompt, []
    for kind, s in parts:
        a = len(tok(text, add_special_tokens=False)["input_ids"]); text += s
        sp.append([kind, a, len(tok(text, add_special_tokens=False)["input_ids"])])
    if total + sp[-1][2] > 500000: break
    total += sp[-1][2]
    texts.append(text); spans.append(sp); meta.append({"id": r["id"], "step": r["detail"]["step"], "n_tools": len(r["tools"])})
json.dump({"type": "teacher_forced", "model": "original", "texts": texts, "return": {"token_logprobs": True, "topk": 20}},
          open("req-mt-tf.json", "w"))
json.dump({"spans": spans, "meta": meta}, open("mt-tf-meta.json", "w"))
print(len(texts), "rows,", total, "tokens; example spans", spans[0][:6])
