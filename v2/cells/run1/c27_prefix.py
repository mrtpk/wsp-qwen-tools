import json, collections
passes = {p["pass"]: p["result"]["file"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
out = json.load(open(passes[23]))
for arm in ("original", "edited"):
    c = collections.Counter(); r_end = collections.Counter()
    for o in out[arm]:
        s = o["samples"][0]
        c[(bool(s["tool_calls"]), repr((s["content"] or "")[:4]))] += 1
        r_end[repr((s["reasoning"] or "")[-3:])] += 1
    print(arm, "content prefix (called, prefix):", c.most_common(8))
    print(arm, "reasoning ends:", r_end.most_common(6))
ids = tok("x\n</think>\n\n<tool_call>", add_special_tokens=False)["input_ids"]
print([tok.decode([i]) for i in ids])
