# Damage check texts: base generations (pass 23 st_B think, 24 st_B nothink, 25 mt_B think), full sequences.
import json
V2 = "@ROOT@/v2/"
passes = {p["pass"]: p["result"]["file"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
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
texts, starts, meta = [], [], []
for p, setname, think, n in [(23, "st_B", True, 96), (24, "st_B", False, 96), (25, "mt_B", True, 64)]:
    R = [json.loads(l) for l in open(V2 + f"cells/run1/{setname}.jsonl")]; out = json.load(open(passes[p]))["original"]
    for r, o in list(zip(R, out))[:n]:
        s = o["samples"][0]
        prompt = tok.apply_chat_template(fix(r["messages"]), tools=r["tools"], add_generation_prompt=True, tokenize=False, enable_thinking=think)
        gen = ((s["reasoning"] or "").strip("\n") + "\n</think>\n\n" if think else "") + (s["content"] or "").strip()
        if s["tool_calls"]: gen += ("\n\n" if (s["content"] or "").strip() else "") + "\n".join(xml_call(c) for c in s["tool_calls"])
        texts.append(prompt + gen + "<|im_end|>"); starts.append(len(tok(prompt, add_special_tokens=False)["input_ids"]))
        meta.append({"set": setname, "think": think, "called": bool(s["tool_calls"])})
for arm in ("original", "edited"):
    json.dump({"type": "teacher_forced", "model": arm, "texts": texts, "return": {"token_logprobs": True, "topk": 20}}, open(f"req-damage-{arm}.json", "w"))
json.dump({"starts": starts, "meta": meta}, open("damage-meta.json", "w"))
print(len(texts))
