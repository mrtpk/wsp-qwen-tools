import json
passes = {p["pass"]: p["result"]["file"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
out = json.load(open(passes[23]))["edited"]
k = 0
for o in out:
    s = o["samples"][0]
    if s["tool_calls"] and (s["content"] or "").strip():
        print(repr(s["content"][-160:]), "| reasoning end:", repr((s["reasoning"] or "")[-120:]))
        k += 1
        if k == 5: break
