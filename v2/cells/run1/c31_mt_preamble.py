import json
passes = {p["pass"]: p["result"]["file"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
for p, name in [(1, "st_A think"), (2, "st_A nothink"), (3, "mt_A think"), (4, "mt_A nothink"), (25, "mt_B think"), (26, "mt_B nothink")]:
    out = json.load(open(passes[p]))["original"]
    called = [o["samples"][0] for o in out if o["samples"][0]["tool_calls"]]
    pre = [s for s in called if (s["content"] or "").strip()]
    print(f"{name:13s} base: calls {len(called)}, calls after a text preamble {len(pre)} ({len(pre)/max(1,len(called)):.2f})")
