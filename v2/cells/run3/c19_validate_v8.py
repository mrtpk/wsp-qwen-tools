# Run 3, cell 19. Validation of edit v8 by generation (passes 21 to 23, both arms = base vs v8, seed 0, held-out).
import json
V2 = "@ROOT@/v2/"
P = {p["pass"]: p["result"].get("file") for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
def ref_name(r):
    if r["source"] == "toolace": return r["detail"]["reference"].strip("[").split("(")[0]
    try: return json.loads(r["detail"]["reference"])["name"]
    except Exception: return r["detail"].get("name")
for p, s, mode in ((21, "tB", "nothink"), (22, "mB", "nothink"), (23, "gB", "nothink")):
    R = [json.loads(l) for l in open(V2 + f"cells/run3/{s}.jsonl")]; out = json.load(open(P[p])); print(f"{s} {mode} (pass {p})")
    for arm in ("original", "edited"):
        st = {}; ok = pre = stray = name_ok = ncall = 0
        for r, o in zip(R, out[arm]):
            smp = o["samples"][0]; c = bool(smp["tool_calls"]); g = r["detail"]["group"]
            a = st.setdefault(g, [0, 0]); a[0] += 1; a[1] += c; ok += c == (r["label"] == "call")
            pre += c and bool((smp["content"] or "").strip()); stray += "<tool_call>" in (smp["content"] or "")
            if c and r["label"] == "call": ncall += 1; name_ok += smp["tool_calls"][0]["name"] == ref_name(r)
        print(f"   {arm:8s} decision acc {ok / len(R):.3f} | " + ", ".join(f"{k} call {v[1] / v[0]:.2f}" for k, v in st.items())
              + f" | name ok {name_ok / max(1, ncall):.2f} | preamble calls {pre} | stray {stray}")
