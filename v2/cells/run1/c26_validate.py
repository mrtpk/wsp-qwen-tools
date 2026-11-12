import json
V2 = "@ROOT@/v2/"
passes = {p["pass"]: p["result"]["file"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
def group(r): return r["detail"]["subtype"] if r["source"] == "when2call" else r["label"]
def ref_name(r): return r["detail"].get("name") or r["detail"]["reference"].strip("[").split("(")[0]
for p, s, mode in [(23, "st_B", "think"), (24, "st_B", "nothink"), (25, "mt_B", "think"), (26, "mt_B", "nothink")]:
    R = [json.loads(l) for l in open(V2 + f"cells/run1/{s}.jsonl")]; out = json.load(open(passes[p]))
    line = []
    for arm in ("original", "edited"):
        st = {}; correct = 0
        for r, o in zip(R, out[arm]):
            smp = o["samples"][0]; k = group(r); c = bool(smp["tool_calls"])
            a = st.setdefault(k, [0, 0, 0]); a[0] += 1; a[1] += c
            if r["label"] == "call" and c: a[2] += smp["tool_calls"][0]["name"] == ref_name(r)
            correct += (c == (r["label"] == "call"))
        line.append(f"{arm}: " + ", ".join(f"{k} call {v[1]/v[0]:.2f}" + (f" name {v[2]/max(1,v[1]):.2f}" if k == "call" else "") for k, v in st.items())
                    + f" | decision acc {correct/len(R):.3f}")
    print(s, mode); [print("   ", l) for l in line]
