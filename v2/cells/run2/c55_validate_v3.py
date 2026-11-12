# Run 2. Validation of edit v3 by generation (passes 39 to 43; 41 = gT, 42 and 43 = mt_B, both arms, seed 0, held-out sets).
# Decision accuracy = the sampled answer calls a tool exactly when the label is "call".
# For call rows: the share whose first call uses the reference function name.
import json
V2 = "@ROOT@/v2/"
passes = {p["pass"]: p["result"].get("file") for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
def group(r): return r["detail"].get("group", r["label"])
def ref_name(r):
    if r["source"] == "toolace": return r["detail"]["reference"].strip("[").split("(")[0]
    try: return json.loads(r["detail"]["reference"])["name"]
    except Exception: return r["detail"].get("name")
for p, s, path, mode in [(39, "cB", "cells/run2/cB.jsonl", "nothink"), (40, "cB", "cells/run2/cB.jsonl", "think"),
                         (41, "gT", "cells/run2/gT.jsonl", "nothink"),
                         (42, "mt_B", "cells/run1/mt_B.jsonl", "nothink"), (43, "mt_B", "cells/run1/mt_B.jsonl", "think")]:
    if p not in passes: continue
    R = [json.loads(l) for l in open(V2 + path)]; out = json.load(open(passes[p]))
    if len(R) == 320: R = [R[i] for i in json.load(open("gen-v3-keep.json"))]
    print(f"{s} {mode} (pass {p})")
    for arm in ("original", "edited"):
        st = {}; correct = 0
        for r, o in zip(R, out[arm]):
            smp = o["samples"][0]; k = group(r); c = bool(smp["tool_calls"])
            a = st.setdefault(k, [0, 0, 0]); a[0] += 1; a[1] += c
            if r["label"] == "call" and c: a[2] += smp["tool_calls"][0]["name"] == ref_name(r)
            correct += (c == (r["label"] == "call"))
        print(f"   {arm:8s} decision acc {correct / len(R):.3f} | " + ", ".join(
            f"{k} call {v[1] / v[0]:.2f}" + (f" (name ok {v[2] / max(1, v[1]):.2f})" if k == "call" else "") for k, v in st.items()))
