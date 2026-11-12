# Run 2. Validation of edit v4 by generation (passes 45 to 47, both arms, seed 0, held-out).
# Also: calls after a text preamble, and "<tool_call>" left in the text (a stray, unparsed call).
import json
V2 = "@ROOT@/v2/"
passes = {p["pass"]: p["result"].get("file") for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
keep = json.load(open("gen-v3-keep.json"))
def group(r): return r["detail"].get("group", r["label"])
for p, s, path, mode in [(45, "mt_B", "cells/run1/mt_B.jsonl", "think"), (46, "mt_B", "cells/run1/mt_B.jsonl", "nothink"),
                         (47, "cB", "cells/run2/cB.jsonl", "nothink")]:
    R = [json.loads(l) for l in open(V2 + path)]
    if len(R) == 320: R = [R[i] for i in keep]
    out = json.load(open(passes[p])); print(f"{s} {mode} (pass {p})")
    for arm in ("original", "edited"):
        st = {}; correct = pre = stray = 0
        for r, o in zip(R, out[arm]):
            smp = o["samples"][0]; c = bool(smp["tool_calls"]); a = st.setdefault(group(r), [0, 0]); a[0] += 1; a[1] += c
            correct += (c == (r["label"] == "call")); pre += c and bool((smp["content"] or "").strip())
            stray += "<tool_call>" in (smp["content"] or "")
        print(f"   {arm:8s} decision acc {correct / len(R):.3f} | " + ", ".join(f"{k} call {v[1] / v[0]:.2f}" for k, v in st.items())
              + f" | preamble calls {pre} | stray markup {stray}")
