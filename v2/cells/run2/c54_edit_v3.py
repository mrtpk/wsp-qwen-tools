# Run 2. Final edit v3: neuron 1488 of layer 23. theta and the read from c51/c52, K = 0.5 (c53: saturation from 0.5).
# Then generation requests (both arms) on the held-out sets cB, mt_B (both modes) and glaive gT (nothink).
import json
d = torch.load("design_v3.pt", weights_only=False); q = torch.load("up_parts.pt", weights_only=False)["q"]
u = torch.load("design_A.pt", weights_only=False)["u"]
K, THETA, J = 0.5, d["theta_final"], 1488
gn, un, dn = (f"model.language_model.layers.23.mlp.{t}_proj.weight" for t in ("gate", "up", "down"))
Wg, Wu, Wd = load_original(gn), load_original(un), load_original(dn)
Wg[J] = d["gate"].to(Wg.dtype); Wu[J] = (d["r2"] - THETA * q.double() / 11.9).to(Wu.dtype); Wd[:, J] = (K * u).to(Wd.dtype)
for name, W in ((gn, Wg), (un, Wu), (dn, Wd)): save_edit(name, W)
print("K", K, "theta", THETA, "changed weights", sum(int((load(n) != load_original(n)).sum()) for n in (gn, un, dn)))
V2 = "@ROOT@/v2/"
def args_map(a):
    while isinstance(a, str):
        try: a = json.loads(a)
        except Exception: return {"value": a}
    return a if isinstance(a, dict) else {"value": a}
def fix(ms):
    return [{**m, "tool_calls": [{**c, "function": {**c["function"], "arguments": args_map(c["function"]["arguments"])}} for c in m["tool_calls"]]} if m.get("tool_calls") else m for m in ms]
for name, path, modes in (("cB", "cells/run2/cB.jsonl", (True, False)), ("mt_B", "cells/run1/mt_B.jsonl", (True, False)), ("gT", "cells/run2/gT.jsonl", (False,))):
    R = [json.loads(l) for l in open(V2 + path)]
    for think in modes:
        json.dump({"type": "generate", "model": "both", "enable_thinking": think, "max_tokens": 8192, "temperature": 1.0,
                   "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "seed": 0,
                   "items": [{"messages": fix(r["messages"]), "tools": r["tools"]} for r in R]},
                  open(f"req-gen-{name}-{'think' if think else 'nothink'}-v3.json", "w"))
        print(name, "think" if think else "nothink", len(R))
