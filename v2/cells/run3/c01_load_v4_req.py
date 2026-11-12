# Run 3, cell 1. Write edit v4 (neuron 1488 of layer 23, from run 2: design_v4.pt) into the run-3 store.
# Then thinking-mode generation requests (both arms) on the fresh sets tA and mA.
import json
d4 = torch.load("design_v4.pt", weights_only=False); u = torch.load("design_A.pt", weights_only=False)["u"]
J = 1488
gn, un, dn = (f"model.language_model.layers.23.mlp.{t}_proj.weight" for t in ("gate", "up", "down"))
Wg, Wu, Wd = load_original(gn), load_original(un), load_original(dn)
Wg[J] = d4["gate"].to(Wg.dtype); Wu[J] = d4["up"].to(Wu.dtype); Wd[:, J] = (d4["K"] * u).to(Wd.dtype)
for name, W in ((gn, Wg), (un, Wu), (dn, Wd)): save_edit(name, W)
print("v4 loaded; changed weights", sum(int((load(n) != load_original(n)).sum()) for n in (gn, un, dn)))
V2 = "@ROOT@/v2/"
def args_map(a):
    while isinstance(a, str):
        try: a = json.loads(a)
        except Exception: return {"value": a}
    return a if isinstance(a, dict) else {"value": a}
def fix_api(ms):
    return [{**m, "tool_calls": [{**c, "function": {**c["function"], "arguments": json.dumps(args_map(c["function"]["arguments"]))}} for c in m["tool_calls"]]} if m.get("tool_calls") else m for m in ms]
for s in ("tA", "mA"):
    R = [json.loads(l) for l in open(V2 + f"cells/run3/{s}.jsonl")]
    json.dump({"type": "generate", "model": "both", "enable_thinking": True, "max_tokens": 8192, "temperature": 1.0,
               "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "seed": 0,
               "items": [{"messages": fix_api(r["messages"]), "tools": r["tools"]} for r in R]}, open(f"req-gen-{s}-think.json", "w"))
    print(s, len(R))
