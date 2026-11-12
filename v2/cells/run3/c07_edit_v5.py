# Run 3, cell 7. Edit v5 = v4 (neuron 1488) + neuron T (slot J2) with THETA, K from c06.
# Generation requests (both arms = base vs v5), held-out: tB think, mB think, tB nothink, gB think.
import json
T = torch.load("design_T.pt", weights_only=False); q = torch.load("up_parts.pt", weights_only=False)["q"].double()
u = torch.load("design_A.pt", weights_only=False)["u"]; J2 = T["J2"]
gn, un, dn = (f"model.language_model.layers.23.mlp.{t}_proj.weight" for t in ("gate", "up", "down"))
Wg, Wu, Wd = load(gn), load(un), load(dn)
Wg[J2] = T["gate"].to(Wg.dtype); Wu[J2] = (T["rT"] - T["theta_final"] * q / 11.9).to(Wu.dtype); Wd[:, J2] = (T["K_final"] * u).to(Wd.dtype)
for name, W in ((gn, Wg), (un, Wu), (dn, Wd)): save_edit(name, W)
print("v5 written: THETA", T["theta_final"], "K", T["K_final"], "| changed weights", sum(int((load(n) != load_original(n)).sum()) for n in (gn, un, dn)))
V2 = "@ROOT@/v2/"
def args_map(a):
    while isinstance(a, str):
        try: a = json.loads(a)
        except Exception: return {"value": a}
    return a if isinstance(a, dict) else {"value": a}
def fix_api(ms):
    return [{**m, "tool_calls": [{**c, "function": {**c["function"], "arguments": json.dumps(args_map(c["function"]["arguments"]))}} for c in m["tool_calls"]]} if m.get("tool_calls") else m for m in ms]
for s, think in (("tB", True), ("mB", True), ("tB", False), ("gB", True)):
    R = [json.loads(l) for l in open(V2 + f"cells/run3/{s}.jsonl")]
    json.dump({"type": "generate", "model": "both", "enable_thinking": think, "max_tokens": 8192, "temperature": 1.0,
               "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "seed": 0,
               "items": [{"messages": fix_api(r["messages"]), "tools": r["tools"]} for r in R]},
              open(f"req-gen-{s}-{'think' if think else 'nothink'}-v5.json", "w"))
    print(s, "think" if think else "nothink", len(R))
