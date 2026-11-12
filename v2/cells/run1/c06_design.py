import json
D = torch.load("probes_A.pt", weights_only=False); F = torch.load("fit_A.pt", weights_only=False)
passes = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
q = F["q"]
for n in D:
    dec = torch.tensor(D[n]["kind"] == "dec"); s = D[n]["X"][dec] @ q
    print(n[6:], "q.x at dec: mean", round(float(s.mean()), 2), "CV", round(float(s.std() / s.mean()), 3))
# Neuron use in layer 23: mean |h_j| * |D_j| over all probe positions.
Dn = load("model.language_model.layers.23.mlp.down_proj.weight")
H = torch.cat([torch.tensor(np.load(passes[p]["file"])["mlp_act_23"]) for p in (5, 6, 7, 8)])
use = H.abs().mean(0) * Dn.norm(dim=0)
order = torch.argsort(use)
print("least-used neurons:", [(int(j), f"{float(use[j]):.2e}") for j in order[:6]], "median use", f"{float(use.median()):.2e}")
# Protected write direction (weights only): raise <tool_call>, change the common practice tokens least.
E = load("model.language_model.embed_tokens.weight"); g = 1 + load("model.language_model.norm.weight")
TC = tok.convert_tokens_to_ids("<tool_call>")
counts = {}
for p in (1, 2, 3, 4):
    for o in json.load(open(passes[p]["file"]))["original"]:
        s = o["samples"][0]
        text = (s["reasoning"] or "") + " " + (s["content"] or "") + " " + " ".join(c["name"] + " " + c["arguments"] for c in s["tool_calls"])
        for t in tok(text, add_special_tokens=False)["input_ids"]:
            counts[t] = counts.get(t, 0) + 1
counts.pop(TC, None)
common = [t for t, _ in sorted(counts.items(), key=lambda x: -x[1])[:4000]]
C = E[common] * g; M = C.T @ C; M = M + 0.1 * M.diagonal().mean() * torch.eye(1024)
t = g * E[TC]; u = torch.linalg.solve(M, t); u = u / u.norm()
side = (C @ u).abs()
print("write u: tool_call gain", round(float(t @ u), 3), "common |gain| mean", round(float(side.mean()), 4), "max", round(float(side.max()), 4),
      "| plain readout gain", round(float(t.norm()), 3))
torch.save({"u": u, "common": common, "use": use}, "design_A.pt")
