import json
passes = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
Gv = torch.load("veto_gate.pt", weights_only=False)["Gv"]
G = torch.load("gate_G.pt", weights_only=False); P = torch.load("up_parts.pt", weights_only=False); C = torch.load("context_c.pt", weights_only=False)
q = P["q"]
def silu(z): return z * torch.sigmoid(z)
def h1(X): return silu(2 * (X @ G) - 2.33 * (X @ C["c"] - C["tau"] * (X @ q) / 11.9) + 0.5 * (X @ q)) * (X @ P["r"] - P["theta"] * (X @ q) / 11.9) / P["sd"]
z = np.load(passes[27]["file"]); X = torch.tensor(z["mlp_in_23"]); rows = z["rows"]; lp = torch.tensor(z["logp"][:, 0]); meta = json.load(open("preamble-meta.json"))
first = []; seen = set()
for k, (i, _) in enumerate(rows):
    if int(i) not in seen: first.append(k); seen.add(int(i))
first = torch.tensor(first); s = X[first] @ Gv
for key in sorted(set(m["set"] + ("-think" if m["think"] else "-nothink") for m in meta)):
    idx = torch.tensor([j for j, m in enumerate(meta) if m["set"] + ("-think" if m["think"] else "-nothink") == key])
    print(f"{key:13s} n={len(idx):3d} veto gate before natural call: max {float(s[idx].max()):6.2f} share>0 {float((s[idx] > 0).float().mean()):.2f}"
          f" | neuron-1 h there: mean {float(h1(X[first[idx]]).mean()):+.2f} min {float(h1(X[first[idx]]).min()):+.2f} | base p_call {float(lp[first[idx]].exp().mean()):.2f}")
