import json
passes = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
G = torch.load("gate_G.pt", weights_only=False); P = torch.load("up_parts.pt", weights_only=False); C = torch.load("context_c.pt", weights_only=False)
q, r, th, sd = P["q"], P["r"], P["theta"], P["sd"]
def silu(z): return z * torch.sigmoid(z)
def gate(X, delta): return 2 * (X @ G) - 1.33 * (X @ C["c"] - C["tau"] * (X @ q) / 11.9) + delta * (X @ q)
def up(X): return (X @ r - th * (X @ q) / 11.9) / sd
D = torch.load("probes_A.pt", weights_only=False)
z = np.load(passes[20]["file"]); info = json.load(open("answer-probe-meta.json")); XA = torch.tensor(z["mlp_in_23"]); rows = z["rows"]
z = np.load(passes[20]["file"]); info = json.load(open("answer-probe-meta.json")); XA = torch.tensor(z["mlp_in_23"]); rows = z["rows"]
lpA = torch.tensor(z["logp"][:, 0])
grp = torch.tensor([info[int(i)]["group"] != "call" for i, _ in rows])
firstpos = torch.zeros(len(rows), dtype=torch.bool); seen = set()
for k, (i, _) in enumerate(rows):
    if int(i) not in seen: firstpos[k] = True; seen.add(int(i))
for delta in (0.0, 0.5):
    h = silu(gate(XA, delta)) * up(XA); m = ~firstpos
    bad = m & grp & (h > 1)
    print(f"delta {delta}: answer positions h>+1: {int((h[m] > 1).sum())}, h<-1: {int((h[m] < -1).sum())} of {int(m.sum())};"
          f" h>+1 where no call is correct: {int(bad.sum())}; base p_call there max {float(lpA[bad].exp().max()) if int(bad.sum()) else 0:.3f}")
