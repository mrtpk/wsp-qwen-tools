import json
passes = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
G = torch.load("gate_G.pt", weights_only=False); P = torch.load("up_parts.pt", weights_only=False); C = torch.load("context_c.pt", weights_only=False)
q, r, th, sd = P["q"], P["r"], P["theta"], P["sd"]
def silu(z): return z * torch.sigmoid(z)
def gate(X, delta): return 2 * (X @ G) - 1.33 * (X @ C["c"] - C["tau"] * (X @ q) / 11.9) + delta * (X @ q)
def up(X): return (X @ r - th * (X @ q) / 11.9) / sd
D = torch.load("probes_A.pt", weights_only=False)
z = np.load(passes[20]["file"]); info = json.load(open("answer-probe-meta.json")); XA = torch.tensor(z["mlp_in_23"]); rows = z["rows"]
z = np.load(passes[22]["file"]); X = torch.tensor(z["mlp_in_23"]); lp = torch.tensor(z["logp"][:, 0]); rows = z["rows"]
second = torch.tensor([k % 2 == 1 for k in range(len(rows))]); think = torch.tensor([int(i) >= 128 for i, _ in rows])
for delta in (0.0, 0.5):
    h = silu(gate(X, delta)) * up(X)
    for t in (False, True):
        for s, nm in ((False, "before 1st call"), (True, "before 2nd call")):
            m = (second == s) & (think == t)
            print(f"delta {delta} {'think' if t else 'nothink'} {nm}: base p_call {float(lp[m].exp().mean()):.3f}  h mean {float(h[m].mean()):+.2f} min {float(h[m].min()):+.2f} max {float(h[m].max()):+.2f}  q.x {float((X[m] @ q).mean()):.2f}")
