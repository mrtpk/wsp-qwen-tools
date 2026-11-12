# Run 2. Offline check of edit v3 (theta, K from c52) on glaive gT (pass 37, nothink) and on cA thinking (pass 31).
# Also K beyond 1.0 on all sets, because K = 1.0 is the grid edge.
import json
P = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
d = torch.load("design_v3.pt", weights_only=False); r2, gate, cm = d["r2"], d["gate"].double(), d["c_median"]
q = torch.load("up_parts.pt", weights_only=False)["q"].double()
def load_dec(p, info):
    z = np.load(P[p]["file"]); first = {}
    for k, (i, _) in enumerate(z["rows"]): first.setdefault(int(i), k)
    idx = [first[i] for i in range(len(info))]
    return torch.tensor(z["mlp_in_23"]).double()[idx], torch.tensor([m["label"] == "call" for m in info]), torch.tensor(z["logp"][:, 0]).double()[idx]
lo = lambda lp: lp - torch.log1p(-lp.exp().clamp(max=1 - 1e-9))
silu = torch.nn.functional.silu
def acc(S, theta, K):
    X, y, lp = S; p = torch.sigmoid(lo(lp) + cm * K * silu(X @ gate) * (X @ r2 - theta * (X @ q) / 11.9))
    return float(torch.where(y, p, 1 - p).mean()), float(p[y].mean()), float(p[~y].mean())
mB = json.load(open("probe-meta-heldout.json")); m2 = json.load(open("probe-meta-r2.json"))
S = {"cA": load_dec(32, m2["probe-cA-nothink"]), "mt_A": load_dec(34, json.load(open("probe-meta-mtA-layers.json"))),
     "cB": load_dec(35, mB["cB"]), "mt_B": load_dec(36, mB["mt_B"]), "gT glaive": load_dec(37, json.load(open("probe-meta-gT.json"))),
     "cA think": load_dec(31, m2["probe-cA-think"])}
th = d["theta_final"]
print("theta", th, "| expected decision accuracy for K = 0 (base), 0.5, 1.0, 1.5, 2.0, 3.0")
for n, s in S.items():
    print(f"  {n:10s} " + "  ".join(f"{acc(s, th, k)[0]:.3f}" for k in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)),
          f"| p_call call/no-call at K=1: {acc(s, th, 1.0)[1]:.2f}/{acc(s, th, 1.0)[2]:.2f} (base {acc(s, th, 0)[1]:.2f}/{acc(s, th, 0)[2]:.2f})")
