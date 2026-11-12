import json
passes = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
D = torch.load("probes_A.pt", weights_only=False)
za = np.load(passes[20]["file"]); info = json.load(open("answer-probe-meta.json")); XA = torch.tensor(za["mlp_in_23"]); rowsA = za["rows"]
pre, ans, decA, pre_row = [], [], [], []
for i, m in enumerate(info):
    ks = [k for k in range(len(rowsA)) if rowsA[k][0] == i]; decA.append(ks[0])
    if m["called"]: pre.append(ks[-1]); pre_row.append(i); ans += ks[1:-1]
    else: ans += ks[1:]
zp = np.load(passes[22]["file"]); XP = torch.tensor(zp["mlp_in_23"])
neg = {"A-probe all": torch.cat([D[n]["X"] for n in D]), "answer": XA[ans], "answer-decision": XA[decA], "parallel": XP}
pos = XA[pre]; tr = torch.tensor([r % 2 == 0 for r in pre_row])
N = torch.cat(list(neg.values())); ntr = torch.rand(len(N), generator=torch.Generator().manual_seed(0)) < 0.5
X1, X0 = pos[tr], N[ntr]
S = torch.cov(torch.cat([X1 - X1.mean(0), X0 - X0.mean(0)]).T); S = S + 0.3 * S.diagonal().mean() * torch.eye(1024)
v = torch.linalg.solve(S, X1.mean(0) - X0.mean(0)); v = v / v.norm()
# Ridge gate on raw x (no intercept): +4 at late-call positions, -8 elsewhere (train halves).
Xt = torch.cat([X1, X0]); yt = torch.cat([torch.full((len(X1),), 4.0), torch.full((len(X0),), -8.0)])
w = torch.ones(len(Xt)); w[: len(X1)] = len(X0) / len(X1)          # balance the classes
Gv = torch.linalg.solve(Xt.T @ (w[:, None] * Xt) + 1.0 * torch.eye(1024), Xt.T @ (w * yt))
print("held-out late-call positions: gate", [round(float(x), 1) for x in (pos[~tr] @ Gv)])
for name, Xn in neg.items():
    s = Xn @ Gv
    print(f"{name:16s} n={len(Xn):6d} gate max {float(s.max()):6.2f} p99.9 {float(s.quantile(0.999)):6.2f} share>0 {float((s > 0).float().mean()):.4f}")
torch.save({"Gv": Gv}, "veto_gate.pt")
