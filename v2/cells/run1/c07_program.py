# Program neuron 1488 of layer 23 as: h = silu(G.x) * (U.x); out = D[:,j] * h.
D = torch.load("probes_A.pt", weights_only=False); F = torch.load("fit_A.pt", weights_only=False)
Ds = torch.load("design_A.pt", weights_only=False)
names = list(D); J = 1488
def auroc(s, y):
    s = torch.as_tensor(s, dtype=torch.float64); y = torch.as_tensor(y).bool()
    order = torch.argsort(s); r = torch.empty_like(s); r[order] = torch.arange(1, len(s) + 1, dtype=torch.float64)
    n1, n0 = int(y.sum()), int((~y).sum()); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
tr = {n: D[n]["rid"] % 2 == 0 for n in names}; dec = {n: torch.tensor(D[n]["kind"] == "dec") for n in names}
# Gate: ridge regression on raw x (no intercept; x has near-constant parts): +4 at decision, -8 elsewhere.
Xtr = torch.cat([D[n]["X"][tr[n]] for n in names]); ytr = torch.cat([torch.where(dec[n][tr[n]], 4.0, -8.0) for n in names])
for lam in (1e-1, 1.0, 10.0):
    G = torch.linalg.solve(Xtr.T @ Xtr + lam * torch.eye(1024), Xtr.T @ ytr)
    zs = {n: D[n]["X"][~tr[n]] @ G for n in names}
    print("lam", lam, {n[6:]: (round(float(zs[n][dec[n][~tr[n]]].min()), 2), round(float(zs[n][~dec[n][~tr[n]]].max()), 2),
                              round(float(zs[n][~dec[n][~tr[n]]].quantile(0.999)), 2)) for n in names})
torch.save(G, "gate_G.pt")
# Up row: r.x - theta * (q.x)/11.9 at decision positions; r with lam 1.0 (best held-out above).
X1, X0 = F["X1"], F["X0"]
S = torch.cov(torch.cat([X1 - X1.mean(0), X0 - X0.mean(0)]).T); S = S + 1.0 * S.diagonal().mean() * torch.eye(1024)
r = torch.linalg.solve(S, X1.mean(0) - X0.mean(0)); r = r / r.norm(); q = F["q"]
for n in names:
    m = dec[n] & tr[n]; s = D[n]["X"][m] @ r; y = D[n]["label"][m]
    print(n[6:], "r.x call-desired", round(float(s[y].mean()), 3), "no-call", round(float(s[~y].mean()), 3), "sd", round(float(s.std()), 3))
s1 = X1 @ r; s0 = X0 @ r; theta = float((s1.mean() + s0.mean()) / 2); sd = float(torch.cat([s1, s0]).std())
print("theta", round(theta, 3), "sd", round(sd, 3))
torch.save({"r": r, "q": q, "theta": theta, "sd": sd}, "up_parts.pt")
