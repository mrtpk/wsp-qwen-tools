D = torch.load("probes_A.pt", weights_only=False)
def auroc(s, y):
    s = torch.as_tensor(s, dtype=torch.float64); y = torch.as_tensor(y).bool()
    order = torch.argsort(s); r = torch.empty_like(s); r[order] = torch.arange(1, len(s) + 1, dtype=torch.float64)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def lda(X1, X0, lam=1e-2):
    S = torch.cov(torch.cat([X1 - X1.mean(0), X0 - X0.mean(0)]).T)
    S = S + lam * S.diagonal().mean() * torch.eye(S.shape[0])
    w = torch.linalg.solve(S, X1.mean(0) - X0.mean(0)); return w / w.norm()
names = list(D)
dec = {n: torch.tensor(D[n]["kind"] == "dec") for n in names}
tr = {n: D[n]["rid"] % 2 == 0 for n in names}
# r: desired call (1) vs desired no-call (0), decision positions of all 4 contexts, even rows.
X1 = torch.cat([D[n]["X"][dec[n] & tr[n] & D[n]["label"]] for n in names])
X0 = torch.cat([D[n]["X"][dec[n] & tr[n] & ~D[n]["label"]] for n in names])
for lam in (1e-3, 1e-2, 1e-1, 1.0):
    r = lda(X1, X0, lam)
    print("lam", lam, "held-out AUROC r (desired label):",
          {n[6:]: round(auroc((D[n]["X"] @ r)[dec[n] & ~tr[n]], D[n]["label"][dec[n] & ~tr[n]]), 3) for n in names})
# q: decision position vs background and in-call positions.
Q1 = torch.cat([D[n]["X"][dec[n] & tr[n]] for n in names]); Q0 = torch.cat([D[n]["X"][~dec[n] & tr[n]] for n in names])
q = lda(Q1, Q0, 1e-2)
for n in names:
    s = D[n]["X"] @ q; te = ~tr[n]
    print(n[6:], "q AUROC dec vs rest", round(auroc(s[te], dec[n][te]), 4), "dec min", round(float(s[dec[n] & te].min()), 2),
          "rest max", round(float(s[~dec[n] & te].max()), 2), "rest p99.9", round(float(s[~dec[n] & te].quantile(0.999)), 2))
# b: bias proxy, CV of mhat . x over all positions.
Xall = torch.cat([D[n]["X"] for n in names]); m = Xall.mean(0); mh = m / m.norm(); s = Xall @ mh
print("bias proxy mhat.x mean", round(float(s.mean()), 3), "CV", round(float(s.std() / s.mean()), 3), "|x| mean", round(float(Xall.norm(dim=1).mean()), 2))
torch.save({"X1": X1, "X0": X0, "q": q, "mh": mh}, "fit_A.pt")
