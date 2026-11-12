D = torch.load("probes_A.pt", weights_only=False); F = torch.load("fit_A.pt", weights_only=False)
G = torch.load("gate_G.pt", weights_only=False); q = F["q"]
names = list(D); tr = {n: D[n]["rid"] % 2 == 0 for n in names}; dec = {n: torch.tensor(D[n]["kind"] == "dec") for n in names}
def auroc(s, y):
    s = torch.as_tensor(s, dtype=torch.float64); y = torch.as_tensor(y).bool()
    order = torch.argsort(s); r = torch.empty_like(s); r[order] = torch.arange(1, len(s) + 1, dtype=torch.float64)
    n1, n0 = int(y.sum()), int((~y).sum()); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
mt = [n for n in names if "mt_" in n]; st = [n for n in names if "st_" in n]
A1 = torch.cat([D[n]["X"][dec[n] & tr[n]] for n in mt]); A0 = torch.cat([D[n]["X"][dec[n] & tr[n]] for n in st])
S = torch.cov(torch.cat([A1 - A1.mean(0), A0 - A0.mean(0)]).T); S = S + 0.1 * S.diagonal().mean() * torch.eye(1024)
c = torch.linalg.solve(S, A1.mean(0) - A0.mean(0)); c = c / c.norm()
# Context score at decision positions, with q.x/11.9 as the bias proxy: e = c.x - tau*q.x/11.9, tau = st mean of c.x.
tau = float((A0 @ c).mean())
for n in names:
    X = D[n]["X"]; e = X @ c - tau * (X @ q) / 11.9; te = ~tr[n]
    print(n[6:], "e at dec (held-out) mean", round(float(e[dec[n] & te].mean()), 2), "sd", round(float(e[dec[n] & te].std()), 2),
          "| e elsewhere min", round(float(e[~dec[n]].min()), 2), "max", round(float(e[~dec[n]].max()), 2))
# Separation of mt from st at held-out decision positions:
s_mt = torch.cat([(D[n]["X"] @ c)[dec[n] & ~tr[n]] for n in mt]); s_st = torch.cat([(D[n]["X"] @ c)[dec[n] & ~tr[n]] for n in st])
print("AUROC mt vs st at dec", round(auroc(torch.cat([s_mt, s_st]), torch.cat([torch.ones(len(s_mt)), torch.zeros(len(s_st))])), 3))
# r trained on mt only: does it beat the model in mt?
X1 = torch.cat([D[n]["X"][dec[n] & tr[n] & D[n]["label"]] for n in mt]); X0 = torch.cat([D[n]["X"][dec[n] & tr[n] & ~D[n]["label"]] for n in mt])
for lam in (0.1, 1.0, 10.0):
    S = torch.cov(torch.cat([X1 - X1.mean(0), X0 - X0.mean(0)]).T); S = S + lam * S.diagonal().mean() * torch.eye(1024)
    rm = torch.linalg.solve(S, X1.mean(0) - X0.mean(0))
    print("r_mt lam", lam, {n[6:]: round(auroc((D[n]["X"] @ rm)[dec[n] & ~tr[n]], D[n]["label"][dec[n] & ~tr[n]]), 3) for n in mt},
          "model p_call", {n[6:]: round(auroc(D[n]["lp"][dec[n] & ~tr[n]], D[n]["label"][dec[n] & ~tr[n]]), 3) for n in mt})
torch.save({"c": c, "tau": tau}, "context_c.pt")
