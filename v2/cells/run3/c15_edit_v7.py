# Run 3, cell 15. Edit v7 = v4 + neuron T7 (slot J2). Same gate as T6 (2 G_T). New up read r_T7.
# G25: T6 learned "answer in text when a tool matches" from the tA "text" group. The target convention is:
# call when a tool matches. So r_T7 = LDA (lam 3) on tA call vs tA swap, plus mA call vs mA text
# (a multi-turn text step follows tool results; it is a real "stop"). tA "text" is not in the fit.
# THETA, K: best mean accuracy on tA (call + swap) and mA, with these limits on mean p_call:
#   tA call and mA call fall by at most 0.02; tA text falls by at most 0.02 (do not suppress matched tools).
import json
T = torch.load("design_T.pt", weights_only=False); D = torch.load("think_A.pt", weights_only=False)
d4 = torch.load("design_v4.pt", weights_only=False); q = torch.load("up_parts.pt", weights_only=False)["q"].double()
gT = 2 * torch.load("gate_T.pt", weights_only=False)["g"]; silu = torch.nn.functional.silu; cm = T["c_median"]
lo = lambda lp: lp - torch.log1p(-lp.exp().clamp(max=1 - 1e-9))
def lda(X1, X0, lam=3.0):
    S = torch.cov(torch.cat([X1 - X1.mean(0), X0 - X0.mean(0)]).T)
    S = S + lam * S.diagonal().mean() * torch.eye(S.shape[0], dtype=S.dtype)
    return torch.linalg.solve(S, X1.mean(0) - X0.mean(0))
def auroc(s, y):
    s = torch.as_tensor(s, dtype=torch.float64); y = torch.as_tensor(y).bool()
    order = torch.argsort(s); r = torch.empty_like(s); r[order] = torch.arange(1, len(s) + 1, dtype=torch.float64)
    n1, n0 = int(y.sum()), int((~y).sum()); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
G = {k: np.array(D[k]["group"]) for k in ("tA", "mA")}
keep = {"tA": torch.tensor(G["tA"] != "text"), "mA": torch.ones(len(G["mA"]), dtype=torch.bool)}
X = torch.cat([D[k]["X"][keep[k]] for k in ("tA", "mA")]); y = torch.cat([D[k]["y"][keep[k]] for k in ("tA", "mA")])
r7 = lda(X[y], X[~y]); r7 = r7 / float((X @ r7).std())
for k in ("tA", "mA"):
    s = D[k]["X"] @ r7
    print(f"  r_T7 on {k}: " + ", ".join(f"{g} mean {float(s[torch.tensor(G[k] == g)].mean()):+.2f}" for g in sorted(set(G[k]))))
    s6 = D[k]["X"] @ T["rT"]
    print(f"  r_T6 on {k}: " + ", ".join(f"{g} mean {float(s6[torch.tensor(G[k] == g)].mean()):+.2f}" for g in sorted(set(G[k]))))
print("  cos(r_T6, r_T7)", round(float(torch.nn.functional.cosine_similarity(T["rT"], r7, dim=0)), 3))
def stats(k, theta, K):
    Xs, yy, lp = D[k]["X"], D[k]["y"], D[k]["lp"]
    h = silu(Xs @ gT) * (Xs @ r7 - theta * (Xs @ q) / 11.9)
    p = torch.sigmoid(lo(lp) + 4.6 * d4["K"] * silu(Xs @ d4["gate"].double()) * (Xs @ d4["up"].double()) + cm * K * h)
    m = keep[k]
    return float(torch.where(yy, p, 1 - p)[m].mean()), {g: round(float(p[torch.tensor(G[k] == g)].mean()), 3) for g in sorted(set(G[k]))}
base = {k: stats(k, 0, 0) for k in ("tA", "mA")}
grid = [(t, kk) for t in np.arange(-1.0, 2.01, 0.125) for kk in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)]
def ok(g):
    a, b = stats("tA", *g)[1], stats("mA", *g)[1]
    return (a["call"] >= base["tA"][1]["call"] - 0.02 and b["call"] >= base["mA"][1]["call"] - 0.02
            and a["text"] >= base["tA"][1]["text"] - 0.02)
okg = [g for g in grid if ok(g)]
THETA, K = max(okg, key=lambda g: stats("tA", *g)[0] + stats("mA", *g)[0])
print("T7: THETA", THETA, "K", K, "| grid points allowed", len(okg), "of", len(grid))
for k in ("tA", "mA"): print(f"  {k}: base {base[k][0]:.3f} {base[k][1]} -> {stats(k, THETA, K)[0]:.3f} {stats(k, THETA, K)[1]}")
u = torch.load("design_A.pt", weights_only=False)["u"]; J2 = T["J2"]; up = r7 - THETA * q / 11.9
gn, un, dn = (f"model.language_model.layers.23.mlp.{t}_proj.weight" for t in ("gate", "up", "down"))
Wg, Wu, Wd = load(gn), load(un), load(dn)
Wg[J2] = gT.to(Wg.dtype); Wu[J2] = up.to(Wu.dtype); Wd[:, J2] = (K * u).to(Wd.dtype)
for name, W in ((gn, Wg), (un, Wu), (dn, Wd)): save_edit(name, W)
torch.save({"gate": gT.float(), "up": up.float(), "rT7": r7, "K": K, "theta": THETA, "J2": J2}, "design_T7.pt")
print("v7 written; changed weights", sum(int((load(n) != load_original(n)).sum()) for n in (gn, un, dn)))
