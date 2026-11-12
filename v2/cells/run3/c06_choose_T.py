# Run 3, cell 6. Choose THETA and K for neuron T under a constraint: p_call on call rows may fall by at most 0.02
# (tA and mA each), because BFCL weights call-needed categories more than irrelevance. Objective: mean accuracy.
T = torch.load("design_T.pt", weights_only=False); D = torch.load("think_A.pt", weights_only=False)
d4 = torch.load("design_v4.pt", weights_only=False); q = torch.load("up_parts.pt", weights_only=False)["q"].double()
silu = torch.nn.functional.silu; cm = T["c_median"]
lo = lambda lp: lp - torch.log1p(-lp.exp().clamp(max=1 - 1e-9))
def stats(k, theta, K):
    Xs, y, lp = D[k]["X"], D[k]["y"], D[k]["lp"]
    h = silu(Xs @ T["gate"].double()) * (Xs @ T["rT"] - theta * (Xs @ q) / 11.9)
    p = torch.sigmoid(lo(lp) + 4.6 * d4["K"] * silu(Xs @ d4["gate"].double()) * (Xs @ d4["up"].double()) + cm * K * h)
    g = np.array(D[k]["group"])
    return float(torch.where(y, p, 1 - p).mean()), {gg: round(float(p[torch.tensor(g == gg)].mean()), 3) for gg in sorted(set(g))}
base = {k: stats(k, 0, 0) for k in ("tA", "mA")}
grid = [(t, k) for t in np.arange(-1.0, 2.01, 0.125) for k in (0.02, 0.05, 0.1, 0.18, 0.3)]
okg = [g for g in grid if all(stats(k, *g)[1]["call"] >= base[k][1]["call"] - 0.02 for k in ("tA", "mA"))]
best = max(okg, key=lambda g: stats("tA", *g)[0] + stats("mA", *g)[0])
print("constrained choice THETA", best[0], "K", best[1])
for k in ("tA", "mA"): print(f"  {k}: base {base[k][0]:.3f} {base[k][1]} -> {stats(k, *best)[0]:.3f} {stats(k, *best)[1]}")
T.update({"theta_final": float(best[0]), "K_final": float(best[1]), "rule": "call-row p_call drop <= 0.02"}); torch.save(T, "design_T.pt")
