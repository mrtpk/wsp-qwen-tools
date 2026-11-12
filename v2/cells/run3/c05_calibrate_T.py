# Run 3, cell 5. Calibrate neuron T (slot J2). Pass 5 (edited: v4 + T at K0) against passes 3, 4 (original), thinking
# decision positions. Measured: h_T and the change of the <tool_call> log-odds. The change also holds the small v4
# part; c = (d_logodds - v4 part) / (K0 h_T), with the v4 part from the run-2 gain (median 4.6 per unit K h).
# Grid over THETA and K on A by expected decision accuracy (tA and mA averaged).
import json
P = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
T = torch.load("design_T.pt", weights_only=False); D = torch.load("think_A.pt", weights_only=False)
d4 = torch.load("design_v4.pt", weights_only=False); q = torch.load("up_parts.pt", weights_only=False)["q"].double()
silu = torch.nn.functional.silu; J2 = T["J2"]
z = np.load(P[5]["file"]); hT = torch.tensor(z["mlp_act_23"][:, J2]).double(); lp_e = torch.tensor(z["logp"][:, 0]).double()
X = torch.cat([D["tA"]["X"], D["mA"]["X"]]); lp_o = torch.cat([D["tA"]["lp"], D["mA"]["lp"]])
lo = lambda lp: lp - torch.log1p(-lp.exp().clamp(max=1 - 1e-9))
h4 = d4["K"] * silu(X @ d4["gate"].double()) * (X @ d4["up"].double())
hT_pred = silu(X @ T["gate"].double()) * (X @ T["up"].double())
print("h_T predicted vs measured: corr", round(float(torch.corrcoef(torch.stack([hT_pred, hT]))[0, 1]), 4), "mean |h_T|", round(float(hT.abs().mean()), 2))
c = (lo(lp_e) - lo(lp_o) - 4.6 * h4) / (T["K0"] * hT); ok = hT.abs() > 0.5
cm = float(c[ok].median()); print("c median", round(cm, 3), "IQR", [round(float(v), 3) for v in torch.quantile(c[ok], torch.tensor([0.25, 0.75], dtype=c.dtype))])
def acc(k, theta, K):
    Xs, y, lp = D[k]["X"], D[k]["y"], D[k]["lp"]
    h = silu(Xs @ T["gate"].double()) * (Xs @ T["rT"] - theta * (Xs @ q) / 11.9)
    h4s = d4["K"] * silu(Xs @ d4["gate"].double()) * (Xs @ d4["up"].double())
    p = torch.sigmoid(lo(lp) + 4.6 * h4s + cm * K * h)
    g = np.array(D[k]["group"])
    return float(torch.where(y, p, 1 - p).mean()), {gg: round(float(p[torch.tensor(g == gg)].mean()), 2) for gg in sorted(set(g))}
grid = [(t, k) for t in np.arange(-0.5, 2.01, 0.25) for k in (0.02, 0.05, 0.1, 0.18, 0.3)]
best = max(grid, key=lambda g: acc("tA", *g)[0] + acc("mA", *g)[0])
print("chosen THETA", best[0], "K", best[1])
for k in ("tA", "mA"):
    print(f"  {k}: base {acc(k, 0, 0)[0]:.3f} {acc(k, 0, 0)[1]} -> {acc(k, *best)[0]:.3f} {acc(k, *best)[1]}")
    print("     K sweep at chosen THETA: " + "  ".join(f"K {kk}: {acc(k, best[0], kk)[0]:.3f}" for kk in (0.02, 0.05, 0.1, 0.18, 0.3)))
torch.save({**T, "theta_final": float(best[0]), "K_final": float(best[1]), "c_median": cm}, "design_T.pt")
