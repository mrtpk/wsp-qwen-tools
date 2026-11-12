# Run 3, cell 13. Edit v6 = v4 (neuron 1488) + neuron T6 (slot J2): gate 2 G_T (c12), up r_T - THETA q / 11.9.
# THETA, K by the c06 rule (call-row p_call drop <= 0.02 on tA, mA), with the logit gain c from pass 5.
# Off-position check on pass 10. Then generation requests are the v5 ones (same sets; the edited arm is now v6).
import json
P = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
T = torch.load("design_T.pt", weights_only=False); D = torch.load("think_A.pt", weights_only=False)
d4 = torch.load("design_v4.pt", weights_only=False); q = torch.load("up_parts.pt", weights_only=False)["q"].double()
gT = 2 * torch.load("gate_T.pt", weights_only=False)["g"]; silu = torch.nn.functional.silu; cm = T["c_median"]
lo = lambda lp: lp - torch.log1p(-lp.exp().clamp(max=1 - 1e-9))
def stats(k, theta, K):
    Xs, y, lp = D[k]["X"], D[k]["y"], D[k]["lp"]
    h = silu(Xs @ gT) * (Xs @ T["rT"] - theta * (Xs @ q) / 11.9)
    p = torch.sigmoid(lo(lp) + 4.6 * d4["K"] * silu(Xs @ d4["gate"].double()) * (Xs @ d4["up"].double()) + cm * K * h)
    g = np.array(D[k]["group"])
    return float(torch.where(y, p, 1 - p).mean()), {gg: round(float(p[torch.tensor(g == gg)].mean()), 3) for gg in sorted(set(g))}
base = {k: stats(k, 0, 0) for k in ("tA", "mA")}
grid = [(t, k) for t in np.arange(-1.0, 2.01, 0.125) for k in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)]
okg = [g for g in grid if all(stats(k, *g)[1]["call"] >= base[k][1]["call"] - 0.02 for k in ("tA", "mA"))]
THETA, K = max(okg, key=lambda g: stats("tA", *g)[0] + stats("mA", *g)[0])
print("T6: THETA", THETA, "K", K)
for k in ("tA", "mA"): print(f"  {k}: base {base[k][0]:.3f} {base[k][1]} -> {stats(k, THETA, K)[0]:.3f} {stats(k, THETA, K)[1]}")
info = json.load(open("probe-meta-offpos.json")); X10 = torch.tensor(np.load(P[10]["file"])["mlp_in_23"]).double()
kind, think = [], []
for inf in info:
    n = 1 + inf["n_inside"] + 6; kind += ["decision"] + ["inside call"] * inf["n_inside"] + ["other"] * 6; think += [inf["think"]] * n
kind, think = np.array(kind), np.array(think)
up = T["rT"] - THETA * q / 11.9; pre = X10 @ gT; kh = K * silu(pre) * (X10 @ up)
for md in (True, False):
    for kd in ("decision", "inside call", "other"):
        m = torch.tensor((kind == kd) & (think == md))
        print(f"  T6 {'think  ' if md else 'nothink'} {kd:11s}: open {float((pre[m] > 0).double().mean()):.2f} mean |K h| {float(kh[m].abs().mean()):.3f} max {float(kh[m].abs().max()):.2f}")
u = torch.load("design_A.pt", weights_only=False)["u"]; J2 = T["J2"]
gn, un, dn = (f"model.language_model.layers.23.mlp.{t}_proj.weight" for t in ("gate", "up", "down"))
Wg, Wu, Wd = load(gn), load(un), load(dn)
Wg[J2] = gT.to(Wg.dtype); Wu[J2] = up.to(Wu.dtype); Wd[:, J2] = (K * u).to(Wd.dtype)
for name, W in ((gn, Wg), (un, Wu), (dn, Wd)): save_edit(name, W)
torch.save({"gate": gT.float(), "up": up.float(), "K": K, "theta": THETA, "J2": J2}, "design_T6.pt")
print("v6 written; changed weights", sum(int((load(n) != load_original(n)).sum()) for n in (gn, un, dn)))
