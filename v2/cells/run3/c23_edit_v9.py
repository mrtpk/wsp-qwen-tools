# Run 3, cell 23. Edit v9 = v8 with neuron T8 in slot J2 (replaces T7). Gate as T6/T7 (2 G_T).
# G27 follow-up: v7 still adds thinking text answers on BFCL (ast_decoder). Convention V2-D11 for the T read too:
# up read r_T8 = joint LDA (lam 3), thinking decisions of tA (pass 3), cA (run-2 pass 31), mA (pass 4).
#   Call: single-turn call + text, multi-turn call. No call: swap, multi-turn text. sd 1.
# THETA, K: best mean accuracy (same targets) on tA, cA, mA, with v4 (neuron 1488) in the model of p_call.
#   Limits: call mean p_call falls by at most 0.02; swap rises by at most 0.03 (tA, cA); mA text rises by at most 0.03.
# No held-out probe set for thinking; held-out check is by generation (tB, mB, gB thinking).
import json
T = torch.load("design_T.pt", weights_only=False); D = torch.load("think_A.pt", weights_only=False)
d4 = torch.load("design_v4.pt", weights_only=False); q = torch.load("up_parts.pt", weights_only=False)["q"].double()
gT = 2 * torch.load("gate_T.pt", weights_only=False)["g"]; silu = torch.nn.functional.silu; cm = T["c_median"]
lo = lambda lp: lp - torch.log1p(-lp.exp().clamp(max=1 - 1e-9))
def lda(X1, X0, lam=3.0):
    S = torch.cov(torch.cat([X1 - X1.mean(0), X0 - X0.mean(0)]).T)
    S = S + lam * S.diagonal().mean() * torch.eye(S.shape[0], dtype=S.dtype)
    return torch.linalg.solve(S, X1.mean(0) - X0.mean(0))
N = {k: {"X": D[k]["X"], "lp": D[k]["lp"], "y": D[k]["y"], "group": np.array(D[k]["group"])} for k in ("tA", "mA")}
m2 = json.load(open("imports/probe-meta-r2.json"))["probe-cA-think"]
z = np.load("imports/run2_pass31.npz"); first = {}
for k, (i, _) in enumerate(z["rows"]): first.setdefault(int(i), k)
idx = [first[i] for i in range(len(m2))]
N["cA"] = {"X": torch.tensor(z["mlp_in_23"]).double()[idx], "lp": torch.tensor(z["logp"][:, 0]).double()[idx],
           "y": torch.tensor([x["label"] == "call" for x in m2]), "group": np.array([x["group"] for x in m2])}
tgt = {k: (torch.tensor(N[k]["group"] != "swap") if k in ("tA", "cA") else N[k]["y"]) for k in N}
FIT = ("tA", "cA", "mA")
Xj = torch.cat([N[k]["X"] for k in FIT]); yj = torch.cat([tgt[k] for k in FIT])
r8 = lda(Xj[yj], Xj[~yj]); r8 = r8 / float((Xj @ r8).std())
for s in N:
    sc = N[s]["X"] @ r8; g = N[s]["group"]
    print(f"  r_T8 on {s}: " + ", ".join(f"{gg} mean {float(sc[torch.tensor(g == gg)].mean()):+.2f}" for gg in sorted(set(g))))
print("  T gate open share:", {s: round(float(((N[s]["X"] @ gT) > 0).double().mean()), 3) for s in N})
def stats(s, theta, K):
    Xs_, lp = N[s]["X"], N[s]["lp"]
    h = silu(Xs_ @ gT) * (Xs_ @ r8 - theta * (Xs_ @ q) / 11.9)
    p = torch.sigmoid(lo(lp) + 4.6 * d4["K"] * silu(Xs_ @ d4["gate"].double()) * (Xs_ @ d4["up"].double()) + cm * K * h)
    g = N[s]["group"]
    return float(torch.where(tgt[s], p, 1 - p).mean()), {gg: round(float(p[torch.tensor(g == gg)].mean()), 3) for gg in sorted(set(g))}
base = {s: stats(s, 0, 0) for s in N}
grid = [(t, k) for t in np.arange(-5.0, 2.01, 0.125) for k in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)]
def ok(g):
    st = {s: stats(s, *g)[1] for s in FIT}
    return (all(st[s]["call"] >= base[s][1]["call"] - 0.02 for s in FIT) and st["mA"]["text"] <= base["mA"][1]["text"] + 0.03
            and all(st[s]["swap"] <= base[s][1]["swap"] + 0.03 for s in ("tA", "cA")))
okg = [g for g in grid if ok(g)]
THETA, K = max(okg, key=lambda g: sum(stats(s, *g)[0] for s in FIT))
print("T8: THETA", THETA, "K", K, "| allowed", len(okg), "of", len(grid))
d7 = torch.load("design_T7.pt", weights_only=False)
def stats7(s):
    Xs_, lp = N[s]["X"], N[s]["lp"]; h = silu(Xs_ @ gT) * (Xs_ @ d7["up"].double())
    p = torch.sigmoid(lo(lp) + 4.6 * d4["K"] * silu(Xs_ @ d4["gate"].double()) * (Xs_ @ d4["up"].double()) + cm * d7["K"] * h)
    g = N[s]["group"]; return {gg: round(float(p[torch.tensor(g == gg)].mean()), 3) for gg in sorted(set(g))}
for s in N: print(f"  {s} think: base {base[s][1]} | T7 {stats7(s)} | T8 {stats(s, THETA, K)[1]}")
u = torch.load("design_A.pt", weights_only=False)["u"]; J2 = T["J2"]; up = r8 - THETA * q / 11.9
gn, un, dn = (f"model.language_model.layers.23.mlp.{t}_proj.weight" for t in ("gate", "up", "down"))
Wg, Wu, Wd = load(gn), load(un), load(dn)
Wg[J2] = gT.to(Wg.dtype); Wu[J2] = up.to(Wu.dtype); Wd[:, J2] = (K * u).to(Wd.dtype)
for name, W in ((gn, Wg), (un, Wu), (dn, Wd)): save_edit(name, W)
torch.save({"gate": gT.float(), "up": up.float(), "rT8": r8, "K": K, "theta": THETA, "J2": J2}, "design_T8.pt")
print("v9 written; changed weights", sum(int((load(n) != load_original(n)).sum()) for n in (gn, un, dn)))
