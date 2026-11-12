# Run 3, cell 27. Edit v10 = v4 + neuron T11 (slot J2). Slot J3 (neuron N, v8/v9) back to the original weights.
# G32: in multi-turn, a main failure is a call where the reference asks for a missing value (miss_param, miss_func).
# r_T11 = LDA (lam 3) on thinking decisions: tA call vs tA swap, mA call vs mA text (as T7), plus wA (when2call "ask",
# pass 31) as "no call". tA text is still out of the fit (T6 lesson, G25/G32).
# THETA, K: best mean accuracy on tA (call, swap), mA, wA. Limits: tA call and mA call mean p_call fall by at most 0.02;
# tA text falls by at most 0.10 (a softer T7 limit: no v6-style refusal spread).
import json
T = torch.load("design_T.pt", weights_only=False); D = torch.load("think_A.pt", weights_only=False)
P = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
mw = json.load(open("probe-meta-think-wA.json"))["wA"]; z = np.load(P[31]["file"]); first = {}
for k, (i, _) in enumerate(z["rows"]): first.setdefault(int(i), k)
idx = [first[i] for i in range(len(mw))]; assert len(mw) == 256
D["wA"] = {"X": torch.tensor(z["mlp_in_23"]).double()[idx], "lp": torch.tensor(z["logp"][:, 0]).double()[idx],
           "y": torch.zeros(len(mw), dtype=torch.bool), "group": ["ask"] * len(mw)}
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
G = {k: np.array(D[k]["group"]) for k in ("tA", "mA", "wA")}
keep = {"tA": torch.tensor(G["tA"] != "text"), "mA": torch.ones(len(G["mA"]), dtype=torch.bool), "wA": torch.ones(256, dtype=torch.bool)}
X = torch.cat([D[k]["X"][keep[k]] for k in ("tA", "mA", "wA")]); y = torch.cat([D[k]["y"][keep[k]] for k in ("tA", "mA", "wA")])
r7 = lda(X[y], X[~y]); r7 = r7 / float((X @ r7).std())
d7 = torch.load("design_T7.pt", weights_only=False)
for k in ("tA", "mA", "wA"):
    s = D[k]["X"] @ r7
    print(f"  r_T11 on {k}: " + ", ".join(f"{g} mean {float(s[torch.tensor(G[k] == g)].mean()):+.2f}" for g in sorted(set(G[k]))))
    s6 = D[k]["X"] @ d7["rT7"]
    print(f"  r_T7 on {k}: " + ", ".join(f"{g} mean {float(s6[torch.tensor(G[k] == g)].mean()):+.2f}" for g in sorted(set(G[k]))))
print("  cos(r_T7, r_T11)", round(float(torch.nn.functional.cosine_similarity(d7["rT7"], r7, dim=0)), 3))
def stats(k, theta, K):
    Xs, yy, lp = D[k]["X"], D[k]["y"], D[k]["lp"]
    h = silu(Xs @ gT) * (Xs @ r7 - theta * (Xs @ q) / 11.9)
    p = torch.sigmoid(lo(lp) + 4.6 * d4["K"] * silu(Xs @ d4["gate"].double()) * (Xs @ d4["up"].double()) + cm * K * h)
    m = keep[k]
    return float(torch.where(yy, p, 1 - p)[m].mean()), {g: round(float(p[torch.tensor(G[k] == g)].mean()), 3) for g in sorted(set(G[k]))}
base = {k: stats(k, 0, 0) for k in ("tA", "mA", "wA")}
grid = [(t, kk) for t in np.arange(-2.0, 3.01, 0.125) for kk in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)]
def ok(g):
    a, b = stats("tA", *g)[1], stats("mA", *g)[1]
    return (a["call"] >= base["tA"][1]["call"] - 0.02 and b["call"] >= base["mA"][1]["call"] - 0.02
            and a["text"] >= base["tA"][1]["text"] - 0.10)
okg = [g for g in grid if ok(g)]
THETA, K = max(okg, key=lambda g: stats("tA", *g)[0] + stats("mA", *g)[0] + stats("wA", *g)[0])
print("T11: THETA", THETA, "K", K, "| grid points allowed", len(okg), "of", len(grid))
for k in ("tA", "mA", "wA"): print(f"  {k}: base {base[k][0]:.3f} {base[k][1]} -> {stats(k, THETA, K)[0]:.3f} {stats(k, THETA, K)[1]}")
u = torch.load("design_A.pt", weights_only=False)["u"]; J2 = T["J2"]; up = r7 - THETA * q / 11.9
gn, un, dn = (f"model.language_model.layers.23.mlp.{t}_proj.weight" for t in ("gate", "up", "down"))
Wg, Wu, Wd = load(gn), load(un), load(dn)
Wg[J2] = gT.to(Wg.dtype); Wu[J2] = up.to(Wu.dtype); Wd[:, J2] = (K * u).to(Wd.dtype)
J3 = torch.load("design_N.pt", weights_only=False)["J3"]; Og, Ou, Od = load_original(gn), load_original(un), load_original(dn)
Wg[J3] = Og[J3]; Wu[J3] = Ou[J3]; Wd[:, J3] = Od[:, J3]
for name, W in ((gn, Wg), (un, Wu), (dn, Wd)): save_edit(name, W)
torch.save({"gate": gT.float(), "up": up.float(), "rT11": r7, "K": K, "theta": THETA, "J2": J2}, "design_T11.pt")
print("v10 written; changed weights", sum(int((load(n) != load_original(n)).sum()) for n in (gn, un, dn)))
# Validation request (thinking, both arms): wB.
R = [json.loads(l) for l in open("@ROOT@/v2/cells/run3/wB.jsonl")]
r = json.load(open("req-gen-tB-think-v5.json")); r["items"] = [{"messages": x["messages"], "tools": x["tools"]} for x in R]
json.dump(r, open("req-gen-wB-think-v10.json", "w")); print("request wB think", len(r["items"]), r["model"])
