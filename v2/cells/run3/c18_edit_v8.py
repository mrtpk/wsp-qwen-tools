# Run 3, cell 18. Edit v8 = v7 + neuron N (nothink decisions, slot J3). Convention V2-D11: a matching tool means "call".
# (1) Gate G_N: ridge without intercept at the layer-23 MLP input. Target +4 at nothink decisions, -8 at thinking
#     decisions, inside-call / other positions (both modes) and background positions. Equal weight per group.
#     Test: pass 10 odd rows.
# (2) Up read r_N: LDA (lam 3) on tA nothink, (call + text) vs swap (pass 19). sd 1.
# (3) THETA, K: best mean accuracy on tA (call, text -> call; swap -> no call) and mA (call -> call, text -> no call),
#     with v4 (neuron 1488) in the model of p_call. Limits: mA text mean p_call rises by at most 0.03;
#     mA call and tA call fall by at most 0.02. (Second run: theta grid -5 to 1; r_N.x is about -2 at decisions.)
import json
P = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
meta = json.load(open("probe-meta-nothink.json")); silu = torch.nn.functional.silu
def dec_and_bg(path):
    z = np.load(path); X = torch.tensor(z["mlp_in_23"]).double(); first = {}
    for k, (i, _) in enumerate(z["rows"]): first.setdefault(int(i), k)
    d = torch.zeros(len(X), dtype=torch.bool); d[list(first.values())] = True
    return X[d], X[~d]
N = {}
for p, s in ((19, "tA"), (20, "mA")):
    z = np.load(P[p]["file"]); assert len(z["rows"]) == len(meta[s]) == 256, (p, len(z["rows"]))
    N[s] = {"X": torch.tensor(z["mlp_in_23"]).double(), "lp": torch.tensor(z["logp"][:, 0]).double(),
            "y": torch.tensor([m["label"] == "call" for m in meta[s]]), "group": np.array([m["group"] for m in meta[s]])}
groups = {"nothink dec tA p19": (N["tA"]["X"], 4.0), "nothink dec mA p20": (N["mA"]["X"], 4.0)}
for p in (32, 34, 35, 36): groups[f"nothink dec r2p{p}"] = (dec_and_bg(f"imports/run2_pass{p}.npz")[0], 4.0)
for p in (3, 4): groups[f"think dec p{p}"] = (dec_and_bg(P[p]["file"])[0], -8.0)
groups["think bg r2p31"] = (dec_and_bg("imports/run2_pass31.npz")[1], -8.0)
groups["nothink bg r2p32"] = (dec_and_bg("imports/run2_pass32.npz")[1], -8.0)
info = json.load(open("probe-meta-offpos.json")); X10 = torch.tensor(np.load(P[10]["file"])["mlp_in_23"]).double()
kind, think, row = [], [], []
for i, inf in enumerate(info):
    n = 1 + inf["n_inside"] + 6
    kind += ["dec"] + ["inside"] * inf["n_inside"] + ["other"] * 6; think += [inf["think"]] * n; row += [i] * n
kind, think, row = np.array(kind), np.array(think), np.array(row); even = row % 2 == 0
for kd in ("dec", "inside", "other"):
    for md in (True, False):
        m = torch.tensor((kind == kd) & (think == md) & even)
        groups[f"p10 {kd} {'think' if md else 'nothink'} even"] = (X10[m], 4.0 if (kd == "dec" and not md) else -8.0)
Xs, ys, ws = [], [], []
for name, (Xg, t) in groups.items():
    Xs.append(Xg); ys.append(torch.full((len(Xg),), t, dtype=torch.float64)); ws.append(torch.full((len(Xg),), 1.0 / len(Xg), dtype=torch.float64))
X = torch.cat(Xs); y = torch.cat(ys); w = torch.cat(ws); w = w / w.mean()
A = (X * w[:, None]).T @ X; A += torch.eye(X.shape[1], dtype=X.dtype) * A.diagonal().mean() * 1e-2
gN = torch.linalg.solve(A, (X * w[:, None]).T @ y)
def show(name, Xg):
    s = Xg @ gN; return f"{name:28s} n {len(Xg):5d}  min {float(s.min()):7.2f}  median {float(s.median()):7.2f}  max {float(s.max()):7.2f}"
for name, (Xg, t) in groups.items(): print("fit ", show(name, Xg))
for kd in ("dec", "inside", "other"):
    for md in (True, False):
        m = torch.tensor((kind == kd) & (think == md) & ~even); print("TEST", show(f"p10 {kd} {'think' if md else 'nothink'} odd", X10[m]))
gate = 2 * gN
# Up read.
def lda(X1, X0, lam=3.0):
    S = torch.cov(torch.cat([X1 - X1.mean(0), X0 - X0.mean(0)]).T)
    S = S + lam * S.diagonal().mean() * torch.eye(S.shape[0], dtype=S.dtype)
    return torch.linalg.solve(S, X1.mean(0) - X0.mean(0))
gA = N["tA"]["group"]; match = torch.tensor(gA != "swap")
rN = lda(N["tA"]["X"][match], N["tA"]["X"][~match]); rN = rN / float((N["tA"]["X"] @ rN).std())
for s in ("tA", "mA"):
    sc = N[s]["X"] @ rN; g = N[s]["group"]
    print(f"  r_N on {s}: " + ", ".join(f"{gg} mean {float(sc[torch.tensor(g == gg)].mean()):+.2f}" for gg in sorted(set(g))))
T = torch.load("design_T.pt", weights_only=False); cm = T["c_median"]
d4 = torch.load("design_v4.pt", weights_only=False); q = torch.load("up_parts.pt", weights_only=False)["q"].double()
lo = lambda lp: lp - torch.log1p(-lp.exp().clamp(max=1 - 1e-9))
tgt = {"tA": torch.tensor(gA != "swap"), "mA": N["mA"]["y"]}
def stats(s, theta, K):
    Xs_, lp = N[s]["X"], N[s]["lp"]
    h = silu(Xs_ @ gate) * (Xs_ @ rN - theta * (Xs_ @ q) / 11.9)
    p = torch.sigmoid(lo(lp) + 4.6 * d4["K"] * silu(Xs_ @ d4["gate"].double()) * (Xs_ @ d4["up"].double()) + cm * K * h)
    g = N[s]["group"]
    return float(torch.where(tgt[s], p, 1 - p).mean()), {gg: round(float(p[torch.tensor(g == gg)].mean()), 3) for gg in sorted(set(g))}
pre = {s: N[s]["X"] @ gate for s in ("tA", "mA")}
print("  G_N gate on probes: share open tA", round(float((pre["tA"] > 0).double().mean()), 3), "mA", round(float((pre["mA"] > 0).double().mean()), 3))
base = {s: stats(s, 0, 0) for s in ("tA", "mA")}
grid = [(t, k) for t in np.arange(-5.0, 1.01, 0.125) for k in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)]
def ok(g):
    a, b = stats("tA", *g)[1], stats("mA", *g)[1]
    return (a["call"] >= base["tA"][1]["call"] - 0.02 and b["call"] >= base["mA"][1]["call"] - 0.02
            and b["text"] <= base["mA"][1]["text"] + 0.03)
okg = [g for g in grid if ok(g)]
THETA, K = max(okg, key=lambda g: stats("tA", *g)[0] + stats("mA", *g)[0])
print("N: THETA", THETA, "K", K, "| allowed", len(okg), "of", len(grid))
for s in ("tA", "mA"): print(f"  {s} nothink (v4 in both): v4 {base[s][0]:.3f} {base[s][1]} -> v8 {stats(s, THETA, K)[0]:.3f} {stats(s, THETA, K)[1]}")
up = rN - THETA * q / 11.9
kh = K * silu(X10 @ gate) * (X10 @ up)
for md in (True, False):
    for kd in ("dec", "inside", "other"):
        m = torch.tensor((kind == kd) & (think == md))
        print(f"  N {'think  ' if md else 'nothink'} {kd:6s}: open {float(((X10 @ gate)[m] > 0).double().mean()):.2f} mean |K h| {float(kh[m].abs().mean()):.3f} max {float(kh[m].abs().max()):.2f}")
use = torch.load("design_A.pt", weights_only=False)["use"]; order = [int(j) for j in torch.argsort(use)]
J3 = [j for j in order if j not in (1488, T["J2"])][0]; print("slot J3", J3)
u = torch.load("design_A.pt", weights_only=False)["u"]
gn, un, dn = (f"model.language_model.layers.23.mlp.{t}_proj.weight" for t in ("gate", "up", "down"))
Wg, Wu, Wd = load(gn), load(un), load(dn)
Wg[J3] = gate.to(Wg.dtype); Wu[J3] = up.to(Wu.dtype); Wd[:, J3] = (K * u).to(Wd.dtype)
for name, W in ((gn, Wg), (un, Wu), (dn, Wd)): save_edit(name, W)
torch.save({"gate": gate.float(), "up": up.float(), "rN": rN, "K": K, "theta": THETA, "J3": J3}, "design_N.pt")
print("v8 written; changed weights", sum(int((load(n) != load_original(n)).sum()) for n in (gn, un, dn)))
# Validation requests (nothink, both arms): mB and gB from the v5 thinking requests.
for s in ("mB", "gB"):
    r = json.load(open(f"req-gen-{s}-think-v5.json")); r["enable_thinking"] = False
    json.dump(r, open(f"req-gen-{s}-nothink-v8.json", "w")); print("request", s, "nothink", len(r["items"]))
