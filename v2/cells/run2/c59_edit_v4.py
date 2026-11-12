# Run 2. Edit v4 = v3 + a thinking-mode "off" term in the gate.
# m = LDA (lam 3), think (cA pass 31 + mt_B pass 44) against nothink (cA pass 32 + mt_B pass 36), decision positions.
# CV: 8 folds (cA by question, mt_B by conversation). Gate row: 2 G + 0.5 q - BETA (m - TAU q / 11.9).
import json
P = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
def auroc(s, y):
    s = torch.as_tensor(s, dtype=torch.float64); y = torch.as_tensor(y).bool()
    order = torch.argsort(s); r = torch.empty_like(s); r[order] = torch.arange(1, len(s) + 1, dtype=torch.float64)
    n1, n0 = int(y.sum()), int((~y).sum()); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def lda(X1, X0, lam=3.0):
    S = torch.cov(torch.cat([X1 - X1.mean(0), X0 - X0.mean(0)]).T)
    S = S + lam * S.diagonal().mean() * torch.eye(S.shape[0], dtype=S.dtype)
    return torch.linalg.solve(S, X1.mean(0) - X0.mean(0))
def dec(p, n):
    z = np.load(P[p]["file"]); first = {}
    for k, (i, _) in enumerate(z["rows"]): first.setdefault(int(i), k)
    return torch.tensor(z["mlp_in_23"]).double()[[first[i] for i in range(n)]]
info = json.load(open("probe-meta-heldout.json"))["mt_B"]
conv = [int(x["id"].split("-")[1]) % 8 for x in info]
fc = [(i % 128) % 8 if i < 256 else i % 8 for i in range(320)]
X = {"cA think": dec(31, 320), "mt_B think": dec(44, 256), "cA nothink": dec(32, 320), "mt_B nothink": dec(36, 256)}
folds = {"cA think": fc, "mt_B think": conv, "cA nothink": fc, "mt_B nothink": conv}
Xall = torch.cat(list(X.values())); y = torch.cat([torch.full((len(v),), k.endswith(" think")) for k, v in X.items()])
fall = torch.tensor(sum((folds[k] for k in X), []))
s = torch.zeros(len(y), dtype=torch.float64)
for f in range(8):
    te = fall == f; tr = ~te; s[te] = Xall[te] @ lda(Xall[tr & y], Xall[tr & ~y])
print("mode read CV AUROC:", round(auroc(s, y), 4), "| mt_B only:", round(auroc(s[320:576].tolist() + s[896:].tolist(), [True] * 256 + [False] * 256), 4))
m = lda(Xall[y], Xall[~y]); sd = float((Xall @ m).std()); m = m / sd
for k, v in X.items():
    t = v @ m; print(f"  m.x {k:12s} min {float(t.min()):6.2f} median {float(t.median()):6.2f} max {float(t.max()):6.2f}")
sT = torch.cat([X["cA think"], X["mt_B think"]]) @ m; sN = torch.cat([X["cA nothink"], X["mt_B nothink"]]) @ m
TAU = float((torch.quantile(sT, 0.02) + torch.quantile(sN, 0.98)) / 2); print("TAU", round(TAU, 3))
d = torch.load("design_v3.pt", weights_only=False); q = torch.load("up_parts.pt", weights_only=False)["q"].double()
G = torch.load("gate_G.pt", weights_only=False).double()
for BETA in (4.0, 8.0, 16.0):
    g = 2 * G + 0.5 * q - BETA * (m - TAU * q / 11.9)
    print(f"BETA {BETA}: gate pre-act " + " | ".join(f"{k} med {float((v @ g).median()):.1f} (min {float((v @ g).min()):.1f}, max {float((v @ g).max()):.1f})" for k, v in X.items()))
BETA = 8.0
gate = (2 * G + 0.5 * q - BETA * (m - TAU * q / 11.9)).float()
silu = torch.nn.functional.silu
for k, v in X.items():
    h = silu(v @ gate.double()) * (v @ d["r2"] - d["theta_final"] * (v @ q) / 11.9)
    print(f"  h (v4) {k:12s} mean |h| {float(h.abs().mean()):.2f}")
torch.save({"m": m, "tau": TAU, "beta": BETA, "gate": gate}, "mode_term.pt")
K, J = 0.5, 1488; u = torch.load("design_A.pt", weights_only=False)["u"]
gn, un, dn = (f"model.language_model.layers.23.mlp.{t}_proj.weight" for t in ("gate", "up", "down"))
Wg, Wu, Wd = load_original(gn), load_original(un), load_original(dn)
Wg[J] = gate.to(Wg.dtype); Wu[J] = (d["r2"] - d["theta_final"] * q / 11.9).to(Wu.dtype); Wd[:, J] = (K * u).to(Wd.dtype)
for name, W in ((gn, Wg), (un, Wu), (dn, Wd)): save_edit(name, W)
print("edit v4 written; changed weights", sum(int((load(n) != load_original(n)).sum()) for n in (gn, un, dn)))
