# Run 2, step 2 (c43: c37 with a ridge sweep, lam 0.01, 0.1, 1.0). On the clean set cA (one source), does a linear read beat the model's own p_call?
# 8-fold CV. Folds by question: a call row and its swap row are in the same fold.
import json
passes = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
meta = json.load(open("probe-meta-r2.json"))
r1 = torch.load("up_parts.pt")["r"].double()
def auroc(s, y):
    s = torch.as_tensor(s, dtype=torch.float64); y = torch.as_tensor(y).bool()
    order = torch.argsort(s); r = torch.empty_like(s); r[order] = torch.arange(1, len(s) + 1, dtype=torch.float64)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def lda(X1, X0, lam=1e-2):
    S = torch.cov(torch.cat([X1 - X1.mean(0), X0 - X0.mean(0)]).T)
    S = S + lam * S.diagonal().mean() * torch.eye(S.shape[0], dtype=S.dtype)
    return torch.linalg.solve(S, X1.mean(0) - X0.mean(0))
out = {}
for p, name in [(31, "probe-cA-think"), (32, "probe-cA-nothink")]:
    z = np.load(passes[p]["file"]); info = meta[name]; rows = z["rows"]
    first = {}
    for k, (i, _) in enumerate(rows): first.setdefault(int(i), k)
    idx = torch.tensor([first[i] for i in range(len(info))])
    y = torch.tensor([m["label"] == "call" for m in info]); g = np.array([m["group"] for m in info])
    fold = torch.tensor([(i % 128) % 8 if i < 256 else i % 8 for i in range(len(info))])
    lp = torch.tensor(z["logp"][:, 0]).double()[idx]
    res = {"model p_call": lp}
    res["run-1 r (mlp_in_23)"] = torch.tensor(z["mlp_in_23"]).double()[idx] @ r1
    for key in ["hidden_8", "hidden_12", "hidden_16", "hidden_20", "mlp_in_23"]:
        X = torch.tensor(z[key]).double()[idx]
        for lam in (1e-2, 1e-1, 1.0):
            s = torch.zeros(len(y), dtype=torch.float64)
            for f in range(8):
                te = fold == f; tr = ~te
                w = lda(X[tr & y], X[tr & ~y], lam); s[te] = X[te] @ w
            res[f"CV {key} lam {lam}"] = s
    sw = torch.tensor(g != "text"); tx = torch.tensor(g != "swap")
    print(f"{name}  (call {int(y.sum())}, swap {int((g=='swap').sum())}, text {int((g=='text').sum())})")
    print(f"  {'score':26s} all   call/swap  call/text")
    for k, s in res.items():
        print(f"  {k:26s} {auroc(s, y):.3f}  {auroc(s[sw], y[sw]):.3f}      {auroc(s[tx], y[tx]):.3f}")
    if "think" in name and "nothink" not in name:
        called = torch.tensor([bool(m["called"]) for m in info])
        for grp in ("call", "swap", "text"):
            m = torch.tensor(g == grp); print(f"  sampled call rate {grp}: {called[m].float().mean():.2f}")
    pc = lp.exp()
    for grp in ("call", "swap", "text"):
        m = torch.tensor(g == grp); print(f"  mean p_call {grp}: {pc[m].mean():.3f}")
    out[name] = {k: v for k, v in res.items()} | {"y": y, "group": g.tolist()}

