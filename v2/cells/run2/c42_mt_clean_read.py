# Run 2, step 2. Multi-turn decision (mt_A, nothink): does a linear read beat the model's p_call?
# 8-fold CV; folds by conversation (id apigen-<conversation>-<step>).
import json
P = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}[34]
info = json.load(open("probe-meta-mtA-layers.json")); z = np.load(P["file"])
y = torch.tensor([m["label"] == "call" for m in info])
conv = [int(m["id"].split("-")[1]) for m in info]; fold = torch.tensor([c % 8 for c in conv])
def auroc(s, y):
    s = torch.as_tensor(s, dtype=torch.float64); y = torch.as_tensor(y).bool()
    order = torch.argsort(s); r = torch.empty_like(s); r[order] = torch.arange(1, len(s) + 1, dtype=torch.float64)
    n1, n0 = int(y.sum()), int((~y).sum()); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def lda(X1, X0, lam=1e-2):
    S = torch.cov(torch.cat([X1 - X1.mean(0), X0 - X0.mean(0)]).T)
    S = S + lam * S.diagonal().mean() * torch.eye(S.shape[0], dtype=S.dtype)
    return torch.linalg.solve(S, X1.mean(0) - X0.mean(0))
print("rows", len(y), "call", int(y.sum()), "conversations", len(set(conv)))
print(f"  model p_call            {auroc(torch.tensor(z['logp'][:, 0]), y):.3f}")
for key in ["hidden_8", "hidden_12", "hidden_16", "hidden_20", "mlp_in_23"]:
    X = torch.tensor(z[key]).double(); s = torch.zeros(len(y), dtype=torch.float64)
    for lam in (1e-2, 1e-1, 1.0):
        for f in range(8):
            te = fold == f; tr = ~te
            s[te] = X[te] @ lda(X[tr & y], X[tr & ~y], lam)
        print(f"  CV read {key:10s} lam {lam:<5} {auroc(s, y):.3f}")
