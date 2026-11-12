# Run 2. Transfer between sources (non-thinking, layer-23 MLP input, lam 3).
# Fit on one source, test on the held-out set of the other source; and a joint fit on cA + mt_A.
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
def load(p, info):
    z = np.load(P[p]["file"]); first = {}
    for k, (i, _) in enumerate(z["rows"]): first.setdefault(int(i), k)
    idx = [first[i] for i in range(len(info))]
    return torch.tensor(z["mlp_in_23"]).double()[idx], torch.tensor([m["label"] == "call" for m in info])
mB = json.load(open("probe-meta-heldout.json"))
D = {"cA": load(32, json.load(open("probe-meta-r2.json"))["probe-cA-nothink"]), "mt_A": load(34, json.load(open("probe-meta-mtA-layers.json"))),
     "cB": load(35, mB["cB"]), "mt_B": load(36, mB["mt_B"])}
fits = {"cA": lda(D["cA"][0][D["cA"][1]], D["cA"][0][~D["cA"][1]]),
        "mt_A": lda(D["mt_A"][0][D["mt_A"][1]], D["mt_A"][0][~D["mt_A"][1]])}
# joint: pooled within-class scatter, mean difference averaged over the two sources (each source centred)
Xs, ys = [], []
for k in ("cA", "mt_A"):
    X, y = D[k]; Xs.append(X - X.mean(0)); ys.append(y)
X, y = torch.cat(Xs), torch.cat(ys); fits["joint"] = lda(X[y], X[~y])
for f, w in fits.items():
    print(f"fit {f:5s}: " + "  ".join(f"{t} {auroc(D[t][0] @ w, D[t][1]):.3f}" for t in ("cB", "mt_B")))
# A neuron applies one threshold to both contexts: AUROC on cB + mt_B pooled.
Xp = torch.cat([D["cB"][0], D["mt_B"][0]]); yp = torch.cat([D["cB"][1], D["mt_B"][1]])
print("pooled cB+mt_B, one threshold: joint read", round(auroc(Xp @ fits["joint"], yp), 3))
torch.save(fits, "reads_transfer.pt")
