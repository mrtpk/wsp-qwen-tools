# Run 2. Fit the read on A (lam 3), test on held-out B. Non-thinking.
# Single-turn: fit cA (pass 32) -> test cB (pass 35). Multi-turn: fit mt_A (pass 34) -> test mt_B (pass 36).
import json
P = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
def auroc(s, y):
    s = torch.as_tensor(s, dtype=torch.float64); y = torch.as_tensor(y).bool()
    order = torch.argsort(s); r = torch.empty_like(s); r[order] = torch.arange(1, len(s) + 1, dtype=torch.float64)
    n1, n0 = int(y.sum()), int((~y).sum()); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def lda(X1, X0, lam):
    S = torch.cov(torch.cat([X1 - X1.mean(0), X0 - X0.mean(0)]).T)
    S = S + lam * S.diagonal().mean() * torch.eye(S.shape[0], dtype=S.dtype)
    return torch.linalg.solve(S, X1.mean(0) - X0.mean(0))
def decision_rows(p, n):
    z = np.load(P[p]["file"]); first = {}
    for k, (i, _) in enumerate(z["rows"]): first.setdefault(int(i), k)
    idx = [first[i] for i in range(n)]
    return {k: torch.tensor(z[k]).double()[idx] for k in ("mlp_in_23", "hidden_16")}, torch.tensor(z["logp"][:, 0]).double()[idx]
mA = json.load(open("probe-meta-r2.json"))["probe-cA-nothink"]; mtA = json.load(open("probe-meta-mtA-layers.json"))
mB = json.load(open("probe-meta-heldout.json"))
reads = {}
for name, (pa, ia, pb, ib) in {"single-turn": (32, mA, 35, mB["cB"]), "multi-turn": (34, mtA, 36, mB["mt_B"])}.items():
    XA, _ = decision_rows(pa, len(ia)); XB, lpB = decision_rows(pb, len(ib))
    yA = torch.tensor([m["label"] == "call" for m in ia]); yB = torch.tensor([m["label"] == "call" for m in ib])
    print(f"{name}: test rows {len(yB)} (call {int(yB.sum())})  model p_call AUROC {auroc(lpB, yB):.3f}")
    for key in ("mlp_in_23", "hidden_16"):
        w = lda(XA[key][yA], XA[key][~yA], 3.0)
        print(f"   read {key:10s} fit A, test B: AUROC {auroc(XB[key] @ w, yB):.3f}")
        reads[(name, key)] = w
    g = np.array([m["group"] for m in ib])
    if "swap" in g:
        sw = torch.tensor(g != "text"); tx = torch.tensor(g != "swap"); s = XB["mlp_in_23"] @ reads[(name, "mlp_in_23")]
        print(f"   by group, read: call/swap {auroc(s[sw], yB[sw]):.3f} call/text {auroc(s[tx], yB[tx]):.3f}"
              f" | model: call/swap {auroc(lpB[sw], yB[sw]):.3f} call/text {auroc(lpB[tx], yB[tx]):.3f}")
torch.save({f"{a}|{b}": w for (a, b), w in reads.items()}, "reads_A_lam3.pt")
