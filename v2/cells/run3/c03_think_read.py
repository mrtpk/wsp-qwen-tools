# Run 3, cell 3. Thinking mode on fresh sets tA (single-turn) and mA (multi-turn).
# (1) Behavior, passes 1 and 2 (base vs v4). (2) Reads at the thinking decision position (passes 3, 4): CV LDA
# (8 folds: tA by question, mA by conversation) against the model's p_call. (3) The v4 gate in thinking mode.
import json
V2 = "@ROOT@/v2/"
P = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
meta = json.load(open("probe-meta-think.json"))
def auroc(s, y):
    s = torch.as_tensor(s, dtype=torch.float64); y = torch.as_tensor(y).bool()
    order = torch.argsort(s); r = torch.empty_like(s); r[order] = torch.arange(1, len(s) + 1, dtype=torch.float64)
    n1, n0 = int(y.sum()), int((~y).sum()); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def lda(X1, X0, lam):
    S = torch.cov(torch.cat([X1 - X1.mean(0), X0 - X0.mean(0)]).T)
    S = S + lam * S.diagonal().mean() * torch.eye(S.shape[0], dtype=S.dtype)
    return torch.linalg.solve(S, X1.mean(0) - X0.mean(0))
for p, s in ((1, "tA"), (2, "mA")):
    R = [json.loads(l) for l in open(V2 + f"cells/run3/{s}.jsonl")]; out = json.load(open(P[p]["file"]))
    for arm in ("original", "edited"):
        st = {}; ok = 0
        for r, o in zip(R, out[arm]):
            c = bool(o["samples"][0]["tool_calls"]); g = r["detail"]["group"]; a = st.setdefault(g, [0, 0]); a[0] += 1; a[1] += c
            ok += c == (r["label"] == "call")
        print(f"{s} think {arm:8s} decision acc {ok / len(R):.3f} | " + ", ".join(f"{k} call {v[1] / v[0]:.2f}" for k, v in st.items()))
d4 = torch.load("design_v4.pt", weights_only=False); silu = torch.nn.functional.silu
D = {}
for p, s in ((3, "tA"), (4, "mA")):
    z = np.load(P[p]["file"]); info = meta[s]; first = {}
    for k, (i, _) in enumerate(z["rows"]): first.setdefault(int(i), k)
    idx = [first[i] for i in range(len(info))]
    X = {k: torch.tensor(z[k]).double()[idx] for k in ("mlp_in_23", "hidden_16")}; lp = torch.tensor(z["logp"][:, 0]).double()[idx]
    y = torch.tensor([m["label"] == "call" for m in info])
    fold = torch.tensor([(i % 96) % 8 if i < 192 else i % 8 for i in range(len(info))]) if s == "tA" else \
           torch.tensor([int(m["id"].split("-")[1]) % 8 for m in info])
    print(f"{s} think: model p_call AUROC {auroc(lp, y):.3f}")
    for key in ("mlp_in_23", "hidden_16"):
        for lam in (1.0, 3.0, 10.0, 30.0):
            sc = torch.zeros(len(y), dtype=torch.float64)
            for f in range(8):
                te = fold == f; tr = ~te; sc[te] = X[key][te] @ lda(X[key][tr & y], X[key][tr & ~y], lam)
            print(f"   CV read {key:10s} lam {lam:5.1f} AUROC {auroc(sc, y):.3f}")
    pre = X["mlp_in_23"] @ d4["gate"].double()
    print(f"   v4 gate in thinking mode: share open {float((pre > 0).double().mean()):.3f}, mean |K h| {float((d4['K'] * silu(pre) * (X['mlp_in_23'] @ d4['up'].double())).abs().mean()):.3f}")
    g = np.array([m["group"] for m in info])
    for grp in sorted(set(g)): print(f"   mean p_call {grp}: {float(lp.exp()[torch.tensor(g == grp)].mean()):.3f}")
    D[s] = {"X": X["mlp_in_23"], "y": y, "lp": lp, "fold": fold, "group": g.tolist()}
torch.save(D, "think_A.pt")
