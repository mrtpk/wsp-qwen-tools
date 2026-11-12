# Run 2. The best gain from a constant shift b of the <tool_call> logit at the decision position.
# Expected decision accuracy: mean of p' (call rows) or 1 - p' (no-call rows); p' = sigmoid(logit(p) + b).
D = torch.load("clean_read_A.pt", weights_only=False)
for name, d in D.items():
    p = d["model p_call"].exp().clamp(1e-6, 1 - 1e-6); y = d["y"]; g = np.array(d["group"])
    lo = torch.log(p) - torch.log1p(-p)
    def acc(b):
        q = torch.sigmoid(lo + b); a = torch.where(y, q, 1 - q)
        return float(a.mean()), {k: round(float(q[torch.tensor(g == k)].mean()), 2) for k in ("call", "swap", "text")}
    best = max(np.arange(-6, 6.01, 0.25), key=lambda b: acc(b)[0])
    print(name, "b=0", acc(0)[0].__round__(3), acc(0)[1], "| best b", best, acc(best)[0].__round__(3), acc(best)[1])
