# Run 3, cell 12 (c11 without intercept: a gate_proj row has no bias). Gate direction G_T: high only at thinking-mode decision positions.
# Ridge regression (layer-23 MLP input): target +4 at thinking decisions, -8 at all other positions.
# Positives: tA, mA thinking decisions (passes 3, 4); pass 10 thinking decisions (even rows).
# Negatives: nothink decisions (run-2 passes 32, 34, 35, 36), pass 10 inside-call / other positions and nothink
# decisions (even rows), background positions of run-2 passes 31 (think) and 32 (nothink).
# Test: pass 10 odd rows (all position kinds, both modes). Each group gets equal total weight.
import json
P = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
def dec_and_bg(path):
    z = np.load(path); X = torch.tensor(z["mlp_in_23"]).double(); first = {}
    for k, (i, _) in enumerate(z["rows"]): first.setdefault(int(i), k)
    d = torch.zeros(len(X), dtype=torch.bool); d[list(first.values())] = True
    return X[d], X[~d]
groups = {}
for p in (3, 4):
    groups[f"think dec p{p}"] = (dec_and_bg(P[p]["file"])[0], 4.0)
for p in (32, 34, 35, 36):
    groups[f"nothink dec r2p{p}"] = (dec_and_bg(f"imports/run2_pass{p}.npz")[0], -8.0)
groups["think bg r2p31"] = (dec_and_bg("imports/run2_pass31.npz")[1], -8.0)
groups["nothink bg r2p32"] = (dec_and_bg("imports/run2_pass32.npz")[1], -8.0)
info = json.load(open("probe-meta-offpos.json")); z = np.load(P[10]["file"]); X10 = torch.tensor(z["mlp_in_23"]).double()
kind, think, row = [], [], []
for i, inf in enumerate(info):
    n = 1 + inf["n_inside"] + 6
    kind += ["dec"] + ["inside"] * inf["n_inside"] + ["other"] * 6; think += [inf["think"]] * n; row += [i] * n
kind, think, row = np.array(kind), np.array(think), np.array(row)
even = row % 2 == 0
for kd in ("dec", "inside", "other"):
    for md in (True, False):
        m = torch.tensor((kind == kd) & (think == md) & even)
        groups[f"p10 {kd} {'think' if md else 'nothink'} even"] = (X10[m], 4.0 if (kd == "dec" and md) else -8.0)
Xs, ys, ws = [], [], []
for name, (Xg, t) in groups.items():
    Xs.append(Xg); ys.append(torch.full((len(Xg),), t, dtype=torch.float64)); ws.append(torch.full((len(Xg),), 1.0 / len(Xg), dtype=torch.float64))
    print(f"  fit group {name:24s} n {len(Xg):5d} target {t}")
X = torch.cat(Xs); y = torch.cat(ys); w = torch.cat(ws); w = w / w.mean()
Xa = X
A = (Xa * w[:, None]).T @ Xa; A += torch.eye(X.shape[1], dtype=X.dtype) * A.diagonal().mean() * 1e-2
g = torch.linalg.solve(A, (Xa * w[:, None]).T @ y); b = 0.0
# Fold the bias into the direction with q (q.x is about 11.9 at decision positions, smaller elsewhere) is not exact;
# keep the bias as a separate check here, and report g.x + b.
print("bias", round(b, 3))
def show(name, Xg):
    s = Xg @ g + b
    return f"{name:30s} n {len(Xg):5d}  min {float(s.min()):7.2f}  median {float(s.median()):7.2f}  max {float(s.max()):7.2f}"
for name, (Xg, t) in groups.items(): print("fit ", show(name, Xg))
odd = ~even
for kd in ("dec", "inside", "other"):
    for md in (True, False):
        m = torch.tensor((kind == kd) & (think == md) & odd); print("TEST", show(f"p10 {kd} {'think' if md else 'nothink'} odd", X10[m]))
torch.save({"g": g, "b": b}, "gate_T.pt")
