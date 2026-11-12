# Run 2. Edit v3 (trial strength). Neuron 1488 of layer 23, the run-1 mechanism with a clean read.
#   gate row: 2 G + 0.5 q        (decision positions; no multi-turn "off" term)
#   up row:   (r2 - theta q / 11.9) / sd   r2 = joint read (cA + mt_A, nothink, lam 3, not centred per source)
#   down col: K u
# Then an edited probe request on cA (320) + mt_A (first 192): the real change of log p(<tool_call>).
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
def load_dec(p, info):
    z = np.load(P[p]["file"]); first = {}
    for k, (i, _) in enumerate(z["rows"]): first.setdefault(int(i), k)
    idx = [first[i] for i in range(len(info))]
    return torch.tensor(z["mlp_in_23"]).double()[idx], torch.tensor([m["label"] == "call" for m in info]), torch.tensor(z["logp"][:, 0]).double()[idx]
mB = json.load(open("probe-meta-heldout.json"))
A = [load_dec(32, json.load(open("probe-meta-r2.json"))["probe-cA-nothink"]), load_dec(34, json.load(open("probe-meta-mtA-layers.json")))]
B = [load_dec(35, mB["cB"]), load_dec(36, mB["mt_B"])]
XA = torch.cat([a[0] for a in A]); yA = torch.cat([a[1] for a in A])
r2 = lda(XA[yA], XA[~yA])
s = XA @ r2; sd = float(s.std()); r2 = r2 / sd; s = s / sd
# theta: the threshold with the best balanced accuracy on A (pooled).
cands = torch.sort(s).values
bal = [((s[yA] > t).double().mean() + (s[~yA] <= t).double().mean()) / 2 for t in cands]
theta = float(cands[int(torch.tensor(bal).argmax())])
print("joint read r2: AUROC A", round(auroc(s, yA), 3), "| B single", round(auroc(B[0][0] @ r2, B[0][1]), 3),
      "multi", round(auroc(B[1][0] @ r2, B[1][1]), 3), "| theta", round(theta, 3))
F = torch.load("up_parts.pt", weights_only=False); q = F["q"].double(); G = torch.load("gate_G.pt", weights_only=False).double()
for name, (X, y, _) in zip(("cA", "mt_A"), A):
    qx = X @ q; gx = X @ (2 * G + 0.5 * q)
    print(f"  {name}: q.x mean {float(qx.mean()):.2f} CV {float(qx.std() / qx.mean()):.3f} | gate pre-act mean {float(gx.mean()):.2f} min {float(gx.min()):.2f}")
up = (r2 - theta * q / 11.9).float(); gate = (2 * G + 0.5 * q).float()
torch.save({"r2": r2, "theta": theta, "up": up, "gate": gate}, "design_v3.pt")
K = 0.35; J = 1488
u = torch.load("design_A.pt", weights_only=False)["u"]
gn, un, dn = (f"model.language_model.layers.23.mlp.{t}_proj.weight" for t in ("gate", "up", "down"))
Wg, Wu, Wd = load_original(gn), load_original(un), load_original(dn)
Wg[J] = gate.to(Wg.dtype); Wu[J] = up.to(Wu.dtype); Wd[:, J] = (K * u).to(Wd.dtype)
for name, W in ((gn, Wg), (un, Wu), (dn, Wd)): save_edit(name, W)
print("K", K, "changed weights", sum(int((load(n) != load_original(n)).sum()) for n in (gn, un, dn)))
# Edited probe request: same texts and decision positions as passes 32 and 34.
TC = tok.convert_tokens_to_ids("<tool_call>")
ra, rm = json.load(open("req-probe-cA-nothink.json")), json.load(open("req-probe-mtA-nothink-layers.json"))
texts = ra["texts"] + rm["texts"][:192]; pos = [p[:1] for p in ra["positions"]] + rm["positions"][:192]
json.dump({"type": "probe", "model": "edited", "texts": texts, "positions": pos, "mlp_in": [23], "mlp_act": [23], "token_ids": [TC]},
          open("req-probe-A-edited-v3.json", "w"))
print("request rows", len(texts))
