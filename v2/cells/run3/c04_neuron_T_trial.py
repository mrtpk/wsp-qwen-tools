# Run 3, cell 4. Second neuron "T" (thinking mode) in layer 23, on top of v4 (neuron 1488).
#   gate row: 2 G + 0.5 q + BETA (m - TAU q / 11.9)   (m: run-2 mode read; + sign = ON in thinking mode)
#   up row:   r_T - THETA q / 11.9                     r_T: joint thinking read (tA + mA, lam 3), scaled to sd 1
#   down col: K u
# Slot: the second least-used neuron of layer 23 (run-1 use measure). Then an edited probe request on tA + mA
# (decision positions) to calibrate K, as in run 2.
import json
D = torch.load("think_A.pt", weights_only=False)
def auroc(s, y):
    s = torch.as_tensor(s, dtype=torch.float64); y = torch.as_tensor(y).bool()
    order = torch.argsort(s); r = torch.empty_like(s); r[order] = torch.arange(1, len(s) + 1, dtype=torch.float64)
    n1, n0 = int(y.sum()), int((~y).sum()); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def lda(X1, X0, lam=3.0):
    S = torch.cov(torch.cat([X1 - X1.mean(0), X0 - X0.mean(0)]).T)
    S = S + lam * S.diagonal().mean() * torch.eye(S.shape[0], dtype=S.dtype)
    return torch.linalg.solve(S, X1.mean(0) - X0.mean(0))
X = torch.cat([D["tA"]["X"], D["mA"]["X"]]); y = torch.cat([D["tA"]["y"], D["mA"]["y"]])
rT = lda(X[y], X[~y]); s = X @ rT; sd = float(s.std()); rT = rT / sd; s = s / sd
cands = torch.sort(s).values
bal = [((s[y] > t).double().mean() + (s[~y] <= t).double().mean()) / 2 for t in cands]
THETA = float(cands[int(torch.tensor(bal).argmax())])
print("r_T in-sample AUROC", round(auroc(s, y), 3), "| tA", round(auroc(D["tA"]["X"] @ rT, D["tA"]["y"]), 3), "mA", round(auroc(D["mA"]["X"] @ rT, D["mA"]["y"]), 3), "| THETA", round(THETA, 3))
q = torch.load("up_parts.pt", weights_only=False)["q"].double(); G = torch.load("gate_G.pt", weights_only=False).double()
mt = torch.load("mode_term.pt", weights_only=False); BETA = 24.0
gate = 2 * G + 0.5 * q + BETA * (mt["m"] - mt["tau"] * q / 11.9); up = rT - THETA * q / 11.9
for k in ("tA", "mA"):
    pre = D[k]["X"] @ gate; print(f"  {k} think: gate pre-act median {float(pre.median()):.1f} min {float(pre.min()):.1f}, share open {float((pre > 0).double().mean()):.3f}")
use = torch.load("design_A.pt", weights_only=False)["use"]; order = torch.argsort(use)
J2 = int(order[1]) if int(order[0]) == 1488 else int(order[0])
print("slot J2", J2, "use", f"{float(use[J2]):.2e}", "(1488 use", f"{float(use[1488]):.2e})")
K0 = 0.18; u = torch.load("design_A.pt", weights_only=False)["u"]
gn, un, dn = (f"model.language_model.layers.23.mlp.{t}_proj.weight" for t in ("gate", "up", "down"))
Wg, Wu, Wd = load(gn), load(un), load(dn)
Wg[J2] = gate.to(Wg.dtype); Wu[J2] = up.to(Wu.dtype); Wd[:, J2] = (K0 * u).to(Wd.dtype)
for name, W in ((gn, Wg), (un, Wu), (dn, Wd)): save_edit(name, W)
print("v4 + T written; changed weights", sum(int((load(n) != load_original(n)).sum()) for n in (gn, un, dn)))
torch.save({"rT": rT, "theta": THETA, "beta": BETA, "gate": gate.float(), "up": up.float(), "J2": J2, "K0": K0}, "design_T.pt")
TC = tok.convert_tokens_to_ids("<tool_call>")
a, b = json.load(open("req-probe-tA-think.json")), json.load(open("req-probe-mA-think.json"))
json.dump({"type": "probe", "model": "edited", "texts": a["texts"] + b["texts"], "positions": [p[:1] for p in a["positions"] + b["positions"]],
           "mlp_in": [23], "mlp_act": [23], "token_ids": [TC]}, open("req-probe-A-think-edited-T.json", "w"))
print("edited probe rows", len(a["texts"]) + len(b["texts"]))
