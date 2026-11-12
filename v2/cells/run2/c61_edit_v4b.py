# Run 2. Edit v4: BETA = 24 closes the gate in thinking mode. The larger gate in non-thinking mode scales h, so
# K is lowered by the ratio of the median non-thinking gate (v3 / v4): the same nothink effect as validated v3.
import json
P = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
def dec(p, n):
    z = np.load(P[p]["file"]); first = {}
    for k, (i, _) in enumerate(z["rows"]): first.setdefault(int(i), k)
    return torch.tensor(z["mlp_in_23"]).double()[[first[i] for i in range(n)]]
X = {"cA think": dec(31, 320), "mt_B think": dec(44, 256), "cA nothink": dec(32, 320), "mt_B nothink": dec(36, 256)}
mt = torch.load("mode_term.pt", weights_only=False); d = torch.load("design_v3.pt", weights_only=False)
q = torch.load("up_parts.pt", weights_only=False)["q"].double(); G = torch.load("gate_G.pt", weights_only=False).double()
silu = torch.nn.functional.silu
g3 = 2 * G + 0.5 * q
BETA = 24.0
g4 = g3 - BETA * (mt["m"] - mt["tau"] * q / 11.9)
up = d["r2"] - d["theta_final"] * q / 11.9
N = torch.cat([X["cA nothink"], X["mt_B nothink"]])
ratio = float(silu(N @ g3).median() / silu(N @ g4).median()); K = 0.5 * ratio
print(f"BETA {BETA}, K {K:.4f} (ratio {ratio:.3f})")
for k, v in X.items():
    pre = v @ g4; h3 = 0.5 * silu(v @ g3) * (v @ up); h4 = K * silu(pre) * (v @ up)
    print(f"  {k:12s} gate pre-act med {float(pre.median()):6.1f} (min {float(pre.min()):6.1f}, max {float(pre.max()):6.1f}) | share open (pre > 0) {float((pre > 0).double().mean()):.2f}"
          f" | mean |K h| v3 {float(h3.abs().mean()):.2f} v4 {float(h4.abs().mean()):.2f}")
J = 1488; u = torch.load("design_A.pt", weights_only=False)["u"]
gn, un, dn = (f"model.language_model.layers.23.mlp.{t}_proj.weight" for t in ("gate", "up", "down"))
Wg, Wu, Wd = load_original(gn), load_original(un), load_original(dn)
Wg[J] = g4.to(Wg.dtype); Wu[J] = up.to(Wu.dtype); Wd[:, J] = (K * u).to(Wd.dtype)
for name, W in ((gn, Wg), (un, Wu), (dn, Wd)): save_edit(name, W)
torch.save({"beta": BETA, "K": K, "gate": g4.float(), "up": up.float()}, "design_v4.pt")
print("edit v4 written; changed weights", sum(int((load(n) != load_original(n)).sum()) for n in (gn, un, dn)))
