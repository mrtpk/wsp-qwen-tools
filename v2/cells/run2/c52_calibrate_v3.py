# Run 2. Calibrate edit v3. Pass 38 (edited, K0 = 0.35) against passes 32 and 34 (original), same positions.
# Measured: h (neuron 1488 activation) and the change of the <tool_call> log-odds. c = d_logodds / (K0 h).
# Offline: h for any theta and row from the layer-23 MLP input; new log-odds = old + c K h.
# Choose theta and K on A by expected decision accuracy; report the same on held-out B.
import json
P = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
d = torch.load("design_v3.pt", weights_only=False); r2, th0 = d["r2"], d["theta"]
q = torch.load("up_parts.pt", weights_only=False)["q"].double(); gate = d["gate"].double()
def load_dec(p, info):
    z = np.load(P[p]["file"]); first = {}
    for k, (i, _) in enumerate(z["rows"]): first.setdefault(int(i), k)
    idx = [first[i] for i in range(len(info))]
    return torch.tensor(z["mlp_in_23"]).double()[idx], torch.tensor([m["label"] == "call" for m in info]), torch.tensor(z["logp"][:, 0]).double()[idx]
mB = json.load(open("probe-meta-heldout.json"))
cA = load_dec(32, json.load(open("probe-meta-r2.json"))["probe-cA-nothink"]); mA = load_dec(34, json.load(open("probe-meta-mtA-layers.json")))
cB = load_dec(35, mB["cB"]); mB_ = load_dec(36, mB["mt_B"])
z = np.load(P[38]["file"]); h_meas = torch.tensor(z["mlp_act_23"][:, 1488]).double(); lp_e = torch.tensor(z["logp"][:, 0]).double()
lp_o = torch.cat([cA[2], mA[2][:192]]); X0 = torch.cat([cA[0], mA[0][:192]])
lo = lambda lp: lp - torch.log1p(-lp.exp().clamp(max=1 - 1e-9))
silu = torch.nn.functional.silu
def h_of(X, theta): return silu(X @ gate) * (X @ r2 - theta * (X @ q) / 11.9)
h_pred = h_of(X0, th0)
print("h predicted vs measured: corr", round(float(torch.corrcoef(torch.stack([h_pred, h_meas]))[0, 1]), 4),
      "mean |h|", round(float(h_meas.abs().mean()), 2))
c = (lo(lp_e) - lo(lp_o)) / (0.35 * h_meas)
ok = h_meas.abs() > 0.5
print("c = d_logodds / (K h): median", round(float(c[ok].median()), 3), "IQR", [round(float(v), 3) for v in torch.quantile(c[ok], torch.tensor([0.25, 0.75], dtype=c.dtype))])
cm = float(c[ok].median())
def acc(S, theta, K):
    X, y, lp = S; l2 = lo(lp) + cm * K * h_of(X, theta); p = torch.sigmoid(l2)
    return float(torch.where(y, p, 1 - p).mean()), float(p[y].mean()), float(p[~y].mean())
base = lambda S: acc(S, 0.0, 0.0)
grid = [(t, k) for t in np.arange(-1.0, 2.01, 0.25) for k in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0)]
score = lambda t, k: (acc(cA, t, k)[0] + acc(mA, t, k)[0]) / 2
t_best, k_best = max(grid, key=lambda g: score(*g))
print(f"chosen on A: theta {t_best:.2f}, K {k_best}")
for name, S in (("cA", cA), ("mt_A", mA), ("cB (held-out)", cB), ("mt_B (held-out)", mB_)):
    b, e = base(S), acc(S, t_best, k_best)
    print(f"  {name:16s} expected accuracy {b[0]:.3f} -> {e[0]:.3f} | p_call on call rows {b[1]:.2f} -> {e[1]:.2f} | on no-call rows {b[2]:.2f} -> {e[2]:.2f}")
torch.save({**d, "theta_final": float(t_best), "K_final": float(k_best), "c_median": cm}, "design_v3.pt")
