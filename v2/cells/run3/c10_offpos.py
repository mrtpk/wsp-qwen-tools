# Run 3, cell 10. Gate pre-activation and K h of three neuron designs at decision / inside-call / other positions
# (pass 10). v3: decision gate only. v4: decision - 24 m. T: decision + 24 m.
import json
P = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
info = json.load(open("probe-meta-offpos.json")); z = np.load(P[10]["file"]); X = torch.tensor(z["mlp_in_23"]).double()
silu = torch.nn.functional.silu; q = torch.load("up_parts.pt", weights_only=False)["q"].double()
d3 = torch.load("design_v3.pt", weights_only=False); d4 = torch.load("design_v4.pt", weights_only=False); T = torch.load("design_T.pt", weights_only=False)
G = torch.load("gate_G.pt", weights_only=False).double()
designs = {"v3": (2 * G + 0.5 * q, d3["r2"] - d3["theta_final"] * q / 11.9, 0.5),
           "v4": (d4["gate"].double(), d4["up"].double(), d4["K"]),
           "T": (T["gate"].double(), T["rT"] - T["theta_final"] * q / 11.9, T["K_final"])}
kind = []; think = []; k = 0
for i, inf in enumerate(info):
    n = 1 + inf["n_inside"] + 6
    kind += ["decision"] + ["inside call"] * inf["n_inside"] + ["other"] * 6; think += [inf["think"]] * n
assert len(kind) == len(X), (len(kind), len(X))
kind = np.array(kind); think = np.array(think)
for name, (g, up, K) in designs.items():
    pre = X @ g; kh = K * silu(pre) * (X @ up)
    for md in (True, False):
        for kd in ("decision", "inside call", "other"):
            m = torch.tensor((kind == kd) & (think == md))
            print(f"{name:3s} {'think  ' if md else 'nothink'} {kd:11s}: open {float((pre[m] > 0).double().mean()):.2f}  mean |K h| {float(kh[m].abs().mean()):6.3f}  max |K h| {float(kh[m].abs().max()):6.2f}")
