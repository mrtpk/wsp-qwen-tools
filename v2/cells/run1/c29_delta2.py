import json
passes = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
G = torch.load("gate_G.pt", weights_only=False); P = torch.load("up_parts.pt", weights_only=False); C = torch.load("context_c.pt", weights_only=False)
q, r, th, sd = P["q"], P["r"], P["theta"], P["sd"]
def silu(z): return z * torch.sigmoid(z)
def h(X, delta, beta): return silu(2 * (X @ G) - beta * (X @ C["c"] - C["tau"] * (X @ q) / 11.9) + delta * (X @ q)) * (X @ r - th * (X @ q) / 11.9) / sd
D = torch.load("probes_A.pt", weights_only=False)
D = torch.load("probes_A.pt", weights_only=False)
za = np.load(passes[20]["file"]); info = json.load(open("answer-probe-meta.json")); XA = torch.tensor(za["mlp_in_23"]); rowsA = za["rows"]
pre = []; firstA = set()
for i, m in enumerate(info):
    ks = [k for k in range(len(rowsA)) if rowsA[k][0] == i]; firstA.add(ks[0])
    if m["called"] and m["arm"] == "edited": pre.append(ks[-1])
ans = [k for k in range(len(rowsA)) if k not in firstA and k not in set(pre)]
zp = np.load(passes[22]["file"]); XP = torch.tensor(zp["mlp_in_23"])
for delta in (0.5, 0.8, 1.0, 1.3):
    beta = 1.33 + 2 * delta
    st = torch.cat([h(D[n]["X"][torch.tensor(D[n]["kind"] == "dec")], delta, beta).abs() for n in D if "st_" in n]).mean()
    mt = torch.cat([h(D[n]["X"][torch.tensor(D[n]["kind"] == "dec")], delta, beta).abs() for n in D if "mt_" in n]).mean()
    K = 4.7 / float(st)
    other = torch.cat([h(D[n]["X"][torch.tensor(D[n]["kind"] != "dec")], delta, beta) for n in D])
    print(f"delta {delta} beta {beta:.2f}: K {K:.3f} | st dec {float(st)*K:.2f} mt dec {float(mt)*K:.2f} | pre-late veto {float(h(XA[pre], delta, beta).mean())*K:+.2f}"
          f" | answer max+ {float(h(XA[ans], delta, beta).max())*K:.2f} | parallel 2nd |max| {float(h(XP[1::2], delta, beta).abs().max())*K:.3f}"
          f" | other pos max+ {float(other.max())*K:.3f} min- {float(other.min())*K:.3f}")
