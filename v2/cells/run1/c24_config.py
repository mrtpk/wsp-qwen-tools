import json
passes = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
G = torch.load("gate_G.pt", weights_only=False); P = torch.load("up_parts.pt", weights_only=False); C = torch.load("context_c.pt", weights_only=False)
q, r, th, sd = P["q"], P["r"], P["theta"], P["sd"]
def silu(z): return z * torch.sigmoid(z)
def h(X, delta, beta): return silu(2 * (X @ G) - beta * (X @ C["c"] - C["tau"] * (X @ q) / 11.9) + delta * (X @ q)) * (X @ r - th * (X @ q) / 11.9) / sd
D = torch.load("probes_A.pt", weights_only=False)
za = np.load(passes[20]["file"]); info = json.load(open("answer-probe-meta.json")); XA = torch.tensor(za["mlp_in_23"]); rowsA = za["rows"]
pre = []; firstA = set()
for i, m in enumerate(info):
    ks = [k for k in range(len(rowsA)) if rowsA[k][0] == i]; firstA.add(ks[0])
    if m["called"] and m["arm"] == "edited": pre.append(ks[-1])
ans = [k for k in range(len(rowsA)) if k not in firstA and k not in set(pre)]
zp = np.load(passes[22]["file"]); XP = torch.tensor(zp["mlp_in_23"])
for delta, beta in ((0.0, 1.33), (0.5, 1.33), (0.5, 2.33), (0.5, 2.0)):
    print(f"delta {delta} beta {beta}: pre-late veto mean {float(h(XA[pre], delta, beta).mean()):+.2f} | answer h>+1: {int((h(XA[ans], delta, beta) > 1).sum())}"
          f" | parallel 1st mean {float(h(XP[0::2], delta, beta).mean()):+.2f}, 2nd |max| {float(h(XP[1::2], delta, beta).abs().max()):.2f}")
    for n in D:
        X = D[n]["X"]; k = D[n]["kind"]; dec = torch.tensor(k == "dec")
        hd = h(X[dec], delta, beta); hb = h(X[~dec], delta, beta)
        print(f"     {n[6:]:13s} dec |h| mean {float(hd.abs().mean()):5.2f} | other positions h max+ {float(hb.max()):5.2f} min- {float(hb.min()):6.2f}")
