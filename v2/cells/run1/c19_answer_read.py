import json
passes = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
z = np.load(passes[20]["file"]); info = json.load(open("answer-probe-meta.json"))
X = torch.tensor(z["mlp_in_23"]); lp = torch.tensor(z["logp"][:, 0]); rows = z["rows"]
G = torch.load("gate_G.pt", weights_only=False); P = torch.load("up_parts.pt", weights_only=False); C = torch.load("context_c.pt", weights_only=False)
q, r, th = P["q"], P["r"], P["theta"]
gate = 2 * (X @ G) - 1.33 * (X @ C["c"] - C["tau"] * (X @ q) / 11.9)
up = (X @ r - th * (X @ q) / 11.9) / P["sd"]
rr = X @ r; qq = X @ q
first = {}
for k, (i, p) in enumerate(rows):
    first.setdefault(int(i), k)
def show(title, idx):
    idx = torch.tensor(idx)
    print(f"{title:44s} n={len(idx):4d} q.x {float(qq[idx].mean()):6.2f}  gate z {float(gate[idx].mean()):6.2f} (max {float(gate[idx].max()):6.2f})"
          f"  r.x {float(rr[idx].mean()):5.2f}  p_call {float(lp[idx].exp().mean()):.3f}")
for arm in ("edited", "original"):
    for grp in ("cannot", "ask", "call"):
        pre, mid, dec = [], [], []
        for i, m in enumerate(info):
            if m["arm"] != arm or m["group"] != grp: continue
            ks = [k for k in range(len(rows)) if rows[k][0] == i]
            dec.append(ks[0])
            if m["called"]: pre.append(ks[-1]); mid += ks[1:-1]
            else: mid += ks[1:]
        if dec: show(f"{arm} {grp} decision pos", dec)
        if pre: show(f"{arm} {grp} position before late <tool_call>", pre)
        if mid: show(f"{arm} {grp} other answer positions", mid)
