import json
passes = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
D = torch.load("probes_A.pt", weights_only=False); index = json.load(open("cal-index.json"))
J = 1488
# Original values at the same rows/positions, from passes 5-8 (D), in calibration order.
orig_lp, kinds, labels, ctx = [], [], [], []
for name, i in index:
    m = D[name]["rid"] == i
    orig_lp.append(D[name]["lp"][m]); kinds += list(D[name]["kind"][m.numpy()]); labels += [bool(D[name]["label"][m][0])] * int(m.sum()); ctx += [name[6:]] * int(m.sum())
orig_lp = torch.cat(orig_lp); kinds = np.array(kinds); labels = torch.tensor(labels); ctx = np.array(ctx)
dec = torch.tensor(kinds == "dec")
for p, K in [(13, 0.1), (14, 0.25), (15, 0.5)]:
    z = np.load(passes[p]["file"]); lp = torch.tensor(z["logp"][:, 0]); h = torch.tensor(z["mlp_act_23"][:, J])
    d = lp - orig_lp
    print(f"K={K}: neuron h at dec |mean| {float(h[dec].abs().mean()):.3f}, elsewhere |mean| {float(h[~dec].abs().mean()):.4f} max {float(h[~dec].abs().max()):.3f}"
          f" | dlogp(tool_call) elsewhere: mean|.| {float(d[~dec].abs().mean()):.4f} max {float(d[~dec].abs().max()):.3f}")
    for c in ["st_A-think", "st_A-nothink", "mt_A-think", "mt_A-nothink"]:
        m = dec & torch.tensor(ctx == c)
        pe, po = lp[m].exp(), orig_lp[m].exp(); y = labels[m]
        print(f"   {c:13s} p_call desired-call {float(po[y].mean()):.3f}->{float(pe[y].mean()):.3f}  desired-no-call {float(po[~y].mean()):.3f}->{float(pe[~y].mean()):.3f}"
              f"  acc@0.5 {float(((po > 0.5) == y).float().mean()):.3f}->{float(((pe > 0.5) == y).float().mean()):.3f}")
