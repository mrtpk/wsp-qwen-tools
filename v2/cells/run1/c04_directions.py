import json
passes = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
meta = json.load(open("probe-meta.json"))
def auroc(s, y):
    s = torch.as_tensor(s, dtype=torch.float64); y = torch.as_tensor(y).bool()
    r = torch.argsort(torch.argsort(s)).double() + 1
    # average ranks for ties
    vals, inv = torch.unique(s, return_inverse=True)
    for v in range(len(vals)):
        m = inv == v
        if m.sum() > 1: r[m] = r[m].mean()
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
D = {}
for p, name in [(5, "probe-st_A-think"), (6, "probe-st_A-nothink"), (7, "probe-mt_A-think"), (8, "probe-mt_A-nothink")]:
    z = np.load(passes[p]["file"]); info = meta[name]
    rows = z["rows"]; X = torch.tensor(z["mlp_in_23"]); lp = torch.tensor(z["logp"][:, 0])
    kind = []; label = []; called = []; rid = []
    starts = {}
    for k, (i, pos) in enumerate(rows):
        j = k - starts.setdefault(int(i), k)
        kind.append("dec" if j == 0 else "bg" if j <= info[i]["n_bg"] else "call")
        label.append(info[i]["label"] == "call"); called.append(info[i]["called"]); rid.append(int(i))
    D[name] = dict(X=X, lp=lp, kind=np.array(kind), label=torch.tensor(label), called=torch.tensor(called), rid=torch.tensor(rid))
    dec = D[name]["kind"] == "dec"
    print(name, "dec", int(dec.sum()), "bg", int((D[name]["kind"] == "bg").sum()), "call", int((D[name]["kind"] == "call").sum()),
          "| AUROC p_call vs sampled called", round(auroc(lp[dec], D[name]["called"][dec]), 3),
          "vs desired label", round(auroc(lp[dec], D[name]["label"][dec]), 3))
torch.save(D, "probes_A.pt")
