# Run 2. Third-source test: reads fitted on ToolACE/APIGen-MT (c48), tested on glaive gT (pass 37). Non-thinking.
import json
P = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
def auroc(s, y):
    s = torch.as_tensor(s, dtype=torch.float64); y = torch.as_tensor(y).bool()
    order = torch.argsort(s); r = torch.empty_like(s); r[order] = torch.arange(1, len(s) + 1, dtype=torch.float64)
    n1, n0 = int(y.sum()), int((~y).sum()); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
info = json.load(open("probe-meta-gT.json")); z = np.load(P[37]["file"])
X = torch.tensor(z["mlp_in_23"]).double(); lp = torch.tensor(z["logp"][:, 0]).double()
y = torch.tensor([m["label"] == "call" for m in info]); g = np.array([m["group"] for m in info])
sw = torch.tensor(g != "text"); tx = torch.tensor(g != "swap")
fits = torch.load("reads_transfer.pt")
rows = {"model p_call": lp} | {f"read {k}": X @ w for k, w in fits.items()}
print(f"gT (glaive): call {int(y.sum())}, swap {int((g == 'swap').sum())}, text {int((g == 'text').sum())}")
print(f"  {'score':18s} all    call/swap  call/text")
for k, s in rows.items():
    print(f"  {k:18s} {auroc(s, y):.3f}  {auroc(s[sw], y[sw]):.3f}      {auroc(s[tx], y[tx]):.3f}")
for grp in ("call", "swap", "text"):
    print(f"  mean p_call {grp}: {lp.exp()[torch.tensor(g == grp)].mean():.3f}")
