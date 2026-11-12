import json
passes = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
G = torch.load("gate_G.pt", weights_only=False); P = torch.load("up_parts.pt", weights_only=False); C = torch.load("context_c.pt", weights_only=False)
q, r, th, sd = P["q"], P["r"], P["theta"], P["sd"]
def silu(z): return z * torch.sigmoid(z)
def gate(X, delta): return 2 * (X @ G) - 1.33 * (X @ C["c"] - C["tau"] * (X @ q) / 11.9) + delta * (X @ q)
def up(X): return (X @ r - th * (X @ q) / 11.9) / sd
D = torch.load("probes_A.pt", weights_only=False)
z = np.load(passes[20]["file"]); info = json.load(open("answer-probe-meta.json")); XA = torch.tensor(z["mlp_in_23"]); rows = z["rows"]
pre = []
for i, m in enumerate(info):
    ks = [k for k in range(len(rows)) if rows[k][0] == i]
    if m["called"] and m["arm"] == "edited": pre.append(ks[-1])
pre = torch.tensor(pre); other = torch.tensor([k for k in range(len(rows)) if k not in set(pre.tolist())])
for delta in (0.0, 0.3, 0.5, 0.7):
    print(f"delta {delta}: pre-late-call silu*up mean {float((silu(gate(XA[pre], delta)) * up(XA[pre])).mean()):+.2f}  "
          f"answer-other |silu*up| max {float((silu(gate(XA[other], delta)) * up(XA[other])).abs().max()):.2f}")
    for n in D:
        X = D[n]["X"]; k = D[n]["kind"]
        for kind in ("dec", "bg", "call"):
            m = torch.tensor(k == kind); h = silu(gate(X[m], delta)) * up(X[m])
            pos = float(h.clamp(min=0).max()); neg = float(h.clamp(max=0).min())
            if kind == "dec": print(f"   {n[6:]:13s} dec h mean {float(h.mean()):+.2f}", end="")
            else: print(f" | {kind} h max+ {pos:.2f} min- {neg:.2f}", end="")
        print()
