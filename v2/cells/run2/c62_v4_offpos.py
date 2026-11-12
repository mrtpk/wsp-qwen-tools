# Run 2. Edit v4 at non-decision positions: gate pre-activation and K h at the background positions of passes 31, 32.
import json
P = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
d4 = torch.load("design_v4.pt", weights_only=False); g4, up, K = d4["gate"].double(), d4["up"].double(), d4["K"]
d3 = torch.load("design_v3.pt", weights_only=False); q = torch.load("up_parts.pt", weights_only=False)["q"].double()
G = torch.load("gate_G.pt", weights_only=False).double(); g3 = 2 * G + 0.5 * q
silu = torch.nn.functional.silu
for p in (31, 32):
    z = np.load(P[p]["file"]); X = torch.tensor(z["mlp_in_23"]).double(); first = {}
    for k, (i, _) in enumerate(z["rows"]): first.setdefault(int(i), k)
    bg = torch.ones(len(X), dtype=torch.bool); bg[list(first.values())] = False
    Xb = X[bg]; pre4 = Xb @ g4; h4 = (K * silu(pre4) * (Xb @ up)).abs(); h3 = (0.5 * silu(Xb @ g3) * (Xb @ up)).abs()
    print(f"pass {p}: background positions {int(bg.sum())} | v4 gate pre-act max {float(pre4.max()):.1f}, share > 0 {float((pre4 > 0).double().mean()):.3f}"
          f" | |K h| v4 mean {float(h4.mean()):.3f} p99 {float(torch.quantile(h4, 0.99)):.3f} max {float(h4.max()):.2f} | v3 mean {float(h3.mean()):.3f} max {float(h3.max()):.2f}")
