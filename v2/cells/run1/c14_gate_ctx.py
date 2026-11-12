D = torch.load("probes_A.pt", weights_only=False); F = torch.load("fit_A.pt", weights_only=False)
G = torch.load("gate_G.pt", weights_only=False); C = torch.load("context_c.pt", weights_only=False); q = F["q"]
names = list(D)
def silu(z): return z * torch.sigmoid(z)
for beta in (0.0, 0.67, 1.0, 1.33):
    row = {}
    for n in names:
        X = D[n]["X"]; dec = torch.tensor(D[n]["kind"] == "dec")
        z = 2 * (X @ G) - beta * (X @ C["c"] - C["tau"] * (X @ q) / 11.9)
        row[n[6:]] = (round(float(silu(z[dec]).mean()), 2), round(float(z[~dec].max()), 2), round(float(silu(z[~dec]).abs().max()), 3),
                      round(float((z[~dec] > -3).float().mean()), 4))
    print("beta", beta, "(silu at dec mean, z elsewhere max, |silu| elsewhere max, share z>-3 elsewhere):", row)
