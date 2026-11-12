K = 0.5
BETA = 1.33
# Shared cell body: write programmed neuron J of layer 23 with strength K (set K before this body).
# K = 0 writes the gate and up rows and a zero down column: the neuron is removed (slot-damage control).
J = 1488
G = torch.load("gate_G.pt", weights_only=False); P = torch.load("up_parts.pt", weights_only=False)
u = torch.load("design_A.pt", weights_only=False)["u"]
gn, un, dn = (f"model.language_model.layers.23.mlp.{t}_proj.weight" for t in ("gate", "up", "down"))
Wg, Wu, Wd = load_original(gn), load_original(un), load_original(dn)
Wg[J] = 2.0 * G
Wu[J] = (P["r"] - P["theta"] * P["q"] / 11.9) / P["sd"]
Wd[:, J] = K * u
for name, W in ((gn, Wg), (un, Wu), (dn, Wd)):
    save_edit(name, W)
changed = sum(int((load(n) != load_original(n)).sum()) for n in (gn, un, dn))
print("K", K, "edited", edited(), "changed weights", changed)
# Context term: switch the neuron off at multi-turn decision positions (c13, c14).
C = torch.load("context_c.pt", weights_only=False)
Wg = load(gn); Wg[J] = 2.0 * G - BETA * (C["c"] - C["tau"] * P["q"] / 11.9); save_edit(gn, Wg)
print("gate row with context term; changed weights", sum(int((load(n) != load_original(n)).sum()) for n in (gn, un, dn)))
