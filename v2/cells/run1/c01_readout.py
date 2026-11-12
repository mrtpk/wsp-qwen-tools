# Weight reading: direct write of each MLP neuron (down_proj column) to chosen token logits,
# as a z-score against the vocabulary spread. Final norm gain g = 1 + w (Qwen3.5 RMSNorm).
E = load("model.language_model.embed_tokens.weight")          # tied output layer, [V, 1024]
g = 1 + load("model.language_model.norm.weight")
ids = {t: tok.convert_tokens_to_ids(t) for t in ["<tool_call>", "</tool_call>", "<think>", "</think>", "<|im_end|>"]}
print(ids)
mu = E.mean(0); C = (E - mu).T @ (E - mu) / E.shape[0]
res = {}
for L in range(16, 24):
    D = load(f"model.language_model.layers.{L}.mlp.down_proj.weight")    # [1024, 3584]
    Gd = g[:, None] * D
    sd = torch.sqrt(torch.einsum("ij,ik,kj->j", Gd, C, Gd))
    m = mu @ Gd
    z = {t: ((E[i] @ Gd - m) / sd) for t, i in ids.items()}
    res[L] = z
    top = torch.topk(z["<tool_call>"].abs(), 5)
    print(L, "tool_call z: max", round(float(z["<tool_call>"].max()), 1), "min", round(float(z["<tool_call>"].min()), 1),
          "top |z| neurons", [(int(j), round(float(z["<tool_call>"][j]), 1)) for j in top.indices])
torch.save({L: {t: v.half() for t, v in z.items()} for L, z in res.items()}, "readout_z.pt")
