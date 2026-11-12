"""Summarize the direct logit attribution of logit(<tool_call>) - logit(best other token).

Usage: uv run python analyze_attribution.py <attr.pt> <model-dir>
Groups: need = needs a call and called; refuse = irrelevance, no call; over = irrelevance, called.
Part index 2*i is the token mixer of layer i, 2*i+1 is the MLP of layer i.
"""

import sys

import torch
from safetensors import safe_open

TOOL_CALL = 248058
FULL = {3, 7, 11, 15, 19, 23}


def name(j):
    i = j // 2
    if j % 2:
        return f"L{i:02d}.mlp"
    return f"L{i:02d}.{'attn' if i in FULL else 'gdn'}"


@torch.no_grad()
def main(attr, model_dir):
    A = torch.load(attr)
    f = safe_open(f"{model_dir}/model.safetensors-00001-of-00001.safetensors", "pt", device="cpu")
    W_U = f.get_tensor("model.language_model.embed_tokens.weight").float()
    g = 1 + f.get_tensor("model.language_model.norm.weight").float()
    d = (W_U[TOOL_CALL][None] - W_U[A["other"]]) * g            # [N, 1024]
    parts = A["parts"].float()                                  # [N, 48, 1024]
    dla = torch.einsum("npd,nd->np", parts, d) / A["rms"][:, None]
    emb = (A["emb"].float() * d).sum(-1) / A["rms"]
    total = dla.sum(1) + emb
    lp = A["p_call"].double().clamp(1e-9, 1 - 1e-9)
    print(f"check: corr(total DLA, logit p_call) = {torch.corrcoef(torch.stack([total.double(), (lp / (1 - lp)).log()]))[0, 1]:.3f}")
    need, called = A["needs_call"], A["called"]
    groups = {"need": need & called, "refuse": ~need & ~called, "over": ~need & called}
    for k, m in groups.items():
        print(f"{k:7s} n {int(m.sum()):4d}  mean total logit diff {total[m].mean():+6.2f}  emb {emb[m].mean():+.2f}")
    mean = {k: dla[m].mean(0) for k, m in groups.items()}

    def top(v, title, n=12):
        print(f"\n{title}")
        for j in v.abs().argsort(descending=True)[:n].tolist():
            print(f"  {name(j):9s} {v[j]:+7.3f}   (need {mean['need'][j]:+6.2f}, refuse {mean['refuse'][j]:+6.2f}, over {mean['over'][j]:+6.2f})")

    top((mean["need"] + mean["refuse"] + mean["over"]) / 3, "largest mean DLA (all groups)")
    top(mean["need"] - mean["refuse"], "largest difference need - refuse (relevance into the readout)")
    top(mean["over"] - mean["refuse"], "largest difference over - refuse (what drives over-calls)")
    by_layer = torch.stack([mean["over"] - mean["refuse"], mean["need"] - mean["refuse"]])
    print("\ncumulative over-refuse and need-refuse by layer (mixer+mlp):")
    cum = by_layer.view(2, 24, 2).sum(-1).cumsum(-1)
    print("  over-refuse:", " ".join(f"{x:+.1f}" for x in cum[0].tolist()))
    print("  need-refuse:", " ".join(f"{x:+.1f}" for x in cum[1].tolist()))
    torch.save({"dla": dla, "emb": emb, "groups": groups}, attr.replace(".pt", "-dla.pt"))


if __name__ == "__main__":
    main(*sys.argv[1:3])
