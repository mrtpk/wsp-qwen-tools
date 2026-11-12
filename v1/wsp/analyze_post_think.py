"""Analyze the decision state after </think> in thinking mode.

Usage: uv run python analyze_post_think.py <prompts-think.jsonl> <acts-post-think.pt> <prompts-nothink.jsonl> <acts-nothink.pt>
"""

import json
import sys

import torch

from analyze_layers import auroc, cv_scores, fit_logreg, fit_meandiff

LAYERS = [9, 13, 16, 20, 24]


def unit(x):
    return x / x.norm()


def main(prompts, acts, prompts_nt, acts_nt):
    rows = [json.loads(line) for line in open(prompts)]
    A = torch.load(acts)
    B = torch.load(acts_nt)
    need = torch.tensor([r["needs_call"] for r in rows])
    called = torch.tensor([r["called"] for r in rows])
    p = A["p_call"]
    print(f"n {len(rows)}; sampled called: needs {called[need].float().mean():.3f}, irrelevance {called[~need].float().mean():.3f}")
    print(f"p_call after </think>: AUROC called {auroc(p, called):.3f}, needs {auroc(p, need):.3f}")
    hist = torch.histc(p, bins=10, min=0, max=1).int().tolist()
    print(f"p_call histogram (10 bins): {hist}")
    agree = ((p > 0.5) == called).float().mean().item()
    print(f"agreement (p>0.5) with sampled called: {agree:.3f}")
    over = ~need & called
    print(f"over-calls (irrelevance, called): {int(over.sum())} of {int((~need).sum())}")
    print(f"{'hs':>3s} {'logreg need':>11s} {'meandiff need':>13s} {'NT rel dir -> need':>18s} "
          f"{'NT rel dir -> over vs correct-irr':>33s}")
    for L in LAYERS:
        X = A["hs"][:, L].float()
        Y = B["hs"][:, L].float()
        lr = cv_scores(X, need, fit_logreg)
        md = cv_scores(X, need, fit_meandiff)
        d_nt = unit(Y[need].mean(0) - Y[~need].mean(0))
        s = X @ d_nt
        irr = ~need
        print(f"{L:3d} {auroc(lr, need):11.3f} {auroc(md, need):13.3f} {auroc(s, need):18.3f} "
              f"{auroc(s[irr], called[irr]):33.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
