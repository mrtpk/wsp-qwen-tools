"""Test whether the relevance direction is a real "tool fits the query" signal.

Usage: uv run python analyze_swap.py <prompts.jsonl> <acts.pt> <swap.jsonl> <acts-swap.pt>
Rows of the swap file come in pairs: own tools, then swapped tools.
"""

import json
import sys

import torch

from analyze_layers import auroc, fit_logreg

LAYERS = [9, 13, 16, 20, 24]


def unit(x):
    return x / x.norm()


def main(prompts, acts, swap, acts_swap):
    rows = [json.loads(line) for line in open(prompts)]
    A = torch.load(acts)
    need = torch.tensor([r["needs_call"] for r in rows])
    srows = [json.loads(line) for line in open(swap)]
    S = torch.load(acts_swap)
    s_need = torch.tensor([r["needs_call"] for r in srows])
    own, sw = torch.arange(0, len(srows), 2), torch.arange(1, len(srows), 2)
    p = S["p_call"]
    print(f"pairs {len(own)}")
    print(f"model p_call: own {p[own].mean():.3f}  swap {p[sw].mean():.3f}  AUROC {auroc(p, s_need):.3f}  "
          f"pair win {(p[own] > p[sw]).float().mean():.3f}")
    print(f"model p_call>0.5 on swap (false call): {(p[sw] > 0.5).float().mean():.3f}")
    half = len(own) // 2
    tr_pairs, te_pairs = torch.arange(half), torch.arange(half, len(own))
    tr = torch.cat([own[tr_pairs], sw[tr_pairs]])
    te = torch.cat([own[te_pairs], sw[te_pairs]])
    print(f"{'hs':>3s} {'BFCL dir->swap AUC':>18s} {'pair win':>8s} {'swap dir->swap AUC':>18s} "
          f"{'swap dir->BFCL AUC':>18s} {'cos(BFCL,swap)':>14s} {'swap logreg AUC':>15s}")
    for L in LAYERS:
        X = A["hs"][:, L].float()
        Y = S["hs"][:, L].float()
        d_bfcl = unit(X[need].mean(0) - X[~need].mean(0))
        s1 = Y @ d_bfcl
        d_swap = unit(Y[tr][s_need[tr]].mean(0) - Y[tr][~s_need[tr]].mean(0))
        s2 = Y[te] @ d_swap
        s3 = X @ d_swap
        lr = fit_logreg(Y[tr], s_need[tr], Y[te])
        print(f"{L:3d} {auroc(s1, s_need):18.3f} {(s1[own] > s1[sw]).float().mean():8.3f} "
              f"{auroc(s2, s_need[te]):18.3f} {auroc(s3, need):18.3f} {torch.dot(d_bfcl, d_swap):14.3f} "
              f"{auroc(lr, s_need[te]):15.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
