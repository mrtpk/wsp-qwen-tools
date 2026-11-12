"""Find a residual direction for the query prior ("I can answer this myself").

Usage: uv run python prior_direction.py <prompts.jsonl> <acts.pt> <swap.jsonl> <acts-swap.pt>
The target is logit(p_call) on the swapped prompts, where no tool fits.
Ridge regression with 2-fold CV over the pairs.
"""

import json
import sys

import torch

from analyze_layers import auroc

LAYERS = [9, 13, 16, 20, 22, 24]
LAMBDA = 1e3
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5]


def unit(x):
    return x / x.norm()


def logit(p):
    p = p.double().clamp(1e-6, 1 - 1e-6)
    return (p / (1 - p)).log().float()


def ridge(X, y, lam=LAMBDA):
    mu, ym = X.mean(0), y.mean()
    Xc = X - mu
    w = torch.linalg.solve(Xc.T @ Xc + lam * torch.eye(X.shape[1]), Xc.T @ (y - ym))
    return w, mu, ym


def r2(pred, y):
    return 1 - ((y - pred) ** 2).sum().item() / ((y - y.mean()) ** 2).sum().item()


def corr(a, b):
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main(prompts, acts, swap, acts_swap):
    rows = [json.loads(line) for line in open(prompts)]
    A = torch.load(acts)
    need = torch.tensor([r["needs_call"] for r in rows])
    S = torch.load(acts_swap)
    own, sw = torch.arange(0, len(S["p_call"]), 2), torch.arange(1, len(S["p_call"]), 2)
    y_sw = logit(S["p_call"][sw])
    y_own = logit(S["p_call"][own])
    y_bfcl = logit(A["p_call"])
    half = len(sw) // 2
    tr, te = torch.arange(half), torch.arange(half, len(sw))
    print(f"{'hs':>3s} {'R2 swap(te)':>11s} {'prior->own corr':>15s} {'prior->BFCL need corr':>21s} "
          f"{'prior AUC need':>14s} {'cos(prior,rel)':>14s}")
    for L in LAYERS:
        Ysw = S["hs"][sw, L].float()
        Yown = S["hs"][own, L].float()
        tr1, tr2 = tr[: half // 2], tr[half // 2:]
        best = max(LAMBDAS, key=lambda lam: r2((Ysw[tr2] - ridge(Ysw[tr1], y_sw[tr1], lam)[1])
                                                @ ridge(Ysw[tr1], y_sw[tr1], lam)[0] + y_sw[tr1].mean(),
                                                y_sw[tr2]))
        w, mu, ym = ridge(Ysw[tr], y_sw[tr], best)
        pred_te = (Ysw[te] - mu) @ w + ym
        pred_own = (Yown[te] - mu) @ w + ym
        X = A["hs"][:, L].float()
        pred_bfcl = (X - mu) @ w + ym
        rel = unit(Yown.mean(0) - Ysw.mean(0))
        print(f"{L:3d} {r2(pred_te, y_sw[te]):11.3f} {corr(pred_own, y_own[te]):15.3f} "
              f"[lambda {best:g}] "
              f"{corr(pred_bfcl[need], y_bfcl[need]):21.3f} {auroc(pred_bfcl, need):14.3f} "
              f"{torch.dot(unit(w), rel):14.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
