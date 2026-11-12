"""Layer analysis of the first-assistant-token residual stream.

Usage: uv run python analyze_layers.py <prompts.jsonl> <acts.pt> <model-dir> [subset]
subset: "all" (default) or "live" (live call categories against live_irrelevance).

hs[0] is the embedding output. hs[1..23] are the outputs of layers 0..22.
hs[24] is the output of layer 23 after the final norm.
"""

import json
import sys

import torch
from safetensors import safe_open

TOOL_CALL = 248058
FOLDS = 5


def auroc(s, y):
    """Mann-Whitney AUROC with average ranks for ties."""
    s, y = s.double(), y.bool()
    _, inv, counts = torch.unique(s, return_inverse=True, return_counts=True)
    ends = counts.cumsum(0).double()
    avg_rank = ends - (counts.double() - 1) / 2
    r = avg_rank[inv]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    return ((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


def cv_scores(X, y, fit):
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(y), generator=g)
    out = torch.empty(len(y))
    for k in range(FOLDS):
        te = perm[k::FOLDS]
        tr = torch.cat([perm[j::FOLDS] for j in range(FOLDS) if j != k])
        out[te] = fit(X[tr], y[tr], X[te])
    return out


def fit_meandiff(Xtr, ytr, Xte):
    d = Xtr[ytr].mean(0) - Xtr[~ytr].mean(0)
    return Xte @ d


def fit_logreg(Xtr, ytr, Xte, wd=1e-2, steps=200):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    w = torch.zeros(A.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")
    yt = ytr.float()

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(A @ w + b, yt) + wd * w.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return (B @ w + b).detach()


@torch.no_grad()
def logit_lens(hs, norm_w, W_U):
    out = []
    for L in range(hs.shape[1]):
        h = hs[:, L].float().cuda()
        if L < hs.shape[1] - 1:
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + norm_w)
        lp = torch.log_softmax(h @ W_U.T, -1)[:, TOOL_CALL]
        out.append(lp.cpu())
    return torch.stack(out, 1)


def main(prompts, acts, model_dir, subset="all"):
    rows = [json.loads(line) for line in open(prompts)]
    A = torch.load(acts)
    keep = list(range(len(rows)))
    if subset == "live":
        keep = [i for i, r in enumerate(rows) if r["category"].startswith("live_")]
    rows = [rows[i] for i in keep]
    hs = A["hs"][keep]
    p_call = A["p_call"][keep]
    need = torch.tensor([r["needs_call"] for r in rows])
    called = torch.tensor([r["called"] for r in rows])

    f = safe_open(f"{model_dir}/model.safetensors-00001-of-00001.safetensors", "pt", device="cuda")
    W_U = f.get_tensor("model.language_model.embed_tokens.weight").float()
    norm_w = f.get_tensor("model.language_model.norm.weight").float()
    ll = logit_lens(hs, norm_w, W_U)

    print(f"subset={subset} n={len(rows)} needs_call={int(need.sum())} irrelevance={int((~need).sum())}")
    print(f"model p_call AUROC: needs_call {auroc(p_call, need):.3f}  called {auroc(p_call, called):.3f}")
    print(f"{'hs':>3s} {'LL logp':>8s} {'LL AUC need':>11s} {'meandiff need':>13s} {'logreg need':>11s} {'logreg called':>13s}")
    for L in range(hs.shape[1]):
        X = hs[:, L].float()
        md = cv_scores(X, need, fit_meandiff)
        lr = cv_scores(X, need, fit_logreg)
        lc = cv_scores(X, called, fit_logreg)
        print(f"{L:3d} {ll[:, L].mean():8.2f} {auroc(ll[:, L], need):11.3f} {auroc(md, need):13.3f} "
              f"{auroc(lr, need):11.3f} {auroc(lc, called):13.3f}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
