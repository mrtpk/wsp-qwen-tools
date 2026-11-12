"""Fit an edit read direction v at each candidate site and measure its selectivity.

Usage: uv run python analyze_sites.py <sites.pt> <out-dirs.pt>
v separates irrelevance (1) from needs-a-call (0) at the action position, in the raw
activation space. Train on even rows, test on odd rows.
"""

import sys

import torch

from analyze_layers import auroc


def raw_direction(Xtr, ytr, wd=1e-2):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = (Xtr - mu) / sd
    w = torch.zeros(A.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")
    yt = ytr.float()

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(A @ w + b, yt) + wd * w.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    v = (w.detach() / sd)
    return v / v.norm()


def main(path, out):
    S = torch.load(path)
    need, called = S["needs_call"], S["called"]
    irr = ~need
    idx = torch.arange(len(need))
    tr, te = idx[idx % 2 == 0], idx[idx % 2 == 1]
    dirs = {}
    print(f"{'site':6s} {'AUC irr/need':>12s} {'AUC over/need':>13s} {'mean s: need':>12s} {'refuse':>7s} {'over':>7s} "
          f"{'bg mean':>8s} {'bg sd':>6s} {'bg p99':>7s} {'|a| last':>8s}")
    for key, X in S["last"].items():
        X = X.float()
        v = raw_direction(X[tr], irr[tr])
        s = X @ v
        bg = S["bg"][key].float() @ v
        t_need, t_irr = te[need[te]], te[irr[te]]
        t_over = te[irr[te] & called[te]]
        t_ref = te[irr[te] & ~called[te]]
        a1 = auroc(torch.cat([s[t_irr], s[t_need]]), torch.cat([torch.ones(len(t_irr)), torch.zeros(len(t_need))]).bool())
        a2 = auroc(torch.cat([s[t_over], s[te[need[te] & called[te]]]]),
                   torch.cat([torch.ones(len(t_over)), torch.zeros(int((need[te] & called[te]).sum()))]).bool())
        print(f"{key:6s} {a1:12.3f} {a2:13.3f} {s[t_need].mean():12.3f} {s[t_ref].mean():7.3f} {s[t_over].mean():7.3f} "
              f"{bg.mean():8.3f} {bg.std():6.3f} {bg.quantile(0.99):7.3f} {X.norm(dim=1).mean():8.2f}")
        dirs[key] = v
    torch.save(dirs, out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
