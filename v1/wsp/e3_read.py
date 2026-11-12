"""E3 read vector: a discriminant that is quiet at background positions.

rho = v + c1 * m_act + c2 * m_bg, with
  v  = (S_act + gamma * S_bg + eps * I)^-1 (mu_irr - mu_need)   (fit on even rows, action position)
  c1, c2 chosen so that rho . a = 0 at the need/over midpoint (action) and mean rho . a = 0 at background.
rho is scaled so that its over - need gap at the action position equals the E1 gap.

Usage: uv run python e3_read.py <sites.pt> <site-dirs.pt> [site]   (prints a table over gamma)
       imported by edit_e3.py
"""

import sys

import torch

from analyze_layers import auroc

SITE = "mlp23"
E1_TAU = -0.092


def stats_of(s, need, called):
    return {"need": s[need & called].mean().item(), "over": s[~need & called].mean().item(),
            "refuse": s[~need & ~called].mean().item()}


def build(S, gamma, site=SITE, eps=1e-3):
    X = S["last"][site].float()
    B = S["bg"][site].float()
    need, called = S["needs_call"], S["called"]
    idx = torch.arange(len(need))
    tr = idx % 2 == 0
    Xt, nt = X[tr], need[tr]
    mu_n, mu_i = Xt[nt].mean(0), Xt[~nt].mean(0)
    Xc = torch.cat([Xt[nt] - mu_n, Xt[~nt] - mu_i])
    S_act = Xc.T @ Xc / len(Xc)
    Bc = B - B.mean(0)
    S_bg = Bc.T @ Bc / len(Bc)
    S_act = S_act / S_act.diagonal().mean()
    S_bg = S_bg / S_bg.diagonal().mean()
    M = S_act + gamma * S_bg + eps * torch.eye(X.shape[1])
    v = torch.linalg.solve(M, mu_i - mu_n)
    m_act = X[tr].mean(0)
    m_act = m_act / m_act.norm()
    m_bg = B.mean(0)
    m_bg = m_bg / m_bg.norm()
    st = stats_of(X[tr] @ v, need[tr], called[tr])
    mid = (st["need"] + st["over"]) / 2
    A = torch.tensor([[(X[tr] @ m_act).mean(), (X[tr] @ m_bg).mean()],
                      [(B @ m_act).mean(), (B @ m_bg).mean()]], dtype=torch.float64)
    rhs = torch.tensor([-mid, -(B @ v).mean()], dtype=torch.float64)
    c = torch.linalg.solve(A, rhs).float()
    rho = v + c[0] * m_act + c[1] * m_bg
    st = stats_of(X[tr] @ rho, need[tr], called[tr])
    return rho / (st["over"] - st["need"]) * 0.196, float(torch.linalg.cond(A))


def evaluate(S, rho, site=SITE):
    X = S["last"][site].float()
    B = S["bg"][site].float()
    need, called = S["needs_call"], S["called"]
    te = torch.arange(len(need)) % 2 == 1
    s = X @ rho
    st = stats_of(s[te], need[te], called[te])
    pos = te & ~need & called
    neg = te & need & called
    a = auroc(torch.cat([s[pos], s[neg]]), torch.cat([torch.ones(int(pos.sum())), torch.zeros(int(neg.sum()))]).bool())
    b = B @ rho
    return st, a, b.mean().item(), b.std().item(), b.abs().quantile(0.99).item()


def e1_rho(S, dirs, site=SITE):
    X = S["last"][site].float()
    B = S["bg"][site].float()
    v = dirs[site].float()
    m = torch.cat([X, B]).mean(0)
    m_hat = m / m.norm()
    mu = (X @ m_hat).mean()
    return v - (E1_TAU / mu) * m_hat


def main(sites, dirs, site=SITE):
    S = torch.load(sites)
    D = torch.load(dirs)
    print(f"{'read':>10s} {'need':>7s} {'over':>7s} {'refuse':>7s} {'AUC over/need':>13s} {'bg mean':>8s} {'bg sd':>6s} {'bg |p99|':>8s} {'cond':>8s}")
    st, a, bm, bs, bq = evaluate(S, e1_rho(S, D, site), site)
    print(f"{'E1':>10s} {st['need']:7.3f} {st['over']:7.3f} {st['refuse']:7.3f} {a:13.3f} {bm:8.3f} {bs:6.3f} {bq:8.3f} {'':>8s}")
    for gamma in [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        rho, cond = build(S, gamma, site)
        st, a, bm, bs, bq = evaluate(S, rho, site)
        print(f"{'g=' + str(gamma):>10s} {st['need']:7.3f} {st['over']:7.3f} {st['refuse']:7.3f} {a:13.3f} {bm:8.3f} {bs:6.3f} {bq:8.3f} {cond:8.1f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
