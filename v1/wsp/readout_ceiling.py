"""Compare the model readout p(<tool_call>) with a linear readout of the residual stream.

Usage: uv run python readout_ceiling.py <prompts.jsonl> <acts.pt>
For each layer, fit a direction with 5-fold CV. Set the threshold so that the
false-positive rate on the irrelevance prompts equals the model rate at
p_call > 0.5. Report the true-positive rate on the prompts that need a call.
"""

import json
import sys

import torch

from analyze_layers import auroc, cv_scores, fit_logreg, fit_meandiff

LAYERS = [9, 13, 16, 20, 24]


def tpr_at_fpr(score, need, fpr):
    neg = score[~need].sort(descending=True).values
    k = int(round(fpr * len(neg)))
    thr = neg[k] if k < len(neg) else neg[-1] - 1
    return (score[need] > thr).float().mean().item(), thr


def main(prompts, acts):
    rows = [json.loads(line) for line in open(prompts)]
    A = torch.load(acts)
    p = A["p_call"]
    need = torch.tensor([r["needs_call"] for r in rows])
    called = torch.tensor([r["called"] for r in rows])
    defect = need & ~called
    fpr = (p[~need] > 0.5).float().mean().item()
    tpr = (p[need] > 0.5).float().mean().item()
    print(f"model p_call>0.5: FPR(irrelevance) {fpr:.3f}  TPR(needs_call) {tpr:.3f}  AUROC {auroc(p, need):.3f}")
    print(f"sampled defect cases (needs call, no call): {int(defect.sum())}; "
          f"model p_call>0.5 on them: {(p[defect] > 0.5).float().mean():.3f}")
    print(f"{'hs':>3s} {'readout':>8s} {'AUROC':>6s} {'TPR@FPR':>8s} {'defect caught':>13s}")
    for L in LAYERS:
        X = A["hs"][:, L].float()
        for name, fit in [("meandiff", fit_meandiff), ("logreg", fit_logreg)]:
            s = cv_scores(X, need, fit)
            t, thr = tpr_at_fpr(s, need, fpr)
            caught = (s[defect] > thr).float().mean().item()
            print(f"{L:3d} {name:>8s} {auroc(s, need):6.3f} {t:8.3f} {caught:13.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
