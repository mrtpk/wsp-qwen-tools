"""Summarize heads_and_patch.py output.

Usage: uv run python analyze_heads.py <heads-patch.json>
"""

import json
import sys
from collections import defaultdict

import torch

SEGMENTS = ["sys_head", "tools", "instructions", "user", "reasoning", "tail"]
FULL = ["3", "7", "11", "15", "19", "23"]


def main(path):
    res = json.load(open(path))
    groups = defaultdict(list)
    for r in res:
        groups[r["group"]].append(r)
    names = list(res[0]["patch"])
    print("mean patching (need/over rows shifted to the refuse mean; refuse rows shifted to the need mean)")
    print(f"{'patch':18s} " + " ".join(f"{g:>8s}" for g in groups))
    print(f"{'original':18s} " + " ".join(f"{sum(r['p'] for r in rs) / len(rs):8.3f}" for rs in groups.values()))
    for n in names:
        print(f"{n:18s} " + " ".join(f"{sum(r['patch'][n] for r in rs) / len(rs):8.3f}" for rs in groups.values()))

    dla = {g: torch.tensor([[r["head_dla"][l] for l in FULL] for r in rs]).mean(0) for g, rs in groups.items()}
    seg = {g: torch.tensor([[r["attn_seg"][l] for l in FULL] for r in rs]).mean(0) for g, rs in groups.items()}
    for title, diff in [("need - refuse", dla["need"] - dla["refuse"]), ("over - refuse", dla["over"] - dla["refuse"])]:
        print(f"\nhead DLA, largest {title}")
        flat = diff.flatten()
        for k in flat.abs().argsort(descending=True)[:8].tolist():
            li, h = divmod(k, diff.shape[1])
            print(f"  L{FULL[li]:>2s}.h{h}  {flat[k]:+6.3f}  (need {dla['need'][li, h]:+6.2f} refuse {dla['refuse'][li, h]:+6.2f} "
                  f"over {dla['over'][li, h]:+6.2f})")
            for g in groups:
                s = seg[g][li, h]
                print(f"      attn {g:6s} " + " ".join(f"{SEGMENTS[i]} {s[i]:.2f}" for i in range(len(SEGMENTS))))


if __name__ == "__main__":
    main(sys.argv[1])
