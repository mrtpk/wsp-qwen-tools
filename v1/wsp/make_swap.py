"""Make tool-swap prompt pairs: the same query with its own tools and with the tools of another entry.

Usage: uv run python make_swap.py <prompts.jsonl> <out.jsonl>
The swap partner comes from the same category and has no function name in common.
"""

import json
import random
import sys

CATS = ["simple_python", "multiple", "live_simple"]


def names(r):
    return {t["function"]["name"] for t in r["tools"]}


def main(src, out):
    rng = random.Random(0)
    rows = [json.loads(line) for line in open(src)]
    out_rows = []
    for cat in CATS:
        group = [r for r in rows if r["category"] == cat]
        for r in group:
            while True:
                o = rng.choice(group)
                if not names(o) & names(r):
                    break
            base = {"category": cat, "messages": r["messages"], "orig_called": r["called"]}
            out_rows.append({**base, "id": r["id"] + ":own", "tools": r["tools"], "needs_call": True})
            out_rows.append({**base, "id": r["id"] + ":swap", "tools": o["tools"], "needs_call": False,
                             "swap_from": o["id"]})
    with open(out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out_rows)} rows")


if __name__ == "__main__":
    main(*sys.argv[1:3])
