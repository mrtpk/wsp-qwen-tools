"""Print a comparison table from score/data_overall.csv. Each model column is the mean of its samples.

Usage: .venv/bin/python compare_table.py <mode: think|nothink> <label>=<served>[+<served>...] [...]
Example: compare_table.py nothink base=Qwen3.5-0.8B base+qwen35-base-s2 E4b=qwen35-e4b+qwen35-e4b-s2 E5c=qwen35-e5c
"""

import csv
import sys

COLS = ["Overall Acc", "Non-Live AST Acc", "Live Acc", "Live Simple AST", "Live Multiple AST", "Multi Turn Acc",
        "Memory Acc", "Relevance Detection", "Irrelevance Detection"]


def main(mode, *specs):
    rows = {r["Model"]: r for r in csv.DictReader(open("score/data_overall.csv"))}
    cols = []
    for spec in specs:
        label, served = spec.split("=", 1)
        names = [f"{s} ({mode} FC)" for s in served.split("+")]
        missing = [n for n in names if n not in rows]
        assert not missing, f"not in data_overall.csv: {missing}"
        cols.append((f"{label} (n={len(names)})", names))
    print("measure | " + " | ".join(label for label, _ in cols))
    for c in COLS:
        vals = [sum(float(rows[n][c].rstrip("%")) for n in names) / len(names) for _, names in cols]
        print(f"{c} | " + " | ".join(f"{v:.2f}" for v in vals))


if __name__ == "__main__":
    main(*sys.argv[1:])
