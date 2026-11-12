"""Print the BFCL-V4 tables of REPORT.md from the score CSVs.
Usage: python3 analysis/report_tables.py [V1_CSV_DIR] [V2_CSV_DIR]
Defaults: results/v1-score-csv and results/v2-score-csv (copies of the scored runs).
After a new `evaluate`, pass v1/bfcl/score and v2/eval/score instead."""
import csv, sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
v1 = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "results/v1-score-csv"
v2 = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "results/v2-score-csv"
rows = {}
for d in (v1, v2):
    for r in csv.DictReader(open(d / "data_overall.csv")):
        rows[r["Model"]] = r

# Label -> BFCL model names per seed. "{m}" is the mode (think / nothink).
MODELS = {
    "base": ["Qwen3.5-0.8B base ({m} FC)", "qwen35-base-s2 ({m} FC)", "qwen35-base-s3 ({m} FC)"],
    "wsp (RikiyaT)": ["Qwen3.5-0.8B RikiyaT WSP ({m} FC)"],
    "v1 E1b": ["qwen35-e1b ({m} FC)", "qwen35-e1b-s2 ({m} FC)"],
    "v1 E4b": ["qwen35-e4b ({m} FC)", "qwen35-e4b-s2 ({m} FC)"],
    "v1 E5c": ["qwen35-e5c ({m} FC)"],
    "v2 run1 (wsp2)": ["qwen35-wsp2 ({m} FC)", "qwen35-wsp2-s2 ({m} FC)"],
    "v2 v4": ["qwen35-wsp4 ({m} FC)", "qwen35-wsp4-s2 ({m} FC)"],
    "v2 v6": ["qwen35-wsp6 ({m} FC)", "qwen35-wsp6-s2 ({m} FC)"],
    "v2 v7": ["qwen35-wsp7 ({m} FC)", "qwen35-wsp7-s2 ({m} FC)", "qwen35-wsp7-s3 ({m} FC)"],
    "v2 v8 (nothink only)": ["qwen35-wsp8 ({m} FC)", "qwen35-wsp8-s2 ({m} FC)"],
    "v2 v9 (think only)": ["qwen35-wsp9 ({m} FC)", "qwen35-wsp9-s2 ({m} FC)"],
    "v2 v10 (think only)": ["qwen35-wsp10 ({m} FC)", "qwen35-wsp10-s2 ({m} FC)"],
}
COLS = ["Overall Acc", "Non-Live AST Acc", "Live Acc", "Multi Turn Acc", "Memory Acc", "Relevance Detection",
        "Irrelevance Detection", "Multi Turn Base", "Multi Turn Miss Func", "Multi Turn Miss Param", "Multi Turn Long Context"]
val = lambda r, c: float(r[c].rstrip("%"))

for mode in ("think", "nothink"):
    print(f"\n=== {mode}: mean of the first two seeds (the report's two-seed tables); [seeds] ===")
    print(f"{'metric':26s}" + "".join(f"{k[:14]:>16s}" for k in MODELS))
    for c in COLS:
        line = f"{c:26s}"
        for names in MODELS.values():
            got = [rows[n.format(m=mode)] for n in names if n.format(m=mode) in rows]
            two = got[:2]
            line += f"{sum(val(r, c) for r in two) / len(two):16.2f}" if two else f"{'-':>16s}"
        print(line)
    print("seeds, overall:")
    for k, names in MODELS.items():
        seeds = [val(rows[n.format(m=mode)], "Overall Acc") for n in names if n.format(m=mode) in rows]
        if seeds:
            print(f"  {k:22s} {' / '.join(f'{x:.2f}' for x in seeds)}" + (f"  (3-seed mean {sum(seeds) / 3:.2f})" if len(seeds) == 3 else ""))
