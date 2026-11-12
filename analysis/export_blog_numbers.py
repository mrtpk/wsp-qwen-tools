"""Export the study numbers for the kaizen blog post to one JSON file.

Reads the score tables (results/v1-score-csv, results/v2-score-csv) and the v1 edit records (v1/edits/*.json).
For each model label and mode, it writes every seed with the overall score and the sub-scores of the report.
Usage: python3 analysis/export_blog_numbers.py <out.json>
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COLS = {"overall": "Overall Acc", "non_live_ast": "Non-Live AST Acc", "live": "Live Acc", "multi_turn": "Multi Turn Acc",
        "memory": "Memory Acc", "relevance": "Relevance Detection", "irrelevance": "Irrelevance Detection"}
MODELS = {
    "base": ["Qwen3.5-0.8B base", "qwen35-base-s2", "qwen35-base-s3"],
    "rikiyat": ["Qwen3.5-0.8B RikiyaT WSP"],
    "e1b": ["qwen35-e1b", "qwen35-e1b-s2"],
    "e4b": ["qwen35-e4b", "qwen35-e4b-s2"],
    "e5c": ["qwen35-e5c"],
    "run1": ["qwen35-wsp2", "qwen35-wsp2-s2"],
    "v4": ["qwen35-wsp4", "qwen35-wsp4-s2"],
    "v6": ["qwen35-wsp6", "qwen35-wsp6-s2"],
    "v7": ["qwen35-wsp7", "qwen35-wsp7-s2", "qwen35-wsp7-s3"],
    "v8": ["qwen35-wsp8", "qwen35-wsp8-s2"],
    "v9": ["qwen35-wsp9", "qwen35-wsp9-s2"],
    "v10": ["qwen35-wsp10", "qwen35-wsp10-s2"],
}


def main() -> None:
    rows = {}
    for d in ("results/v1-score-csv", "results/v2-score-csv"):
        for r in csv.DictReader(open(ROOT / d / "data_overall.csv")):
            rows[r["Model"]] = r
    out: dict = {"models": {}, "edits": {}}
    for label, names in MODELS.items():
        for mode in ("think", "nothink"):
            seeds = [{k: float(rows[f"{n} ({mode} FC)"][c].rstrip("%")) for k, c in COLS.items()}
                     for n in names if f"{n} ({mode} FC)" in rows]
            if seeds:
                out["models"].setdefault(label, {})[mode] = seeds
    for f in sorted((ROOT / "v1/edits").glob("*.json")):
        rec = json.loads(f.read_text())
        out["edits"][f.stem] = {t: {"shape": v["shape"], "changed_values": v["changed_values"],
                                    "relative_norm": round(v["relative_norm"], 4)} for t, v in rec.items()}
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(f"wrote {sys.argv[1]}: {len(out['models'])} models, {len(out['edits'])} v1 edits")


if __name__ == "__main__":
    main()
