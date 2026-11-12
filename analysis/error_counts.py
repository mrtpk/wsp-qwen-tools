"""Count BFCL single-turn error types (two seeds summed) for base, v7 and v10, thinking mode (G25, G35).
Needs the per-entry score files: run tools/score_all.sh first.
Usage: python3 analysis/error_counts.py"""
import collections, glob, json, os
from pathlib import Path

root = Path(__file__).resolve().parent.parent
v1, v2 = root / "v1/bfcl/score", root / "v2/eval/score"
dirs = {"base": [v1 / "qwen35-base-think-FC", v1 / "qwen35-base-s2-think-FC"],
        "v6": [v2 / "qwen35-wsp6-think-FC", v2 / "qwen35-wsp6-s2-think-FC"],
        "v7": [v2 / "qwen35-wsp7-think-FC", v2 / "qwen35-wsp7-s2-think-FC"],
        "v10": [v2 / "qwen35-wsp10-think-FC", v2 / "qwen35-wsp10-s2-think-FC"]}
for sub in ("non_live", "live"):
    for fn in sorted(os.path.basename(f) for f in glob.glob(str(dirs["v7"][0] / sub / "*.json"))):
        if "irrelevance" in fn:
            continue
        counts = {}
        for k, ds in dirs.items():
            c: collections.Counter[str] = collections.Counter()
            for d in ds:
                for line in open(d / sub / fn).read().splitlines()[1:]:
                    c[json.loads(line).get("error_type", "?").split(":")[0]] += 1
            counts[k] = c
        print(sub, fn)
        for et in sorted(set().union(*counts.values())):
            print(f"   {et:42s}" + "".join(f"{k}:{c[et]:5d}  " for k, c in counts.items()))
