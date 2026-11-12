"""Harness only: list the normalized BFCL user messages, for the practice-data leak filter."""
import glob, json, re, sys
D = sys.argv[1]
def norm(s): return re.sub(r"\W+", " ", s.lower()).strip()
out = set()
for f in glob.glob(D + "/BFCL_v4_*.json"):
    for line in open(f):
        try: x = json.loads(line)
        except json.JSONDecodeError: continue
        q = x.get("question")
        for turn in (q if isinstance(q, list) else []):
            for m in (turn if isinstance(turn, list) else [turn]):
                if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                    out.add(norm(m["content"]))
json.dump(sorted(out), open(sys.argv[2], "w"))
print(len(out))
