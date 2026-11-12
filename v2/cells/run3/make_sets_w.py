"""Editor I/O, run 3 (V2-D13 extension): when2call "ask" sets. Seed 13.
wA (fit), wB (held-out): 256 "ask" rows each (single turn, a required value is missing; label no_call).
No row of an earlier set is used."""
import json, random, glob
from pathlib import Path
rng = random.Random(13); OUT = Path("cells/run3")
used = set()
for p in glob.glob("cells/run*/*.jsonl"):
    for l in open(p): used.add(json.loads(l)["id"].removesuffix("-swap"))
L = [json.loads(l) for l in open("practice/when2call.jsonl")]
ask = [r for r in L if r["detail"].get("subtype") == "ask" and r["id"] not in used]
seen, rows = set(), []
rng.shuffle(ask)
for r in ask:
    q = r["messages"][0]["content"].strip().lower()
    if q in seen: continue
    seen.add(q); rows.append({**r, "detail": {**r["detail"], "group": "ask"}})
for name, part in (("wA", rows[:256]), ("wB", rows[256:512])):
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in part: f.write(json.dumps(r) + "\n")
    print(name, len(part))
