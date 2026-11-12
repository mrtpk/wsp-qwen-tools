"""Editor I/O, run 2: third-source test set gT from glaive (test only, never fitted). Seed 5.
Groups as in make_sets.py: call, swap (same question, tools of another call row with no shared name), text.
Questions are unique (glaive repeats many questions)."""
import json, random
from pathlib import Path
rng = random.Random(5); OUT = Path("cells/run2")
L = [json.loads(l) for l in open("practice/glaive.jsonl")]
rng.shuffle(L)
seen, call, text = set(), [], []
for r in L:
    q = r["messages"][0]["content"].strip().lower()
    if q in seen: continue
    seen.add(q); (call if r["label"] == "call" else text).append(r)
def names(r): return {t["function"]["name"] for t in r["tools"]}
def swap(r):
    while True:
        d = rng.choice(call)
        if d is not r and len(d["tools"]) == len(r["tools"]) and not names(d) & names(r):
            return {**r, "id": r["id"] + "-swap", "tools": d["tools"], "label": "no_call", "detail": {"group": "swap", "donor": d["id"]}}
def tag(r, g): return {**r, "detail": {**r["detail"], "group": g}}
rows = [tag(r, "call") for r in call[:128]] + [swap(r) for r in call[:128]] + [tag(r, "text") for r in text[:64]]
with open(OUT / "gT.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print("gT", len(rows), "unique call", len(call), "unique text", len(text))
