"""Editor I/O, run 3: fresh practice sets. No row (or swap source question) of an earlier set is used. Seed 3.

Single-turn (ToolACE, as run 2): call, swap, text.   tA (fit), tB (held-out): 96 call + 96 swap + 64 text = 256.
Multi-turn (APIGen-MT steps >= 1): call step, text step. mA, mB: 128 + 128 = 256.
Glaive (test only): gB: 96 call + 96 swap + 64 text = 256, questions not in gT.
256 rows = 512 samples with both arms.
"""
import json, random
from pathlib import Path
P = Path("practice"); OUT = Path("cells/run3"); OUT.mkdir(exist_ok=True); rng = random.Random(3)
def load(p): return [json.loads(l) for l in open(p)]
used = set()
for p in ["cells/run1/st_A.jsonl", "cells/run1/st_B.jsonl", "cells/run1/mt_A.jsonl", "cells/run1/mt_B.jsonl",
          "cells/run2/cA.jsonl", "cells/run2/cB.jsonl", "cells/run2/gT.jsonl"]:
    for r in load(p): used.add(r["id"].removesuffix("-swap"))
# run-2 teacher-forced rows (c39) are not in a set file; they are APIGen call steps. Exclude every row id seen in c39 meta.
used |= {m["id"] for m in json.load(open("work/mt-tf-meta.json"))["meta"]}
def names(r): return {t["function"]["name"] for t in r["tools"]}
def swaps(rows, pool):
    out = []
    for r in rows:
        while True:
            d = rng.choice(pool)
            if d is not r and len(d["tools"]) == len(r["tools"]) and not names(d) & names(r):
                out.append({**r, "id": r["id"] + "-swap", "tools": d["tools"], "label": "no_call", "detail": {"group": "swap", "donor": d["id"]}}); break
    return out
def tag(r, g): return {**r, "detail": {**r["detail"], "group": g}}
ace = [r for r in load(P / "toolace.jsonl") if len(r["messages"]) == 1 and r["id"] not in used]
call = [r for r in ace if r["label"] == "call"]; text = [r for r in ace if r["label"] == "no_call" and r["detail"]["reference"].strip()]
rng.shuffle(call); rng.shuffle(text)
sets = {}
for name, c, t in (("tA", call[:96], text[:64]), ("tB", call[96:192], text[64:128])):
    sets[name] = [tag(r, "call") for r in c] + swaps(c, call) + [tag(r, "text") for r in t]
mt = [r for r in load(P / "apigen_mt.jsonl") if r["detail"]["step"] >= 1 and r["id"] not in used]
mc = [r for r in mt if r["label"] == "call"]; mx = [r for r in mt if r["label"] == "no_call"]
rng.shuffle(mc); rng.shuffle(mx)
for name, c, x in (("mA", mc[:128], mx[:128]), ("mB", mc[128:256], mx[128:256])):
    sets[name] = [tag(r, "call") for r in c] + [tag(r, "text") for r in x]
gT_q = {r["messages"][0]["content"].strip().lower() for r in load("cells/run2/gT.jsonl")}
gl = [r for r in load(P / "glaive.jsonl") if r["id"] not in used]; rng.shuffle(gl)
seen, gc, gx = set(gT_q), [], []
for r in gl:
    q = r["messages"][0]["content"].strip().lower()
    if q in seen: continue
    seen.add(q); (gc if r["label"] == "call" else gx).append(r)
sets["gB"] = [tag(r, "call") for r in gc[:96]] + swaps(gc[:96], gc) + [tag(r, "text") for r in gx[:64]]
for name, rows in sets.items():
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(name, len(rows), "call", sum(r["label"] == "call" for r in rows))
