"""Editor I/O: pick practice rows and write generate requests. Seed 0. Fit rows and held-out rows are disjoint."""
import json, random
from pathlib import Path
P = Path("practice"); OUT = Path("cells/run1"); rng = random.Random(0)
def load(n): return [json.loads(l) for l in open(P / f"{n}.jsonl")]
w2c, ace, mt = load("when2call"), load("toolace"), load("apigen_mt")
cannot = [r for r in w2c if r["detail"]["subtype"] == "cannot"]
ask = [r for r in w2c if r["detail"]["subtype"] == "ask"]
call = [r for r in ace if r["label"] == "call" and len(r["messages"]) == 1]
mt_call = [r for r in mt if r["detail"]["step"] >= 1 and r["label"] == "call"]
mt_text = [r for r in mt if r["detail"]["step"] >= 1 and r["label"] == "no_call"]
for pool in (cannot, ask, call, mt_call, mt_text): rng.shuffle(pool)
# A: first sets (fit/analysis). B: held-out sets for validation later.
sets = {
  "st_A": cannot[:96] + ask[:32] + call[:128], "st_B": cannot[96:192] + ask[32:64] + call[128:256],
  "mt_A": mt_call[:128] + mt_text[:128], "mt_B": mt_call[128:256] + mt_text[128:256],
}
for name, rows in sets.items():
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(name, len(rows), sum(r["label"] == "call" for r in rows))
def req(rows, thinking):
    return {"type": "generate", "model": "original", "enable_thinking": thinking, "max_tokens": 8192,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "seed": 0,
            "items": [{"messages": r["messages"], "tools": r["tools"]} for r in rows]}
for s in ("st_A", "mt_A"):
    for t in (True, False):
        json.dump(req(sets[s], t), open(OUT / f"req-{s}-{'think' if t else 'nothink'}.json", "w"))
print(len(cannot), len(ask), len(call), len(mt_call), len(mt_text))
