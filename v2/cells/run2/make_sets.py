"""Editor I/O, run 2: clean single-turn sets. Every row comes from ToolACE (one source, one format). Seed 2.

Groups:
  call: a ToolACE row whose reference answer is a call.
  swap: the same question with the tools of another call row (same tool count, no shared name). No tool fits.
  text: a ToolACE row whose reference answer is text (mostly a question for a missing value). Empty
        references are dropped (label not certain).
A call row and its swap row share the question, so they are always in the same set.
A: analysis and fit. B: held-out validation.
"""
import json, random
from collections import defaultdict
from pathlib import Path
P = Path("practice"); OUT = Path("cells/run2"); rng = random.Random(2)
ace = [json.loads(l) for l in open(P / "toolace.jsonl")]
call = [r for r in ace if r["label"] == "call" and len(r["messages"]) == 1]
text = [r for r in ace if r["label"] == "no_call" and len(r["messages"]) == 1 and r["detail"]["reference"].strip()]
rng.shuffle(call); rng.shuffle(text)
by_n = defaultdict(list)
for r in call: by_n[len(r["tools"])].append(r)
def names(r): return {t["function"]["name"] if "function" in t else t["name"] for t in r["tools"]}
def swap(r):
    while True:
        d = rng.choice(by_n[len(r["tools"])])
        if d is not r and not names(d) & names(r):
            return {**r, "id": r["id"] + "-swap", "tools": d["tools"], "label": "no_call",
                    "detail": {"group": "swap", "donor": d["id"]}}
def tag(r, g): return {**r, "detail": {**r["detail"], "group": g}}
sets = {}
for name, q, t in (("cA", call[:128], text[:64]), ("cB", call[128:256], text[64:128])):
    sets[name] = [tag(r, "call") for r in q] + [swap(r) for r in q] + [tag(r, "text") for r in t]
for name, rows in sets.items():
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(name, len(rows), sum(r["label"] == "call" for r in rows))
req = {"type": "generate", "model": "original", "enable_thinking": True, "max_tokens": 8192,
       "temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "seed": 0,
       "items": [{"messages": r["messages"], "tools": r["tools"]} for r in sets["cA"]]}
json.dump(req, open(OUT / "req-cA-think.json", "w"))
print(len(call), len(text))
