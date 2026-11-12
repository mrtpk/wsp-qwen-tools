# Run 2, step 2. Where does the base model's top-1 differ from the gold call (APIGen-MT, nothink)?
# Position t predicts token t+1. A span [a, b) is right when the top-1 matches at every token of the span.
import json
from collections import defaultdict
P = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}[33]
m = json.load(open("mt-tf-meta.json")); spans, meta = m["spans"], m["meta"]
ids = np.load(P["directory"] + "/input_ids.npy", mmap_mode="r")
top = np.load(P["directory"] + "/topk_ids.npy", mmap_mode="r")
lp = np.load(P["directory"] + "/token_logprobs.npy", mmap_mode="r")
ok = defaultdict(list); first_ok = defaultdict(list); lpsum = defaultdict(list); fname_lp = []
row_ok = []
for i, sp in enumerate(spans):
    all_ok = True
    for kind, a, b in sp:
        if a == b: continue  # empty value
        hit = [int(top[i, t - 1, 0]) == int(ids[i, t]) for t in range(a, b)]
        ok[kind].append(all(hit)); first_ok[kind].append(hit[0])
        lpsum[kind].append(float(sum(lp[i, t - 1] for t in range(a, b))))
        all_ok &= all(hit)
    row_ok.append(all_ok)
print(f"rows {len(spans)}; mean tools {np.mean([x['n_tools'] for x in meta]):.1f}; mean step {np.mean([x['step'] for x in meta]):.1f}")
for kind in ("decision", "fname", "pname", "value", "fixed"):
    print(f"  {kind:9s} spans {len(ok[kind]):4d}  top-1 whole span {np.mean(ok[kind]):.3f}  first token {np.mean(first_ok[kind]):.3f}"
          f"  mean log p {np.mean(lpsum[kind]):.2f}")
print(f"  whole call right (greedy): {np.mean(row_ok):.3f}")
# Given the right function name, how often are all values right?
vals = defaultdict(list)
for i, sp in enumerate(spans):
    v = [all(int(top[i, t - 1, 0]) == int(ids[i, t]) for t in range(a, b)) for k, a, b in sp if k == "value" and a < b]
    f = [all(int(top[i, t - 1, 0]) == int(ids[i, t]) for t in range(a, b)) for k, a, b in sp if k == "fname"][0]
    if v: vals[f].append(all(v))
print("  all values right | fname right:", round(np.mean(vals[True]), 3), len(vals[True]), " | fname wrong:", round(np.mean(vals[False]), 3) if vals[False] else None, len(vals[False]))
