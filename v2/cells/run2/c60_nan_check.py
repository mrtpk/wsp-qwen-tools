# Run 2. Find the non-finite values in the probe outputs used by c59.
import json
P = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
for p in (31, 32, 36, 44):
    z = np.load(P[p]["file"]); X = z["mlp_in_23"]; bad = ~np.isfinite(X).all(1)
    print(p, X.shape, "non-finite rows", int(bad.sum()), "row ids", sorted(set(int(z["rows"][k][0]) for k in np.where(bad)[0]))[:10],
          "| logp non-finite", int((~np.isfinite(z["logp"])).sum()))
r = json.load(open("req-probe-mtB-think.json")); L = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in r["texts"]]
print("mt_B think lengths: max", max(L), "n > 32768:", sum(l > 32768 for l in L))
