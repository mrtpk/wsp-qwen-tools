import json
passes = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
m = json.load(open("damage-meta.json")); starts, meta = m["starts"], m["meta"]
A, B = passes[28]["directory"], passes[29]["directory"]
lo, le = np.load(A + "/token_logprobs.npy", mmap_mode="r"), np.load(B + "/token_logprobs.npy", mmap_mode="r")
ko, ke = np.load(A + "/topk_logprobs.npy", mmap_mode="r"), np.load(B + "/topk_logprobs.npy", mmap_mode="r")
io, ie = np.load(A + "/topk_ids.npy", mmap_mode="r"), np.load(B + "/topk_ids.npy", mmap_mode="r")
lengths = passes[28]["lengths"]
groups = {}
for i, (st, mm) in enumerate(zip(starts, meta)):
    L = lengths[i]; sl = slice(st, L - 1)
    d = np.abs(le[i, sl] - lo[i, sl])
    # approximate KL(orig || edited) on the original top-20 support (ids missing in edited top-20 are skipped)
    kl = []
    for t in range(st, L - 1):
        po = np.exp(ko[i, t]); pe = dict(zip(ie[i, t].tolist(), ke[i, t].tolist()))
        common = [(p, lp, pe[j]) for p, lp, j in zip(po, ko[i, t], io[i, t].tolist()) if j in pe]
        kl.append(sum(p * (a - b) for p, a, b in common))
    top1 = (io[i, sl, 0] == ie[i, sl, 0]).mean()
    key = mm["set"] + ("-think" if mm["think"] else "-nothink")
    g = groups.setdefault(key, {"d": [], "kl": [], "top1": [], "max": 0.0})
    g["d"].append(d.mean()); g["kl"] += kl; g["top1"].append(top1); g["max"] = max(g["max"], float(d.max()))
for key, g in groups.items():
    kl = np.array(g["kl"])
    print(f"{key:13s} generated tokens {len(kl):6d}: mean |dlogp(obs)| {np.mean(g['d']):.4f}, max {g['max']:.3f}, approx KL mean {kl.mean():.5f} p99 {np.quantile(kl, 0.99):.4f}, top-1 agree {np.mean(g['top1']):.4f}")
