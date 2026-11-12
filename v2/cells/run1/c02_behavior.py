import json
V2 = "@ROOT@/v2/"
passes = [json.loads(l) for l in open("passes.jsonl")]
files = {p["pass"]: p["result"]["file"] for p in passes if p["status"] == "done"}
def rows(name): return [json.loads(l) for l in open(V2 + f"cells/run1/{name}.jsonl")]
def group(r):
    if r["source"] == "when2call": return r["detail"]["subtype"]
    return r["label"]
for p, setname, mode in [(1, "st_A", "think"), (2, "st_A", "nothink"), (3, "mt_A", "think"), (4, "mt_A", "nothink")]:
    out = json.load(open(files[p]))["original"]
    R = rows(setname)
    stats = {}
    for r, o in zip(R, out):
        s = o["samples"][0]; k = group(r)
        st = stats.setdefault(k, [0, 0, 0])
        st[0] += 1; st[1] += bool(s["tool_calls"])
        if r["label"] == "call" and s["tool_calls"]:
            st[2] += s["tool_calls"][0]["name"] == (r["detail"].get("name") or r["detail"]["reference"].strip("[").split("(")[0])
    print(setname, mode, {k: f"n={v[0]} call={v[1]/v[0]:.2f} right_name={v[2]/max(1,v[1]):.2f}" for k, v in stats.items()})
