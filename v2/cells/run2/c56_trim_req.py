# Run 2. Trim the cB and gT generation requests to 256 rows (512 samples with both arms):
# the first 96 call rows, their 96 swap rows, and all 64 text rows. Writes the kept row indices.
import json
keep = list(range(96)) + list(range(128, 224)) + list(range(256, 320))
for name in ("cB", "gT"):
    for mode in (("think", "nothink") if name == "cB" else ("nothink",)):
        f = f"req-gen-{name}-{mode}-v3.json"; r = json.load(open(f))
        assert len(r["items"]) == 320
        r["items"] = [r["items"][i] for i in keep]; json.dump(r, open(f, "w")); print(f, len(r["items"]))
json.dump(keep, open("gen-v3-keep.json", "w"))
