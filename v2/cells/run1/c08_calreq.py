# Calibration probe request: odd rows of the four A probes, the same texts and positions as passes 5 to 8.
import json
out = {"type": "probe", "model": "edited", "texts": [], "positions": [], "mlp_in": [23], "mlp_act": [23], "token_ids": [tok.convert_tokens_to_ids("<tool_call>")]}
index = []
for name in ["probe-st_A-think", "probe-st_A-nothink", "probe-mt_A-think", "probe-mt_A-nothink"]:
    req = json.load(open(f"req-{name}.json"))
    for i, (t, p) in enumerate(zip(req["texts"], req["positions"])):
        if i % 2 == 1:
            out["texts"].append(t); out["positions"].append(p); index.append([name, i])
json.dump(out, open("req-cal.json", "w")); json.dump(index, open("cal-index.json", "w"))
print(len(out["texts"]))
