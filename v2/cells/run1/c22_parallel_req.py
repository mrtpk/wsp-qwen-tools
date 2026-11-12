# Parallel-call positions: ToolACE rows whose reference has 2+ calls "[A(...), B(...)]". Not used in any fit.
import json, random, re
V2 = "@ROOT@/v2/"
used = {json.loads(l)["id"] for f in ("st_A", "st_B") for l in open(V2 + f"cells/run1/{f}.jsonl")}
rows = [json.loads(l) for l in open(V2 + "practice/toolace.jsonl")]
def calls(ref):
    body = ref.strip()[1:-1]
    return [m.group(1) for m in re.finditer(r"(?:^|,\s*)([A-Za-z_][\w .-]*?)\(", body)]
par = [r for r in rows if r["id"] not in used and r["label"] == "call" and len(r["messages"]) == 1 and len(calls(r["detail"]["reference"])) >= 2]
random.Random(2).shuffle(par); par = par[:128]
TC = tok.convert_tokens_to_ids("<tool_call>")
texts, positions = [], []
for think in (False, True):
    for r in par:
        names = calls(r["detail"]["reference"])
        p = tok.apply_chat_template(r["messages"], tools=r["tools"], add_generation_prompt=True, tokenize=False, enable_thinking=think)
        if think: p += "The user needs several calls. I will call the tools.\n</think>\n\n"
        first = f"<tool_call>\n<function={names[0]}>\n</function>\n</tool_call>\n"
        text = p + first + f"<tool_call>\n<function={names[1]}>\n</function>\n</tool_call>"
        ids = tok(text, add_special_tokens=False)["input_ids"]
        n0 = len(tok(p, add_special_tokens=False)["input_ids"])
        k = [j for j, t in enumerate(ids) if t == TC and j >= n0]
        assert len(k) == 2, k
        texts.append(text); positions.append([k[0] - 1, k[1] - 1])     # before 1st and before 2nd <tool_call>
json.dump({"type": "probe", "model": "original", "texts": texts, "positions": positions, "mlp_in": [23], "token_ids": [TC]}, open("req-parallel.json", "w"))
print(len(par), len(texts))
