"""Collect the inputs of candidate edit sites at the action position and at background positions.

Usage: uv run python collect_sites.py <model-dir> <prompts-think.jsonl> <attr.pt> <out.pt>
Sites: mlp.down_proj of layers 20..23 (3584-d input), linear_attn.out_proj of layers 20..22 (2048-d).
Background: 8 random positions in the user text or the reasoning of the first 400 rows.
"""

import json
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from edit_e1 import xml_call

SITES = [("mlp", 20), ("mlp", 21), ("mlp", 22), ("mlp", 23), ("gdn", 20), ("gdn", 21), ("gdn", 22)]
N_BG_ROWS = 400
N_BG_POS = 8
# BG_MODE=user (default, F21): background from the user text and the reasoning.
# BG_MODE=all: background from the whole sequence, plus 4 positions inside the model's own first tool call.
BG_MODE = os.getenv("BG_MODE", "user")
assert BG_MODE in ("user", "all"), BG_MODE
N_CALL_POS = 4
# MODE=think (default): decision token after the saved reasoning and "</think>\n\n"; rows from <attr.pt>.
# MODE=nothink: decision token at the end of the enable_thinking=False template; all rows of <prompts>
# (pass "-" for <attr.pt>).
MODE = os.getenv("MODE", "think")
assert MODE in ("think", "nothink"), MODE


@torch.no_grad()
def main(model_dir, prompts, attr, out):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda")
    layers = model.model.layers
    rows = {r["id"]: r for r in (json.loads(line) for line in open(prompts))}
    if attr == "-":
        ids_all = list(rows)
        A = {"ids": ids_all, "needs_call": torch.tensor([rows[i]["needs_call"] for i in ids_all]),
             "called": torch.tensor([rows[i]["called"] for i in ids_all])}
    else:
        A = torch.load(attr)
    cur = {}

    def pre(key):
        def f(_m, args):
            cur[key] = args[0][0]
        return f

    for kind, i in SITES:
        mod = layers[i].mlp.down_proj if kind == "mlp" else layers[i].linear_attn.out_proj
        mod.register_forward_pre_hook(pre(f"{kind}{i}"))
    rng = random.Random(0)
    last = {f"{k}{i}": [] for k, i in SITES}
    bg = {f"{k}{i}": [] for k, i in SITES}
    for n, rid in enumerate(A["ids"]):
        r = rows[rid]
        prompt = tok.apply_chat_template(r["messages"], tools=r["tools"], add_generation_prompt=True,
                                         enable_thinking=MODE == "think", tokenize=False)
        u0 = len(tok(prompt[: prompt.rindex("<|im_start|>user")], add_special_tokens=False)["input_ids"])
        text = prompt + (r["reasoning_content"] or "") + "</think>\n\n" if MODE == "think" else prompt
        ids = tok(text, add_special_tokens=False)["input_ids"]
        n_act = len(ids)
        start = 0 if BG_MODE == "all" else u0
        call_pos = []
        if BG_MODE == "all" and n < N_BG_ROWS and isinstance(r["result"], list) and r["result"]:
            (name, args), = r["result"][0].items()
            ids = ids + tok(xml_call(name, args), add_special_tokens=False)["input_ids"]
            call_pos = rng.sample(range(n_act, len(ids)), min(N_CALL_POS, len(ids) - n_act))
        model(torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=1)
        pos = rng.sample(range(start, n_act - 3), min(N_BG_POS, n_act - 3 - start)) if n < N_BG_ROWS else []
        pos = pos + call_pos
        for key, a in cur.items():
            last[key].append(a[n_act - 1].half().cpu())
            if pos:
                bg[key].append(a[pos].half().cpu())
        if n % 250 == 0:
            print(n, len(A["ids"]), flush=True)
    torch.save({"ids": A["ids"], "needs_call": A["needs_call"], "called": A["called"],
                "last": {k: torch.stack(v) for k, v in last.items()},
                "bg": {k: torch.cat(v) for k, v in bg.items()}}, out)
    print("saved", out)


if __name__ == "__main__":
    main(*sys.argv[1:5])
