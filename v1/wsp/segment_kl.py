"""KL(base || edited) of the next-token distribution, split by context segment.

Usage: uv run python segment_kl.py <base-dir> <prompts-think.jsonl> <edited-dir> [<edited-dir> ...]
Segments: system, user, reasoning (generated), action (the token after </think>\n\n), call (inside the
model's own first tool call, generated). Rows: 40 random rows (seed 0), 20 with a call.
Also reports the largest single-position KL and its segment.
"""

import json
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from edit_e1 import xml_call

SEGS = ["system", "user", "reasoning", "action", "call"]


def build(tok, r):
    prompt = tok.apply_chat_template(r["messages"], tools=r["tools"], add_generation_prompt=True,
                                     enable_thinking=True, tokenize=False)
    u0 = len(tok(prompt[: prompt.rindex("<|im_start|>user")], add_special_tokens=False)["input_ids"])
    r0 = len(tok(prompt, add_special_tokens=False)["input_ids"])
    ids = tok(prompt + (r["reasoning_content"] or "") + "</think>\n\n", add_special_tokens=False)["input_ids"]
    n_act = len(ids)
    if isinstance(r["result"], list) and r["result"]:
        (name, args), = r["result"][0].items()
        ids = ids + tok(xml_call(name, args), add_special_tokens=False)["input_ids"]
    # seg[i] is the segment of the prediction made at position i (the prediction of token i+1).
    seg = []
    for i in range(len(ids) - 1):
        if i < u0:
            seg.append(0)
        elif i < r0 - 1:
            seg.append(1)
        elif i < n_act - 1:
            seg.append(2)
        elif i == n_act - 1:
            seg.append(3)
        else:
            seg.append(4)
    return ids, torch.tensor(seg)


@torch.no_grad()
def logprobs(model, ids):
    return torch.log_softmax(model(torch.tensor([ids], device="cuda"), use_cache=False).logits[0, :-1].float(), -1)


def main(base_dir, prompts, *edited):
    tok = AutoTokenizer.from_pretrained(base_dir)
    rows = [json.loads(line) for line in open(prompts)]
    rng = random.Random(0)
    with_call = [r for r in rows if isinstance(r["result"], list) and r["result"]]
    without = [r for r in rows if not (isinstance(r["result"], list) and r["result"])]
    sel = rng.sample(with_call, 20) + rng.sample(without, 20)
    data = [build(tok, r) for r in sel]
    base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16, device_map="cuda")
    base_lp = [logprobs(base, ids).cpu() for ids, _ in data]
    del base
    torch.cuda.empty_cache()
    for d in edited:
        m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16, device_map="cuda")
        sums = torch.zeros(len(SEGS))
        counts = torch.zeros(len(SEGS))
        top = (0.0, None, None)
        for (ids, seg), lb in zip(data, base_lp):
            le = logprobs(m, ids).cpu()
            kl = (lb.exp() * (lb - le)).sum(-1)
            for s in range(len(SEGS)):
                sums[s] += kl[seg == s].sum()
                counts[s] += (seg == s).sum()
            j = int(kl.argmax())
            if kl[j] > top[0]:
                top = (kl[j].item(), SEGS[int(seg[j])], tok.decode(ids[max(0, j - 6): j + 2]))
        mean = sums / counts.clamp_min(1)
        print(d.split("/")[-1], " ".join(f"{SEGS[s]} {mean[s]:.4f} (n {int(counts[s])})" for s in range(len(SEGS))),
              f"| max {top[0]:.3f} in {top[1]}: {top[2]!r}")
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(*sys.argv[1:])
