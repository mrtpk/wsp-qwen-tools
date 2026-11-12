"""Add an explicit conclusion to the end of the saved reasoning and measure p(<tool_call>).

Usage: uv run python conclusion_test.py <model-dir> <prompts-think.jsonl> <out.json>
Rows: all irrelevance over-calls, and 400 random "needs call" rows that called.
"""

import json
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOOL_CALL = 248058
VARIANTS = {
    "original": "",
    "answer": "\n\nConclusion: I will answer directly without using any tools.\n",
    "tool": "\n\nConclusion: I will call the tool.\n",
}


@torch.no_grad()
def main(model_dir, prompts, out):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda")
    rows = [json.loads(line) for line in open(prompts)]
    over = [r for r in rows if not r["needs_call"] and r["called"]]
    good = [r for r in rows if r["needs_call"] and r["called"]]
    random.Random(0).shuffle(good)
    sel = [("irrelevance_overcall", r) for r in over] + [("needs_call_called", r) for r in good[:400]]
    res = []
    for group, r in sel:
        prompt = tok.apply_chat_template(r["messages"], tools=r["tools"], add_generation_prompt=True,
                                         enable_thinking=True, tokenize=False)
        item = {"id": r["id"], "group": group}
        for name, suffix in VARIANTS.items():
            reasoning = (r["reasoning_content"] or "").rstrip("\n") + suffix if suffix else (r["reasoning_content"] or "")
            ids = tok(prompt + reasoning + "</think>\n\n", add_special_tokens=False)["input_ids"]
            logits = model(torch.tensor([ids], device="cuda"), logits_to_keep=1, use_cache=False).logits[0, -1]
            item[name] = torch.softmax(logits.float(), -1)[TOOL_CALL].item()
        res.append(item)
    json.dump(res, open(out, "w"))
    for group in ["irrelevance_overcall", "needs_call_called"]:
        g = [x for x in res if x["group"] == group]
        line = "  ".join(f"{k} {sum(x[k] for x in g) / len(g):.3f}" for k in VARIANTS)
        low = sum(x["answer"] < 0.5 for x in g) / len(g)
        print(f"{group:22s} n {len(g):4d}  mean p_call: {line}  | frac p<0.5 with 'answer': {low:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
