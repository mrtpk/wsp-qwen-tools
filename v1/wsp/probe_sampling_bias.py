"""Compute p(<tool_call>) after </think> with and without the BFCL sampling settings.

Usage: uv run python probe_sampling_bias.py <model-dir> <prompts-think.jsonl> <out.pt>
Settings: presence_penalty 1.5 on the generated tokens (reasoning + "</think>\n\n"),
temperature 1.0, top_k 20, top_p 0.95.
"""

import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOOL_CALL = 248058
PRESENCE = 1.5
TOP_K = 20
TOP_P = 0.95


def sample_prob(logits, generated, presence):
    z = logits.clone()
    if presence:
        z[torch.unique(generated)] -= presence
    top = torch.topk(z, TOP_K)
    probs = torch.softmax(top.values, -1)
    keep = (probs.cumsum(-1) - probs) < TOP_P
    probs = probs * keep
    probs = probs / probs.sum()
    hit = (top.indices == TOOL_CALL).nonzero()
    return probs[hit[0, 0]].item() if len(hit) else 0.0


@torch.no_grad()
def main(model_dir, prompts, out):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda")
    rows = [json.loads(line) for line in open(prompts)]
    raw, filt, pen = [], [], []
    for i, r in enumerate(rows):
        prompt = tok.apply_chat_template(r["messages"], tools=r["tools"], add_generation_prompt=True,
                                         enable_thinking=True, tokenize=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        g_ids = tok((r["reasoning_content"] or "") + "</think>\n\n", add_special_tokens=False)["input_ids"]
        x = torch.tensor([p_ids + g_ids], device="cuda")
        logits = model(x, logits_to_keep=1, use_cache=False).logits[0, -1].float()
        gen = torch.tensor(g_ids, device="cuda")
        raw.append(torch.softmax(logits, -1)[TOOL_CALL].item())
        filt.append(sample_prob(logits, gen, 0.0))
        pen.append(sample_prob(logits, gen, PRESENCE))
        if i % 500 == 0:
            print(i, flush=True)
    torch.save({"ids": [r["id"] for r in rows], "p_raw": torch.tensor(raw),
                "p_topk": torch.tensor(filt), "p_sampled": torch.tensor(pen)}, out)
    print("saved", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
