"""Record the decision state after the saved reasoning trace of each thinking-mode prompt.

Usage: uv run python probe_post_think.py <model-dir> <prompts.jsonl> <out.pt>
The sequence is: chat template (enable_thinking=True) + reasoning + "</think>\n\n".
The output has p(<tool_call>) at the next token, the greedy token, and the
residual stream at the last position after the embedding and after each layer.
"""

import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOOL_CALL = 248058


@torch.no_grad()
def main(model_dir, prompts, out):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    rows = [json.loads(line) for line in open(prompts)]
    p_call, top, hs, lens = [], [], [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        prompt = tok.apply_chat_template(r["messages"], tools=r["tools"], add_generation_prompt=True,
                                         enable_thinking=True, tokenize=False)
        text = prompt + (r["reasoning_content"] or "") + "</think>\n\n"
        ids = tok(text, add_special_tokens=False)["input_ids"]
        x = torch.tensor([ids], device="cuda")
        o = model(x, output_hidden_states=True, logits_to_keep=1, use_cache=False)
        logits = o.logits[0, -1].float()
        p_call.append(torch.softmax(logits, -1)[TOOL_CALL].item())
        top.append(int(logits.argmax()))
        hs.append(torch.stack([h[0, -1] for h in o.hidden_states]).half().cpu())
        lens.append(len(ids))
        if i % 250 == 0:
            print(f"{i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    torch.save({"ids": [r["id"] for r in rows], "p_call": torch.tensor(p_call), "top": torch.tensor(top),
                "hs": torch.stack(hs), "len": torch.tensor(lens)}, out)
    print("saved", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
