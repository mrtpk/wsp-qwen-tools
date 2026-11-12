"""Record the first-assistant-token state of each BFCL single-turn prompt.

Usage: uv run python probe_first_token.py <model-dir> <prompts.jsonl> <out.pt>
The output has, for each prompt: p(<tool_call>), the greedy token, and the
residual stream at the last prompt position after the embedding and after each
decoder layer (25 x 1024, float16).
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
    p_call, top, hs = [], [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        ids = tok.apply_chat_template(r["messages"], tools=r["tools"], add_generation_prompt=True,
                                      enable_thinking=False, tokenize=True, return_dict=True)["input_ids"]
        if "input_token_count" in r:
            assert len(ids) == r["input_token_count"], (r["id"], len(ids), r["input_token_count"])
        x = torch.tensor([ids], device="cuda")
        o = model(x, output_hidden_states=True, logits_to_keep=1, use_cache=False)
        logits = o.logits[0, -1].float()
        p_call.append(torch.softmax(logits, -1)[TOOL_CALL].item())
        top.append(int(logits.argmax()))
        hs.append(torch.stack([h[0, -1] for h in o.hidden_states]).half().cpu())
        if i % 250 == 0:
            print(f"{i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    torch.save({"ids": [r["id"] for r in rows], "p_call": torch.tensor(p_call),
                "top": torch.tensor(top), "hs": torch.stack(hs)}, out)
    print("saved", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
