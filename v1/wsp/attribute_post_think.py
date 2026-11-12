"""Direct logit attribution of the <tool_call> decision after </think>.

Usage: uv run python attribute_post_think.py <model-dir> <prompts-think.jsonl> <out.pt> [n_need]
Rows: all irrelevance prompts and n_need random "needs call" prompts (default 800).
Saves, at the action position: the embedding, the output of each token mixer
and each MLP (48 parts), the final norm RMS, and the strongest non-<tool_call> token.
"""

import json
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOOL_CALL = 248058


@torch.no_grad()
def main(model_dir, prompts, out, n_need="800"):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda")
    layers = model.model.layers
    rows = [json.loads(line) for line in open(prompts)]
    need = [r for r in rows if r["needs_call"]]
    random.Random(0).shuffle(need)
    sel = [r for r in rows if not r["needs_call"]] + need[: int(n_need)]

    store = {}

    def hook(key):
        def f(_m, _inp, output):
            o = output[0] if isinstance(output, tuple) else output
            store[key] = o[0, -1].float()
        return f

    for i, layer in enumerate(layers):
        mixer = layer.linear_attn if layer.block_type == "linear_attention" else layer.self_attn
        mixer.register_forward_hook(hook(2 * i))
        layer.mlp.register_forward_hook(hook(2 * i + 1))
    model.model.embed_tokens.register_forward_hook(hook("emb"))

    parts, emb, rms, other, p_call, max_err = [], [], [], [], [], 0.0
    norm_w = model.model.norm.weight.float()
    for k, r in enumerate(sel):
        prompt = tok.apply_chat_template(r["messages"], tools=r["tools"], add_generation_prompt=True,
                                         enable_thinking=True, tokenize=False)
        ids = tok(prompt + (r["reasoning_content"] or "") + "</think>\n\n", add_special_tokens=False)["input_ids"]
        o = model(torch.tensor([ids], device="cuda"), output_hidden_states=True, logits_to_keep=1, use_cache=False)
        P = torch.stack([store[j] for j in range(2 * len(layers))])
        resid = store["emb"] + P.sum(0)
        r_rms = torch.sqrt(resid.pow(2).mean() + 1e-6)
        normed = resid / r_rms * (1 + norm_w)
        max_err = max(max_err, (normed - o.hidden_states[-1][0, -1].float()).abs().max().item())
        logits = o.logits[0, -1].float()
        lg = logits.clone()
        lg[TOOL_CALL] = -1e9
        parts.append(P.half().cpu())
        emb.append(store["emb"].half().cpu())
        rms.append(r_rms.item())
        other.append(int(lg.argmax()))
        p_call.append(torch.softmax(logits, -1)[TOOL_CALL].item())
        if k % 250 == 0:
            print(k, len(sel), f"max recon err {max_err:.4f}", flush=True)
    torch.save({"ids": [r["id"] for r in sel], "needs_call": torch.tensor([r["needs_call"] for r in sel]),
                "called": torch.tensor([r["called"] for r in sel]), "parts": torch.stack(parts),
                "emb": torch.stack(emb), "rms": torch.tensor(rms), "other": torch.tensor(other),
                "p_call": torch.tensor(p_call)}, out)
    print("saved", out, f"max recon err {max_err:.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
