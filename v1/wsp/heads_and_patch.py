"""Per-head attribution, attention by context segment, and mean patching at the action position.

Usage: uv run python heads_and_patch.py <model-dir> <prompts-think.jsonl> <attr.pt> <out-prefix> [n_per_group]
Groups come from <attr.pt>: need (needs call, called), refuse (irrelevance, no call),
over (irrelevance, called).
"""

import json
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOOL_CALL = 248058
FULL = [3, 7, 11, 15, 19, 23]
HEAD_DIM = 256
PATCH = {"L15.attn": [30], "L19.attn": [38], "L20.gdn": [40], "L21.gdn": [42], "L22.gdn": [44],
         "L15+L19 attn": [30, 38], "L20-22 gdn": [40, 42, 44], "all five": [30, 38, 40, 42, 44],
         "control L17.mlp": [35]}
SEGMENTS = ["sys_head", "tools", "instructions", "user", "reasoning", "tail"]


def segments(text, prompt_len, reasoning_len):
    """Return the char ranges of the context segments."""
    t0 = text.find("<tools>")
    if t0 < 0:
        t0 = t1 = 0
    else:
        t1 = text.index("</tools>") + len("</tools>")
    u0 = text.index("<|im_start|>user")
    a0 = text.rindex("<|im_start|>assistant")
    r0 = prompt_len
    r1 = prompt_len + reasoning_len
    return [(0, t0), (t0, t1), (t1, u0), (u0, a0), (r0, r1), (r1, len(text))]


@torch.no_grad()
def main(model_dir, prompts, attr, out, n_per_group="300"):
    n = int(n_per_group)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="eager")
    layers = model.model.layers
    rows = {r["id"]: r for r in (json.loads(line) for line in open(prompts))}
    A = torch.load(attr)
    need, called = A["needs_call"], A["called"]
    groups = {"need": need & called, "refuse": ~need & ~called, "over": ~need & called}
    means = {k: A["parts"][m].float().mean(0).cuda() for k, m in groups.items()}
    rms_of = dict(zip(A["ids"], A["rms"].tolist()))
    rng = random.Random(0)
    sel = []
    for k, m in groups.items():
        idx = m.nonzero().flatten().tolist()
        rng.shuffle(idx)
        sel += [(k, A["ids"][i]) for i in idx[:n]]

    state = {"shift": None, "o_in": {}}

    def mixer_hook(j):
        def f(_m, _inp, output):
            if state["shift"] is not None and j in state["shift"]:
                d = state["shift"][j]
                if isinstance(output, tuple):
                    o = output[0].clone()
                    o[0, -1] += d.to(o.dtype)
                    return (o,) + tuple(output[1:])
                o = output.clone()
                o[0, -1] += d.to(o.dtype)
                return o
        return f

    def o_pre(i):
        def f(_m, args):
            state["o_in"][i] = args[0][0, -1].float()
        return f

    for i, layer in enumerate(layers):
        mixer = layer.linear_attn if layer.block_type == "linear_attention" else layer.self_attn
        mixer.register_forward_hook(mixer_hook(2 * i))
        layer.mlp.register_forward_hook(mixer_hook(2 * i + 1))
        if i in FULL:
            layer.self_attn.o_proj.register_forward_pre_hook(o_pre(i))

    W_U = model.model.embed_tokens.weight.float()
    g = 1 + model.model.norm.weight.float()
    res = []
    for k, (group, rid) in enumerate(sel):
        r = rows[rid]
        prompt = tok.apply_chat_template(r["messages"], tools=r["tools"], add_generation_prompt=True,
                                         enable_thinking=True, tokenize=False)
        reasoning = r["reasoning_content"] or ""
        text = prompt + reasoning + "</think>\n\n"
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        x = torch.tensor([enc["input_ids"]], device="cuda")
        state["shift"] = None
        o = model(x, output_hidden_states=True, output_attentions=True, logits_to_keep=1, use_cache=False)
        logits = o.logits[0, -1].float()
        lg = logits.clone()
        lg[TOOL_CALL] = -1e9
        other = int(lg.argmax())
        item = {"id": rid, "group": group, "p": torch.softmax(logits, -1)[TOOL_CALL].item()}
        # Per-head direct attribution through the final norm, with the saved pre-norm RMS.
        d = (W_U[TOOL_CALL] - W_U[other]) * g / rms_of[rid]
        item["head_dla"] = {}
        for i in FULL:
            W_o = layers[i].self_attn.o_proj.weight.float()
            v = state["o_in"][i].view(-1, HEAD_DIM)
            contrib = torch.stack([W_o[:, h * HEAD_DIM:(h + 1) * HEAD_DIM] @ v[h] for h in range(v.shape[0])])
            item["head_dla"][i] = (contrib @ d).tolist()
        # Attention mass by segment for each head of each full-attention layer.
        segs = segments(text, len(prompt), len(reasoning))
        offs = torch.tensor([s for s, _ in enc["offset_mapping"]])
        seg_id = torch.full((len(offs),), -1)
        for s_i, (a, b) in enumerate(segs):
            seg_id[(offs >= a) & (offs < b)] = s_i
        item["attn_seg"] = {}
        for li, i in enumerate(FULL):
            att = o.attentions[li][0, :, -1].float().cpu()      # [heads, seq]
            item["attn_seg"][i] = [[att[h][seg_id == s].sum().item() for s in range(len(SEGMENTS))]
                                   for h in range(att.shape[0])]
        # Mean patching toward the refuse mean (need and over rows) or toward the need mean (refuse rows).
        target = "need" if group == "refuse" else "refuse"
        item["patch"] = {}
        for pname, js in PATCH.items():
            state["shift"] = {j: means[target][j] - means[group][j] for j in js}
            lo = model(x, logits_to_keep=1, use_cache=False).logits[0, -1].float()
            item["patch"][pname] = torch.softmax(lo, -1)[TOOL_CALL].item()
        state["shift"] = None
        res.append(item)
        if k % 100 == 0:
            print(k, len(sel), flush=True)
    json.dump(res, open(out + ".json", "w"))
    print("saved", out + ".json")


if __name__ == "__main__":
    main(*sys.argv[1:])
