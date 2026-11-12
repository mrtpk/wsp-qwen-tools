"""Compare the wsp edit directions with the tool-relevance direction and the <tool_call> readout.

Usage: uv run python wsp_geometry.py <base-dir> <wsp-dir> <prompts.jsonl> <acts-base.pt> [acts-wsp.pt]
"""

import json
import sys

import torch
from safetensors import safe_open

TOOL_CALL = 248058
TEXT_TOKENS = ["I", "To", "Here", "The"]
FILE = "model.safetensors-00001-of-00001.safetensors"
P = "model.language_model"


def unit(x):
    return x / x.norm()


@torch.no_grad()
def main(base_dir, wsp_dir, prompts, acts_base, acts_wsp=None):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_dir)
    fb = safe_open(f"{base_dir}/{FILE}", "pt", device="cpu")
    fw = safe_open(f"{wsp_dir}/{FILE}", "pt", device="cpu")
    W_U = fb.get_tensor(f"{P}.embed_tokens.weight").float()
    norm_w = fb.get_tensor(f"{P}.norm.weight").float()
    text_ids = [tok.encode(t, add_special_tokens=False)[0] for t in TEXT_TOKENS]
    # Readout direction in the residual stream before the final norm gain.
    readout = unit((W_U[TOOL_CALL] - W_U[text_ids].mean(0)) * (1 + norm_w))

    rows = [json.loads(line) for line in open(prompts)]
    need = torch.tensor([r["needs_call"] for r in rows])
    A = torch.load(acts_base)

    for layer in (15, 19):
        name = f"{P}.layers.{layer}.mlp.down_proj.weight"
        D = fw.get_tensor(name).float() - fb.get_tensor(name).float()
        U, S, Vh = torch.linalg.svd(D, full_matrices=False)
        u = U[:, 0]
        hs_idx = layer + 1
        X = A["hs"][:, hs_idx].float()
        rel = unit(X[need].mean(0) - X[~need].mean(0))
        mean_h = X.mean(0)
        print(f"layer {layer}: delta sigma1 {S[0]:.3f} (sigma2 {S[1]:.4f})")
        print(f"  cos(u, relevance dir at hs{hs_idx}) = {torch.dot(u, rel):+.3f}")
        print(f"  cos(u, <tool_call>-text readout)   = {torch.dot(u, readout):+.3f}")
        print(f"  cos(u, mean residual at hs{hs_idx})   = {torch.dot(u, unit(mean_h)):+.3f}")
        proj = X @ u
        print(f"  mean proj of residual on u: needs_call {proj[need].mean():+.3f}  irrelevance {proj[~need].mean():+.3f}")
        print(f"  cos(relevance dir, readout) at hs{hs_idx} = {torch.dot(rel, readout):+.3f}")
        # Effect of the edit on the residual at this token: the change is D @ a, where a is the MLP activation.
        if acts_wsp:
            B = torch.load(acts_wsp)
            dX = B["hs"][:, hs_idx].float() - X
            print(f"  wsp - base residual at hs{hs_idx}: mean norm {dX.norm(dim=1).mean():.3f}, "
                  f"cos(mean change, u) {torch.dot(unit(dX.mean(0)), u):+.3f}, "
                  f"mean change on u: needs_call {(dX[need] @ u).mean():+.3f} irrelevance {(dX[~need] @ u).mean():+.3f}")
    if acts_wsp:
        B = torch.load(acts_wsp)
        # hs24 is after the final norm, so the readout has no norm gain here.
        post = W_U[TOOL_CALL] - W_U[text_ids].mean(0)
        X24, Y24 = A["hs"][:, 24].float(), B["hs"][:, 24].float()
        print(f"final logit diff <tool_call> - text: base needs {(X24[need] @ post).mean():+.3f} irr {(X24[~need] @ post).mean():+.3f}; "
              f"wsp needs {(Y24[need] @ post).mean():+.3f} irr {(Y24[~need] @ post).mean():+.3f}")
        print(f"mean p_call: base needs {A['p_call'][need].mean():.3f} irr {A['p_call'][~need].mean():.3f}; "
              f"wsp needs {B['p_call'][need].mean():.3f} irr {B['p_call'][~need].mean():.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
