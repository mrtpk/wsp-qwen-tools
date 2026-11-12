"""Edit E3: E1 form with the background-quiet read vector of e3_read.py.

W' = W - k * u rho^T at L23.mlp.down_proj. u is the E2 write direction (tool-call tokens protected).

Usage (set EDIT_VARIANT=e2 for the protected token list, EDIT_GAMMA for gamma, EDIT_K for the grid):
  EDIT_VARIANT=e2 EDIT_GAMMA=10 EDIT_K=5,10,20,40 uv run python edit_e3.py sweep <model-dir> <prompts-think.jsonl> <sites.pt> <site-dirs.pt> <out.json>
  EDIT_VARIANT=e2 EDIT_GAMMA=10 uv run python edit_e3.py save <model-dir> <sites.pt> <site-dirs.pt> <k> 0 <out-model-dir>
"""

import os
import sys

import torch

import edit_e1
import e3_read

GAMMA = float(os.getenv("EDIT_GAMMA", "10"))


def edit_parts(sites, dirs, W_U, g, common_ids):
    rho, _ = e3_read.build(sites, GAMMA)
    w_hat = edit_e1.write_direction(W_U, g, common_ids)
    s = sites["last"][edit_e1.SITE].float() @ rho
    stats = e3_read.stats_of(s, sites["needs_call"], sites["called"])
    return rho, torch.zeros_like(rho), 1.0, w_hat, stats


edit_e1.edit_parts = edit_parts

if __name__ == "__main__":
    assert os.getenv("EDIT_VARIANT") == "e2", "E3 uses the e2 protected token list"
    {"sweep": edit_e1.sweep, "save": edit_e1.save}[sys.argv[1]](*sys.argv[2:])
