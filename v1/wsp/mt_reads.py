"""Read values of the E1b and E4b read vectors at multi-turn decision tokens (test of H9).

Usage: uv run python mt_reads.py <base-dir> <mt-decisions.jsonl> <out.pt> [n_rows]
Only rows whose rebuilt prompt matches the vLLM token count are used. The decision token is the
one after the step reasoning and "</think>\n\n". The read vectors are rebuilt exactly as for the
saved checkpoints: E1b from sites-base.pt (F22), E4b from edit_e4.read_vector (F33).
"""

import json
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import edit_e4

TOOL_CALL = 248058
E1_TAU = -0.092


def e1b_rho():
    S = torch.load("sites-base.pt")
    D = torch.load("site-dirs-base.pt")
    X, B = S["last"]["mlp23"].float(), S["bg"]["mlp23"].float()
    m = torch.cat([X, B]).mean(0)
    m_hat = m / m.norm()
    mu = (X @ m_hat).mean()
    return D["mlp23"].float() - (E1_TAU / mu) * m_hat


@torch.no_grad()
def main(base_dir, decisions, out, n_rows="1000"):
    rho1 = e1b_rho()
    rho4, _, _ = edit_e4.read_vector(torch.load("sites-base-allbg.pt"), torch.load("sites-base-nothink-allbg.pt"))
    tok = AutoTokenizer.from_pretrained(base_dir)
    rows = [json.loads(line) for line in open(decisions)]
    ok = []
    for r in rows:
        p = tok.apply_chat_template(r["messages"], tools=r["tools"], add_generation_prompt=True,
                                    enable_thinking=True, tokenize=False)
        if len(tok(p, add_special_tokens=False)["input_ids"]) == r["input_token_count"]:
            ok.append((r, p))
    random.Random(0).shuffle(ok)
    ok = ok[: int(n_rows)]
    model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16, device_map="cuda")
    cur = {}
    model.model.layers[23].mlp.down_proj.register_forward_pre_hook(lambda _m, a: cur.__setitem__("a", a[0][0, -1].float()))
    meta, A, P = [], [], []
    for k, (r, p) in enumerate(ok):
        ids = tok(p + r["reasoning_content"] + "</think>\n\n", add_special_tokens=False)["input_ids"]
        lo = model(torch.tensor([ids], device="cuda"), logits_to_keep=1, use_cache=False).logits[0, -1].float()
        A.append(cur["a"].cpu())
        P.append(torch.softmax(lo, -1)[TOOL_CALL].item())
        meta.append({"id": r["id"], "turn": r["turn"], "step": r["step"], "called": r["called"]})
        if k % 100 == 0:
            print(k, len(ok), flush=True)
    A = torch.stack(A)
    torch.save({"meta": meta, "p_call": torch.tensor(P), "read_e1b": A @ rho1, "read_e4b": A @ rho4,
                "act": A.half()}, out)
    called = torch.tensor([m["called"] for m in meta])
    later = torch.tensor([m["step"] > 0 for m in meta])
    print(f"rows {len(meta)}: step 0 {int((~later).sum())}, later steps {int(later.sum())}")
    for name in ["read_e1b", "read_e4b"]:
        s = A @ (rho1 if name == "read_e1b" else rho4)
        for lab, m in [("step0 called", ~later & called), ("step0 stopped", ~later & ~called),
                       ("later called", later & called), ("later stopped", later & ~called)]:
            print(f"{name} {lab:14s} n {int(m.sum()):4d} mean {s[m].mean():+.3f} frac>0 {(s[m] > 0).float().mean():.3f}")
    print("p_call mean: later called", torch.tensor(P)[later & called].mean().item(), "later stopped", torch.tensor(P)[later & ~called].mean().item())


if __name__ == "__main__":
    main(*sys.argv[1:])
