"""Edit E4: the E1 form, fitted on the decision tokens of both thinking modes.

W' = W - k * u rho^T at L23.mlp.down_proj, with
  rho = v + c1 * m_think + c2 * m_nothink + c3 * m_bg
  v: logistic direction (irrelevance > need) on the decision tokens of both modes (even rows).
  c: rho . a = 0 at the need/over midpoint in each mode, and mean rho . a = 0 at background.
  u: the E2 write direction (tool-call tokens protected, EDIT_VARIANT=e2 list).

Usage:
  uv run python edit_e4.py read      <sites-think.pt> <sites-nothink.pt>
  uv run python edit_e4.py calibrate <model-dir> <prompts-think.jsonl> <prompts-nothink.jsonl> <sites-think.pt> <sites-nothink.pt> <k,k,...> <out.json>
  uv run python edit_e4.py save      <model-dir> <sites-think.pt> <sites-nothink.pt> <k> <out-model-dir>
"""

import json
import random
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from analyze_layers import auroc
from analyze_sites import raw_direction
from edit_e1 import FILE, KEY, LAYER, TOOL_CALL, write_direction
from segment_kl import build as build_segments

SITE = "mlp23"
COMMON_FILE = "e2-common-ids.json"
TARGET_GAP = 0.196


def groups(need, called, mask):
    return {"need": mask & need & called, "over": mask & ~need & called, "refuse": mask & ~need & ~called}


def read_vector(St, Sn):
    Xt, Xn = St["last"][SITE].float(), Sn["last"][SITE].float()
    B = torch.cat([St["bg"][SITE].float(), Sn["bg"][SITE].float()])
    trt = torch.arange(len(Xt)) % 2 == 0
    trn = torch.arange(len(Xn)) % 2 == 0
    X = torch.cat([Xt[trt], Xn[trn]])
    y = torch.cat([~St["needs_call"][trt], ~Sn["needs_call"][trn]])
    v = raw_direction(X, y)
    m_t = Xt[trt].mean(0)
    m_n = Xn[trn].mean(0)
    m_b = B.mean(0)
    P = [m_t / m_t.norm(), m_n / m_n.norm(), m_b / m_b.norm()]
    gt = groups(St["needs_call"], St["called"], trt)
    gn = groups(Sn["needs_call"], Sn["called"], trn)
    mid_t = ((Xt[gt["need"]] @ v).mean() + (Xt[gt["over"]] @ v).mean()) / 2
    mid_n = ((Xn[gn["need"]] @ v).mean() + (Xn[gn["over"]] @ v).mean()) / 2
    # The constraints use the group midpoint vectors, so that rho . a = 0 at each mode's threshold.
    a_t = (Xt[gt["need"]].mean(0) + Xt[gt["over"]].mean(0)) / 2
    a_n = (Xn[gn["need"]].mean(0) + Xn[gn["over"]].mean(0)) / 2
    A = torch.tensor([[a_t @ p for p in P], [a_n @ p for p in P], [m_b @ p for p in P]], dtype=torch.float64)
    rhs = torch.tensor([-(a_t @ v), -(a_n @ v), -(m_b @ v)], dtype=torch.float64)
    c = torch.linalg.solve(A, rhs).float()
    rho = v + sum(ci * p for ci, p in zip(c, P))
    gap = (Xt[gt["over"]] @ rho).mean() - (Xt[gt["need"]] @ rho).mean()
    return rho / gap * TARGET_GAP, float(torch.linalg.cond(A)), (float(mid_t), float(mid_n))


def report_read(St, Sn, rho):
    for name, S in [("think", St), ("nothink", Sn)]:
        X = S["last"][SITE].float()
        te = torch.arange(len(X)) % 2 == 1
        g = groups(S["needs_call"], S["called"], te)
        s = X @ rho
        a = auroc(torch.cat([s[g["over"]], s[g["need"]]]),
                  torch.cat([torch.ones(int(g["over"].sum())), torch.zeros(int(g["need"].sum()))]).bool())
        b = S["bg"][SITE].float() @ rho
        print(f"  {name:8s} held-out mean read: need {s[g['need']].mean():+.3f} over {s[g['over']].mean():+.3f} "
              f"refuse {s[g['refuse']].mean():+.3f} | AUROC over/need {a:.3f} | bg mean {b.mean():+.3f} sd {b.std():.3f}")


def read(sites_t, sites_n):
    St, Sn = torch.load(sites_t), torch.load(sites_n)
    rho, cond, mids = read_vector(St, Sn)
    print(f"E4 read vector: condition number {cond:.1f}, v midpoints think {mids[0]:+.3f} nothink {mids[1]:+.3f}")
    report_read(St, Sn, rho)


def delta(rho, u, k):
    return -k * torch.outer(u, rho)


@torch.no_grad()
def calibrate(model_dir, prompts_t, prompts_n, sites_t, sites_n, ks, out):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda")
    St, Sn = torch.load(sites_t), torch.load(sites_n)
    rho, _, _ = read_vector(St, Sn)
    W_U = model.model.embed_tokens.weight.float().cpu()
    g = (1 + model.model.norm.weight.float()).cpu()
    u = write_direction(W_U, g, json.load(open(COMMON_FILE)))
    rows_t = {r["id"]: r for r in (json.loads(line) for line in open(prompts_t))}
    rows_n = {r["id"]: r for r in (json.loads(line) for line in open(prompts_n))}
    rng = random.Random(0)

    def pick(S, n=120):
        te = torch.arange(len(S["ids"])) % 2 == 1
        out_ = {}
        for name, m in groups(S["needs_call"], S["called"], te).items():
            idx = m.nonzero().flatten().tolist()
            out_[name] = [S["ids"][i] for i in rng.sample(idx, min(n, len(idx)))]
        return out_

    def seq_t(r):
        p = tok.apply_chat_template(r["messages"], tools=r["tools"], add_generation_prompt=True,
                                    enable_thinking=True, tokenize=False)
        return tok(p + (r["reasoning_content"] or "") + "</think>\n\n", add_special_tokens=False)["input_ids"]

    def seq_n(r):
        p = tok.apply_chat_template(r["messages"], tools=r["tools"], add_generation_prompt=True,
                                    enable_thinking=False, tokenize=False)
        return tok(p, add_special_tokens=False)["input_ids"]

    sel_t = {k_: [seq_t(rows_t[i]) for i in v_] for k_, v_ in pick(St).items()}
    sel_n = {k_: [seq_n(rows_n[i]) for i in v_] for k_, v_ in pick(Sn).items()}
    kl_rows = rng.sample([r for r in rows_t.values() if isinstance(r["result"], list) and r["result"]], 20) + \
        rng.sample([r for r in rows_t.values() if not (isinstance(r["result"], list) and r["result"])], 20)
    kl_data = [build_segments(tok, r) for r in kl_rows]

    def p_last(ids):
        lo = model(torch.tensor([ids], device="cuda"), logits_to_keep=1, use_cache=False).logits[0, -1].float()
        return torch.softmax(lo, -1)[TOOL_CALL].item()

    def lps(ids):
        return torch.log_softmax(model(torch.tensor([ids], device="cuda"), use_cache=False).logits[0, :-1].float(), -1)

    base_lp = [lps(ids).cpu() for ids, _ in kl_data]
    W = model.model.layers[LAYER].mlp.down_proj.weight
    W0 = W.detach().clone()
    results = []
    for k in [0.0] + [float(x) for x in ks.split(",")]:
        W.copy_(W0 + delta(rho, u, k).to(W.device, W.dtype))
        res = {"k": k}
        for mode, sel in [("think", sel_t), ("nothink", sel_n)]:
            for name, ids_list in sel.items():
                res[f"{mode}_{name}"] = sum(p_last(ids) for ids in ids_list) / len(ids_list)
        kls = {"reasoning": [0.0, 0], "call": [0.0, 0]}
        for (ids, seg), lb in zip(kl_data, base_lp):
            kl = (lb.exp() * (lb - lps(ids).cpu())).sum(-1)
            for name, s in [("reasoning", 2), ("call", 4)]:
                kls[name][0] += kl[seg == s].sum().item()
                kls[name][1] += int((seg == s).sum())
        res["reasoning_kl"] = kls["reasoning"][0] / max(1, kls["reasoning"][1])
        res["call_kl"] = kls["call"][0] / max(1, kls["call"][1])
        results.append(res)
        print(json.dumps({k_: round(v_, 4) for k_, v_ in res.items()}), flush=True)
    W.copy_(W0)
    json.dump(results, open(out, "w"), indent=1)


def save(model_dir, sites_t, sites_n, k, out_dir):
    St, Sn = torch.load(sites_t), torch.load(sites_n)
    rho, _, _ = read_vector(St, Sn)
    sd = load_file(f"{model_dir}/{FILE}")
    W_U = sd["model.language_model.embed_tokens.weight"].float()
    g = 1 + sd["model.language_model.norm.weight"].float()
    u = write_direction(W_U, g, json.load(open(COMMON_FILE)))
    W = sd[KEY]
    sd[KEY] = (W.float() + delta(rho, u, float(k))).to(W.dtype)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in Path(model_dir).iterdir():
        if f.is_file() and f.name != FILE:
            shutil.copy(f, out / f.name)
    save_file(sd, str(out / FILE), metadata={"format": "pt"})
    print("saved", out, "changed", KEY, "k", k)


if __name__ == "__main__":
    {"read": read, "calibrate": calibrate, "save": save}[sys.argv[1]](*sys.argv[2:])
