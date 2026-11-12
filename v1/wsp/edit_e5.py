"""Edit E5: E4 with a constraint on the mean read at mid-conversation decision tokens.

rho = v + c1 m_think + c2 m_nothink + c3 m_bg + c4 m_mid, with v the E4 logistic direction, and
  rho . a = 0 at the need/over midpoint in each mode (first decisions),
  mean rho . a = 0 at background,
  mean rho . a = t_mid at the later-step decision tokens of the multi-turn traces (even entry IDs).
W' = W - k * u rho^T at L23.mlp.down_proj (u: E2 write direction).

Usage:
  uv run python edit_e5.py read      <t_mid>
  uv run python edit_e5.py calibrate <model-dir> <t_mid,t_mid,...> <k> <out.json>
  uv run python edit_e5.py save      <model-dir> <t_mid> <k> <out-model-dir>
Fixed inputs (in v1/wsp): sites-base-allbg.pt, sites-base-nothink-allbg.pt, mt-reads-base.pt,
mt-decisions-base.jsonl, prompts-base-think.jsonl, prompts-base-nothink-v2.jsonl.
"""

import json
import random
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import edit_e4
from analyze_sites import raw_direction
from edit_e1 import FILE, KEY, LAYER, TOOL_CALL, write_direction
from segment_kl import build as build_segments

SITE = "mlp23"
COMMON_FILE = "e2-common-ids.json"
TARGET_GAP = 0.196


def entry_num(entry_id):
    return int(entry_id.rsplit("_", 1)[1])


def load_inputs():
    St = torch.load("sites-base-allbg.pt")
    Sn = torch.load("sites-base-nothink-allbg.pt")
    M = torch.load("mt-reads-base.pt")
    return St, Sn, M


def parse_targets(t):
    """'t_called:t_stopped' or one value for both."""
    parts = str(t).split(":")
    return (float(parts[0]), float(parts[-1]))


def read_vector(St, Sn, M, t):
    """t = (t_called, t_stopped): mean read targets at the later-step decision tokens that called and stopped."""
    t_c, t_s = parse_targets(t) if not isinstance(t, tuple) else t
    Xt, Xn = St["last"][SITE].float(), Sn["last"][SITE].float()
    B = torch.cat([St["bg"][SITE].float(), Sn["bg"][SITE].float()])
    trt = torch.arange(len(Xt)) % 2 == 0
    trn = torch.arange(len(Xn)) % 2 == 0
    v = raw_direction(torch.cat([Xt[trt], Xn[trn]]), torch.cat([~St["needs_call"][trt], ~Sn["needs_call"][trn]]))
    A_mid = M["act"].float()
    fit_ids = torch.tensor([entry_num(m["id"]) % 2 == 0 for m in M["meta"]])
    lc = torch.tensor([m["step"] > 0 and m["called"] for m in M["meta"]]) & fit_ids
    ls = torch.tensor([m["step"] > 0 and not m["called"] for m in M["meta"]]) & fit_ids
    gt = edit_e4.groups(St["needs_call"], St["called"], trt)
    gn = edit_e4.groups(Sn["needs_call"], Sn["called"], trn)
    a_t = (Xt[gt["need"]].mean(0) + Xt[gt["over"]].mean(0)) / 2
    a_n = (Xn[gn["need"]].mean(0) + Xn[gn["over"]].mean(0)) / 2
    m_b = B.mean(0)
    m_c = A_mid[lc].mean(0)
    m_s = A_mid[ls].mean(0)
    P = [x / x.norm() for x in (Xt[trt].mean(0), Xn[trn].mean(0), m_b, m_c, m_s)]

    # The scale of rho is set after the solve. Solve with targets 0 to get the scale, then with the
    # targets divided by that scale. The final over-need gap is then close to TARGET_GAP.
    def solve(tc, ts):
        A = torch.tensor([[a @ p for p in P] for a in (a_t, a_n, m_b, m_c, m_s)], dtype=torch.float64)
        rhs = torch.tensor([-(a_t @ v), -(a_n @ v), -(m_b @ v), tc - (m_c @ v), ts - (m_s @ v)], dtype=torch.float64)
        c = torch.linalg.solve(A, rhs).float()
        return v + sum(ci * p for ci, p in zip(c, P)), float(torch.linalg.cond(A))

    rho0, cond = solve(0.0, 0.0)
    gap0 = (Xt[gt["over"]] @ rho0).mean() - (Xt[gt["need"]] @ rho0).mean()
    scale = float(TARGET_GAP / gap0)
    rho, _ = solve(t_c / scale, t_s / scale)
    return rho * scale, cond


def report(St, Sn, M, rho):
    edit_e4.report_read(St, Sn, rho)
    A_mid = M["act"].float()
    s = A_mid @ rho
    for split, want in [("fit (even IDs)", 0), ("held-out (odd IDs)", 1)]:
        ids = torch.tensor([entry_num(m["id"]) % 2 == want for m in M["meta"]])
        for lab, sel in [("step0", lambda m: m["step"] == 0), ("later called", lambda m: m["step"] > 0 and m["called"]),
                         ("later stopped", lambda m: m["step"] > 0 and not m["called"])]:
            mask = ids & torch.tensor([sel(m) for m in M["meta"]])
            print(f"  multi-turn {split:18s} {lab:13s} n {int(mask.sum()):4d} mean read {s[mask].mean():+.3f}")


def read(t_mid):
    St, Sn, M = load_inputs()
    rho, cond = read_vector(St, Sn, M, t_mid)
    print(f"E5 read vector, t_mid {t_mid}: condition number {cond:.1f}")
    report(St, Sn, M, rho)


@torch.no_grad()
def calibrate(model_dir, t_mids, k, out):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda")
    St, Sn, M = load_inputs()
    W_U = model.model.embed_tokens.weight.float().cpu()
    g = (1 + model.model.norm.weight.float()).cpu()
    u = write_direction(W_U, g, json.load(open(COMMON_FILE)))
    rows_t = {r["id"]: r for r in (json.loads(line) for line in open("prompts-base-think.jsonl"))}
    rows_n = {r["id"]: r for r in (json.loads(line) for line in open("prompts-base-nothink-v2.jsonl"))}
    mt = [json.loads(line) for line in open("mt-decisions-base.jsonl")]
    rng = random.Random(0)

    def pick(S, n=100):
        te = torch.arange(len(S["ids"])) % 2 == 1
        return {name: [S["ids"][i] for i in rng.sample(m.nonzero().flatten().tolist(), min(n, int(m.sum())))]
                for name, m in edit_e4.groups(S["needs_call"], S["called"], te).items()}

    def prompt(r, thinking):
        return tok.apply_chat_template(r["messages"], tools=r["tools"], add_generation_prompt=True,
                                       enable_thinking=thinking, tokenize=False)

    sel = {}
    for name, ids in pick(St).items():
        sel[f"think_{name}"] = [tok(prompt(rows_t[i], True) + (rows_t[i]["reasoning_content"] or "") + "</think>\n\n",
                                    add_special_tokens=False)["input_ids"] for i in ids]
    for name, ids in pick(Sn).items():
        sel[f"nothink_{name}"] = [tok(prompt(rows_n[i], False), add_special_tokens=False)["input_ids"] for i in ids]
    held = [r for r in mt if entry_num(r["id"]) % 2 == 1 and r["step"] > 0]
    for name, flag in [("mt_later_called", True), ("mt_later_stopped", False)]:
        cand = [r for r in held if r["called"] == flag]
        out_ids = []
        for r in rng.sample(cand, min(len(cand), 150)):
            p = prompt(r, True)
            if len(tok(p, add_special_tokens=False)["input_ids"]) != r["input_token_count"]:
                continue
            out_ids.append(tok(p + r["reasoning_content"] + "</think>\n\n", add_special_tokens=False)["input_ids"])
            if len(out_ids) == 100:
                break
        sel[name] = out_ids
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
    grid = [("base", None)] + [("E4", None)] + [(f"E5 t={t}", t) for t in t_mids.split(",")]
    for name, t in grid:
        if name == "base":
            W.copy_(W0)
        elif name == "E4":
            rho, _, _ = edit_e4.read_vector(St, Sn)
            W.copy_(W0 + edit_e4.delta(rho, u, float(k)).to(W.device, W.dtype))
        else:
            rho, _ = read_vector(St, Sn, M, t)
            W.copy_(W0 + edit_e4.delta(rho, u, float(k)).to(W.device, W.dtype))
        res = {"edit": name, "k": float(k)}
        for key, ids_list in sel.items():
            res[key] = sum(p_last(ids) for ids in ids_list) / max(1, len(ids_list))
        kls = {"reasoning": [0.0, 0], "call": [0.0, 0]}
        for (ids, seg), lb in zip(kl_data, base_lp):
            kl = (lb.exp() * (lb - lps(ids).cpu())).sum(-1)
            for kname, s in [("reasoning", 2), ("call", 4)]:
                kls[kname][0] += kl[seg == s].sum().item()
                kls[kname][1] += int((seg == s).sum())
        res["reasoning_kl"] = kls["reasoning"][0] / max(1, kls["reasoning"][1])
        res["call_kl"] = kls["call"][0] / max(1, kls["call"][1])
        results.append(res)
        print(json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in res.items()}), flush=True)
    W.copy_(W0)
    json.dump(results, open(out, "w"), indent=1)


def save(model_dir, t_mid, k, out_dir):
    St, Sn, M = load_inputs()
    rho, _ = read_vector(St, Sn, M, t_mid)
    sd = load_file(f"{model_dir}/{FILE}")
    W_U = sd["model.language_model.embed_tokens.weight"].float()
    g = 1 + sd["model.language_model.norm.weight"].float()
    u = write_direction(W_U, g, json.load(open(COMMON_FILE)))
    W = sd[KEY]
    sd[KEY] = (W.float() + edit_e4.delta(rho, u, float(k))).to(W.dtype)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in Path(model_dir).iterdir():
        if f.is_file() and f.name != FILE:
            shutil.copy(f, out / f.name)
    save_file(sd, str(out / FILE), metadata={"format": "pt"})
    print("saved", out, "changed", KEY, "t_mid", t_mid, "k", k)


if __name__ == "__main__":
    {"read": read, "calibrate": calibrate, "save": save}[sys.argv[1]](*sys.argv[2:])
