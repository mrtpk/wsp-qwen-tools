"""Edit E1: a thresholded rank-1 edit of L23.mlp.down_proj that lowers <tool_call> when the tool does not fit.

W' = W - k * w_hat (v - (tau / mu) * m_hat)^T
  v      read direction (irrelevance > need) at the action position, from analyze_sites.py
  m_hat  mean activation direction (action + background), used as a bias proxy
  mu     mean of m_hat . a at the action position
  tau    threshold on v . a (between the need mean and the over mean)
  w_hat  unit residual direction of the <tool_call> logit through the final norm gain

Usage:
  uv run python edit_e1.py sweep <model-dir> <prompts-think.jsonl> <sites.pt> <site-dirs.pt> <out.json>
  uv run python edit_e1.py save  <model-dir> <sites.pt> <site-dirs.pt> <k> <tau> <out-model-dir>
"""

import json
import os
import random
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

TOOL_CALL = 248058
LAYER = 23
SITE = "mlp23"
FILE = "model.safetensors-00001-of-00001.safetensors"
KEY = f"model.language_model.layers.{LAYER}.mlp.down_proj.weight"
# EDIT_VARIANT=e1: protect the reasoning and query tokens. e2: also protect the tool-call tokens.
VARIANT = os.getenv("EDIT_VARIANT", "e1")
assert VARIANT in ("e1", "e2"), VARIANT
PROTECT_CALLS = VARIANT == "e2"
COMMON_FILE = f"{VARIANT}-common-ids.json"


def write_direction(W_U, g, common_ids, lam=0.1):
    """Residual direction that raises the <tool_call> logit and changes the common-token logits least.

    Minimize |C (g * u)|^2 + lam' |u|^2 subject to W_U[tc] . (g * u) = 1, where C are the common-token rows.
    """
    C = W_U[common_ids] * g
    M = C.T @ C
    M = M + lam * M.diagonal().mean() * torch.eye(M.shape[0])
    t = g * W_U[TOOL_CALL]
    u = torch.linalg.solve(M, t)
    return u / u.norm()


def edit_parts(sites, dirs, W_U, g, common_ids):
    X = sites["last"][SITE].float()
    B = sites["bg"][SITE].float()
    v = dirs[SITE].float()
    m = torch.cat([X, B]).mean(0)
    m_hat = m / m.norm()
    mu = (X @ m_hat).mean().item()
    w_hat = write_direction(W_U, g, common_ids)
    need, called = sites["needs_call"], sites["called"]
    s = X @ v
    stats = {"need": s[need & called].mean().item(), "over": s[~need & called].mean().item(),
             "refuse": s[~need & ~called].mean().item()}
    return v, m_hat, mu, w_hat, stats


def delta(v, m_hat, mu, w_hat, k, tau):
    return -k * torch.outer(w_hat, v - (tau / mu) * m_hat)



def common_tokens(tok, rows, n=3000, n_call=2000):
    """The most common reasoning and query tokens. With PROTECT_CALLS, also the most common tool-call tokens."""
    counts = {}
    for r in rows[:1500]:
        for t in tok((r["reasoning_content"] or "") + " " + r["messages"][-1]["content"], add_special_tokens=False)["input_ids"]:
            counts[t] = counts.get(t, 0) + 1
    counts.pop(TOOL_CALL, None)
    ids = [t for t, _ in sorted(counts.items(), key=lambda x: -x[1])[:n]]
    if PROTECT_CALLS:
        cc = {}
        for r in rows:
            if isinstance(r["result"], list):
                for call in r["result"]:
                    (name, args), = call.items()
                    for t in tok(xml_call(name, args), add_special_tokens=False)["input_ids"]:
                        cc[t] = cc.get(t, 0) + 1
        cc.pop(TOOL_CALL, None)
        seen = set(ids)
        ids += [t for t, _ in sorted(cc.items(), key=lambda x: -x[1]) if t not in seen][:n_call]
    return ids


def xml_call(name, args):
    out = f"<tool_call>\n<function={name}>\n"
    try:
        parsed = json.loads(args)
    except (json.JSONDecodeError, TypeError):
        parsed = {}
    for a, val in parsed.items():
        val = val if isinstance(val, str) else json.dumps(val)
        out += f"<parameter={a}>\n{val}\n</parameter>\n"
    return out + "</function>\n</tool_call>"


@torch.no_grad()
def sweep(model_dir, prompts, sites_path, dirs_path, out):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda")
    S = torch.load(sites_path)
    D = torch.load(dirs_path)
    W_U = model.model.embed_tokens.weight.float().cpu()
    g = (1 + model.model.norm.weight.float()).cpu()
    rows = {r["id"]: r for r in (json.loads(line) for line in open(prompts))}
    common_ids = common_tokens(tok, list(rows.values()))
    json.dump(common_ids, open(COMMON_FILE, "w"))
    v, m_hat, mu, w_hat, stats = edit_parts(S, D, W_U, g, common_ids)
    print("v.a means at the action position:", {k: round(x, 3) for k, x in stats.items()}, "mu", round(mu, 3))
    t = (g * W_U[TOOL_CALL]) @ w_hat
    side = ((W_U[common_ids] * g) @ w_hat).abs()
    print(f"write direction: tool_call logit gain {t:.3f}; common-token |gain| mean {side.mean():.4f} max {side.max():.4f}")
    need, called = S["needs_call"], S["called"]
    test = [i for i in range(len(S["ids"])) if i % 2 == 1]
    rng = random.Random(0)
    groups = {"need": [i for i in test if need[i] and called[i]], "refuse": [i for i in test if not need[i] and not called[i]],
              "over": [i for i in test if not need[i] and called[i]]}
    sel = {g_: rng.sample(ix, min(120, len(ix))) for g_, ix in groups.items()}
    par = [r for r in rows.values() if r["category"] in ("parallel", "parallel_multiple", "live_parallel")
           and isinstance(r["result"], list) and len(r["result"]) >= 2][:80]
    W = model.model.layers[LAYER].mlp.down_proj.weight
    W0 = W.detach().clone()

    def seq(r, extra=""):
        p = tok.apply_chat_template(r["messages"], tools=r["tools"], add_generation_prompt=True,
                                    enable_thinking=True, tokenize=False)
        return p + (r["reasoning_content"] or "") + "</think>\n\n" + extra

    inputs = {g_: [tok(seq(rows[S["ids"][i]]), add_special_tokens=False)["input_ids"] for i in ix] for g_, ix in sel.items()}
    par_inputs = []
    for r in par:
        (name, args), = r["result"][0].items()
        par_inputs.append(tok(seq(r, xml_call(name, args) + "\n"), add_special_tokens=False)["input_ids"])
    bg_rows = [rows[S["ids"][i]] for i in rng.sample(range(len(S["ids"])), 30)]
    bg_inputs = [tok(seq(r), add_special_tokens=False)["input_ids"] for r in bg_rows]
    call_rows = [r for r in rows.values() if r["needs_call"] and isinstance(r["result"], list) and r["result"]]
    call_inputs = []
    for r in rng.sample(call_rows, 30):
        (name, args), = r["result"][0].items()
        prefix = tok(seq(r), add_special_tokens=False)["input_ids"]
        full = tok(seq(r, xml_call(name, args)), add_special_tokens=False)["input_ids"]
        call_inputs.append((full, len(prefix)))

    def p_last(ids):
        return torch.softmax(model(torch.tensor([ids], device="cuda"), logits_to_keep=1, use_cache=False)
                             .logits[0, -1].float(), -1)[TOOL_CALL].item()

    def bg_logprobs(ids):
        return torch.log_softmax(model(torch.tensor([ids], device="cuda"), use_cache=False).logits[0, :-1].float(), -1)

    base_bg = [bg_logprobs(ids).cpu() for ids in bg_inputs]
    base_call = [bg_logprobs(ids)[start - 1:].cpu() for ids, start in call_inputs]
    results = []
    mid = (stats["need"] + stats["over"]) / 2
    if os.getenv("EDIT_K"):
        grid = [(0.0, mid)] + [(float(k), mid) for k in os.getenv("EDIT_K").split(",")]
    elif VARIANT == "e1":
        grid = [(0.0, stats["need"])] + [(k, tau) for k in (2, 5, 10, 20) for tau in
                                          (stats["need"], mid, stats["over"])]
    else:
        grid = [(0.0, mid), (2, mid), (5, mid), (8, mid), (5, stats["need"])]
    for k, tau in grid:
        W.copy_(W0 + delta(v, m_hat, mu, w_hat, k, tau).to(W.device, W.dtype))
        res = {"k": k, "tau": tau}
        for g_, ids_list in inputs.items():
            res[g_] = sum(p_last(ids) for ids in ids_list) / len(ids_list)
        res["parallel_2nd_call"] = sum(p_last(ids) for ids in par_inputs) / len(par_inputs)
        kl = []
        for ids, lb in zip(bg_inputs, base_bg):
            le = bg_logprobs(ids).cpu()
            kl.append((lb.exp() * (lb - le)).sum(-1).mean().item())
        res["bg_kl"] = sum(kl) / len(kl)
        kl = []
        for (ids, start), lb in zip(call_inputs, base_call):
            le = bg_logprobs(ids)[start - 1:].cpu()
            kl.append((lb.exp() * (lb - le)).sum(-1).mean().item())
        res["in_call_kl"] = sum(kl) / len(kl)
        results.append(res)
        print(json.dumps({k_: (round(x, 4) if isinstance(x, float) else x) for k_, x in res.items()}), flush=True)
    W.copy_(W0)
    json.dump(results, open(out, "w"), indent=1)


def save(model_dir, sites_path, dirs_path, k, tau, out_dir):
    S = torch.load(sites_path)
    D = torch.load(dirs_path)
    sd = load_file(f"{model_dir}/{FILE}")
    W_U = sd["model.language_model.embed_tokens.weight"].float()
    g = 1 + sd["model.language_model.norm.weight"].float()
    v, m_hat, mu, w_hat, _ = edit_parts(S, D, W_U, g, json.load(open(COMMON_FILE)))
    W = sd[KEY]
    sd[KEY] = (W.float() + delta(v, m_hat, mu, w_hat, float(k), float(tau))).to(W.dtype)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in Path(model_dir).iterdir():
        if f.is_file() and f.name != FILE:
            shutil.copy(f, out / f.name)
    save_file(sd, str(out / FILE), metadata={"format": "pt"})
    print("saved", out, "changed", KEY, "k", k, "tau", tau)


if __name__ == "__main__":
    {"sweep": sweep, "save": save}[sys.argv[1]](*sys.argv[2:])
