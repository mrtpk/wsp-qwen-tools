# Run 2. A "thinking mode" term for the gate of neuron 1488 (switch the neuron off after a reasoning block).
# m = LDA (lam 3) at the layer-23 MLP input, decision positions: cA think (pass 31) against cA nothink (pass 32).
# 8-fold CV by question. Bias via q (q.x is about 11.9 at decision positions). Also builds the mt_B think
# probe request (prompt + the base reasoning from pass 43) to check the gate on multi-turn thinking states.
import json
P = {p["pass"]: p["result"] for p in map(json.loads, open("passes.jsonl")) if p["status"] == "done"}
def auroc(s, y):
    s = torch.as_tensor(s, dtype=torch.float64); y = torch.as_tensor(y).bool()
    order = torch.argsort(s); r = torch.empty_like(s); r[order] = torch.arange(1, len(s) + 1, dtype=torch.float64)
    n1, n0 = int(y.sum()), int((~y).sum()); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def lda(X1, X0, lam=3.0):
    S = torch.cov(torch.cat([X1 - X1.mean(0), X0 - X0.mean(0)]).T)
    S = S + lam * S.diagonal().mean() * torch.eye(S.shape[0], dtype=S.dtype)
    return torch.linalg.solve(S, X1.mean(0) - X0.mean(0))
def dec(p, n):
    z = np.load(P[p]["file"]); first = {}
    for k, (i, _) in enumerate(z["rows"]): first.setdefault(int(i), k)
    return torch.tensor(z["mlp_in_23"]).double()[[first[i] for i in range(n)]]
T, N = dec(31, 320), dec(32, 320)
fold = torch.tensor([(i % 128) % 8 if i < 256 else i % 8 for i in range(320)])
s = torch.zeros(640, dtype=torch.float64); f2 = torch.cat([fold, fold]); y = torch.cat([torch.ones(320), torch.zeros(320)]).bool()
X = torch.cat([T, N])
for f in range(8):
    te = f2 == f; tr = ~te; s[te] = X[te] @ lda(X[tr & y], X[tr & ~y])
print("mode read CV AUROC (think vs nothink, cA):", round(auroc(s, y), 4))
m = lda(T, N); sT, sN = T @ m, N @ m; sd = float(torch.cat([sT, sN]).std()); m = m / sd; sT, sN = sT / sd, sN / sd
tau = float((sT.min() + sN.max()) / 2) if sT.min() > sN.max() else float((sT.median() + sN.median()) / 2)
print(f"m.x think: min {float(sT.min()):.2f} median {float(sT.median()):.2f} | nothink: max {float(sN.max()):.2f} median {float(sN.median()):.2f} | tau {tau:.2f}")
mB = json.load(open("probe-meta-heldout.json")); NmB = dec(36, 256)
print(f"m.x on mt_B nothink (pass 36): max {float((NmB @ m).max()):.2f} median {float((NmB @ m).median()):.2f}")
torch.save({"m": m, "tau": tau}, "mode_term.pt")
# mt_B think probe request.
V2 = "@ROOT@/v2/"
def args_map(a):
    while isinstance(a, str):
        try: a = json.loads(a)
        except Exception: return {"value": a}
    return a if isinstance(a, dict) else {"value": a}
def fix(ms):
    return [{**mm, "tool_calls": [{**c, "function": {**c["function"], "arguments": args_map(c["function"]["arguments"])}} for c in mm["tool_calls"]]} if mm.get("tool_calls") else mm for mm in ms]
R = [json.loads(l) for l in open(V2 + "cells/run1/mt_B.jsonl")]; out = json.load(open(P[43]["file"]))["original"]
TC = tok.convert_tokens_to_ids("<tool_call>"); texts, pos = [], []
for r, o in zip(R, out):
    t = tok.apply_chat_template(fix(r["messages"]), tools=r["tools"], add_generation_prompt=True, tokenize=False, enable_thinking=True)
    t += (o["samples"][0]["reasoning"] or "").strip("\n") + "\n</think>\n\n"
    texts.append(t); pos.append([len(tok(t, add_special_tokens=False)["input_ids"]) - 1])
json.dump({"type": "probe", "model": "original", "texts": texts, "positions": pos, "mlp_in": [23], "token_ids": [TC]},
          open("req-probe-mtB-think.json", "w"))
print("mt_B think probe rows", len(texts))
