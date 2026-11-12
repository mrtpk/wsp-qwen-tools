"""Generate thinking-mode answers for the tool-swap prompts and record whether the model calls a tool.

Run with the BFCL venv: ../bfcl/.venv/bin/python swap_gen.py <served-model> <port> <swap.jsonl> <out.jsonl> [think|nothink]
Sampling: the BFCL settings of run_bfcl.py (temperature 1.0, top_p 0.95, top_k 20, presence 1.5).
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI


def main(model, port, src, out, mode="think"):
    assert mode in ("think", "nothink"), mode
    thinking = mode == "think"
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")
    rows = [json.loads(line) for line in open(src)]

    def run(r):
        resp = client.chat.completions.create(
            model=model, messages=r["messages"], tools=r["tools"], temperature=1.0, top_p=0.95,
            presence_penalty=1.5, max_tokens=16384,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": thinking}})
        m = resp.choices[0].message
        calls = [{"name": c.function.name, "arguments": c.function.arguments} for c in (m.tool_calls or [])]
        return {"id": r["id"], "needs_call": r["needs_call"], "category": r["category"],
                "called": bool(calls), "calls": calls, "finish_reason": resp.choices[0].finish_reason}

    with ThreadPoolExecutor(48) as ex:
        res = list(ex.map(run, rows))
    with open(out, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    for need in (True, False):
        g = [r for r in res if r["needs_call"] == need]
        print(f"{model} {mode} {'own tools' if need else 'swapped tools'}: n {len(g)}, "
              f"call rate {sum(r['called'] for r in g) / len(g):.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
