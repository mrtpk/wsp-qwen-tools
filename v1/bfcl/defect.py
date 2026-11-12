"""Count BFCL failures where the model answers in text instead of a tool call."""
import collections, json, sys
from pathlib import Path

CATS = ["simple_python", "simple_java", "simple_javascript", "multiple", "parallel",
        "parallel_multiple", "live_simple", "live_multiple", "live_parallel",
        "live_parallel_multiple"]


def load(model, cat):
    f = next(Path("score", model).rglob(f"BFCL_v4_{cat}_score.json"))
    lines = [json.loads(l) for l in open(f)]
    r = next(Path("result", model).rglob(f"BFCL_v4_{cat}_result.json"))
    return lines[0], lines[1:], sum(1 for _ in open(r))


for model in sys.argv[1:]:
    print(f"== {model}")
    tot_fail = tot_text = tot_n = 0
    for cat in CATS:
        head, fails, n = load(model, cat)
        text = sum(1 for x in fails if x.get("error_type") == "ast_decoder:decoder_failed"
                   and isinstance(x.get("model_result_raw"), str))
        tot_fail += len(fails); tot_text += text; tot_n += n
        print(f"{cat:24s} acc {head['accuracy']*100:5.1f}  fail {len(fails):4d}  text-instead-of-call {text:4d} ({text/n*100:4.1f}% of all)")
    print(f"{'TOTAL':24s} n {tot_n}  fail {tot_fail}  text-instead-of-call {tot_text} ({tot_text/tot_n*100:.1f}% of all, {tot_text/tot_fail*100:.1f}% of failures)")
    for cat in ["irrelevance", "live_irrelevance", "live_relevance"]:
        head, fails, n = load(model, cat)
        print(f"{cat:24s} acc {head['accuracy']*100:5.1f}  fail {len(fails)} / {n}")
