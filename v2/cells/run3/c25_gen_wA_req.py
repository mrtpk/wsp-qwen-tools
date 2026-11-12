# Run 3, cell 25. Thinking generation on wA (when2call "ask", original model): base reasoning for the probes (as c01/c02).
import json
V2 = "@ROOT@/v2/"
R = [json.loads(l) for l in open(V2 + "cells/run3/wA.jsonl")]
json.dump({"type": "generate", "model": "original", "enable_thinking": True, "max_tokens": 8192, "temperature": 1.0,
           "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "seed": 0,
           "items": [{"messages": r["messages"], "tools": r["tools"]} for r in R]}, open("req-gen-wA-think.json", "w"))
print("wA", len(R))
