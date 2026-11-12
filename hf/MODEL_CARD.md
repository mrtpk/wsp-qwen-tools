---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/LICENSE
base_model: Qwen/Qwen3.5-0.8B
base_model_relation: finetune
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - weight-space-programming
  - model-editing
  - tool-use
  - function-calling
  - bfcl
---

# Qwen3.5-0.8B with a weight-space programming edit for tool use (v7)

This model is [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) (revision `@BASE_REV@`) with edit **v7**. The edit reprograms two neurons of the layer-23 MLP. It changes @V7_CHANGED@ values in 3 tensors. All other tensors, including the vision encoder, are identical to the base model. No gradient training of the model weights was used.

The goal of the edit: the model calls a tool when a tool call is necessary, and answers in text when no tool applies.

This is a research artifact. It is not a better tool-calling model for production use. The gain is small, it is clear in thinking mode only, and it is optimistic (see [Limits](#limits)).

| Model | BFCL-V4 overall, thinking | BFCL-V4 overall, non-thinking |
|---|---|---|
| base model | 20.46 (2 seeds); 20.84 (3 seeds) | 17.68 (2 seeds) |
| **v7** (this model) | **22.54** (3 seeds: 22.65, 22.42, 22.56) | 19.55 (2 seeds) |
| E4b, an edit of the earlier study (not in this repository) | 21.49 (2 seeds) | 20.29 (2 seeds) |

BFCL-V4 overall, with web search not run and counted as 0. Base seeds in thinking mode: 20.04, 20.88, 21.59. The lowest v7 seed is above the highest base seed. The seed spread is 0.5 to 1.5 points overall.

- The study, the code and the logs: [github.com/mrtpk/wsp-qwen-tools](https://github.com/mrtpk/wsp-qwen-tools)
- Write-up: [Weight-Space Programming - Teaching Qwen3.5-0.8B when to call a tool](https://github.com/mrtpk/kaizen/tree/master/learnings/20261112_wsp_qwen_tool_use)
- The method: [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming) (Takehi, Teoh, Zhang and Khattab, MIT CSAIL)

## Revisions

| Branch | Content |
|---|---|
| `main` | v7 |
| `v7` | v7 (the same commit as `main` at upload) |

Later edits get their own branches.

## Use

The model has the architecture, tokenizer and chat template of the base model. The chat template starts in **non-thinking mode** by default. v7 helps mainly in **thinking mode**, so set `enable_thinking=True`.

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

repo = "@REPO@"
model = AutoModelForImageTextToText.from_pretrained(repo, revision="v7", dtype="auto")
processor = AutoProcessor.from_pretrained(repo, revision="v7")
messages = [{"role": "user", "content": [{"type": "text", "text": "What is the weather in Paris?"}]}]
inputs = processor.apply_chat_template(messages, tools=TOOLS, enable_thinking=True, add_generation_prompt=True,
                                       tokenize=True, return_dict=True, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=1024)
```

`TOOLS` is a list of tool definitions in the OpenAI function format.

The BFCL runs used vLLM 0.30.0 with these flags:

```
vllm serve @REPO@ --revision v7 --language-model-only \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
```

In a request, set `"chat_template_kwargs": {"enable_thinking": true}`. The study sampled with temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5 and at most 16,384 output tokens.

## The edit

The agent was Claude, with the [OpenWSP](https://github.com/openwsp/openwsp) protocol. Each programmed neuron computes approximately `h = silu(g·x) · (r·x − θ)` at the input `x` of the layer-23 MLP, and writes `K · h · u` to the residual stream.

- `g` (the row in `gate_proj`): opens the neuron only at tool-decision positions. It is a ridge regression on activations.
- `r` and `θ` (the row in `up_proj`): a linear read of "a tool call is necessary here". It is a regularized LDA on activations.
- `u` (the column in `down_proj`): a write direction that raises the `<tool_call>` token and changes the 4,000 most common tokens of the practice outputs little. It is computed from the embedding weights.

Neuron 1488 acts in non-thinking mode. Neuron 1152 acts in thinking mode.

All fits used public practice data and activations of the base model, not BFCL data: ToolACE, APIGen-MT-5k and When2Call. When2Call enters only through the gate of neuron 1488. glaive-function-calling-v2 was used as a test set only.

The file `wsp_edit.json` gives the changed tensor names, shapes, changed-value counts and SHA-256 hashes. The edit itself (`v2/edits/run3-v7`, OpenWSP edit format) and the scripts that apply it to the base model are in the GitHub repository. The edit addresses 6,144 values. 3 of them keep their base value in bfloat16, so @V7_CHANGED@ values change.

## Limits

- BFCL scores chose v7 from four candidates. The v2 design runs 2 and 3 saw BFCL scores. No blind test exists, so the gain is optimistic.
- In non-thinking mode, v7 (19.55) is above the base (17.68), but below E4b (20.29), an edit of the earlier study.
- The edit moves the model along a trade-off between false calls and correct calls. In thinking mode, no tested edit raised irrelevance detection without a loss of AST accuracy (mean of non-live and live).
- The edit does not fix wrong argument values, multi-turn state or memory.
- Only BFCL-V4 was measured. Other abilities of the base model (general chat, math, vision) were not tested after the edit.

## Licence

Apache 2.0, the licence of the base model. The base model is by the Qwen team, Alibaba Cloud. The file `LICENSE` is the base model's licence. The changes to the base model are the 3 tensors listed in `wsp_edit.json`.
