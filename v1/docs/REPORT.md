# Report: a weight edit for the tool-call decision of Qwen3.5-0.8B

Status: FINAL, 2026-11-07 (UTC+4).
Details: `findings.md` (F1 to F49), `decisions.md` (D1 to D18), and the time
log `progress-log.md`. Each statement below gives its F or D number.

## 1. Summary

- `models-text/modelq.base64` is the unchanged `Qwen/Qwen3.5-0.8B`
  checkpoint (F1).
- The model has two tool-call faults, one in each mode:
  - In non-thinking mode, it answers in text when a fitting tool exists
    ("I can answer this myself"). This is 47% of its single-turn failures
    (F5, F10).
  - In thinking mode, it over-calls. It calls a tool for 46.7% of the
    prompts where no tool fits (F13).
- In both faults, the model has the correct information internally. A
  linear probe reads "a tool fits" much better than the model's own output
  uses it (F7, F20).
- **The recommended edit is E4b.** It is one rank-1 change of
  `model.language_model.layers.23.mlp.down_proj.weight`. It reads the "the
  tool does not fit" signal and moves the `<tool_call>` logit around a
  threshold for each mode (D16, F32, F33).
- E4b improves BFCL-V4 in both modes (two samples each, F39, F43):

| BFCL-V4 overall, no web search | thinking | non-thinking |
|---|---|---|
| base | 20.46 | 17.68 |
| E4b | 21.49 (+1.03) | 20.29 (+2.61) |

- E4b generalizes to prompts that the fit never saw. On 858 tool-swap pairs,
  it removes 41% of the false calls in thinking mode. In non-thinking mode,
  it raises the correct calls by 13.3 points without more false calls (F41).
- The RikiyaT "weight-space programming" edit is worse than base for tool
  calls in both modes. It is the negative reference of this study (F12,
  D10).

## 2. Scope and setup

- Model: `Qwen/Qwen3.5-0.8B`, revision
  `2fc06364715b967f1860aea9cf38778875588b17`. 24 layers. Layers 3, 7, 11, 15,
  19, and 23 use full attention. The other layers use Gated DeltaNet (F2).
- Benchmark: BFCL-V4 with `bfcl-eval==2026.3.23` in a separate venv (D1).
  Requests go to `vllm serve` (`vllm==0.30.0`) with the `qwen3_coder` tool
  parser and the `qwen3` reasoning parser (D2).
- Sampling: temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5,
  maximum 16,384 output tokens (D3).
- Samples: base and E4b have two samples in each mode (vLLM seeds 0 and 1).
  E1b has two samples in thinking mode. The other edits have one sample
  (D17).
- Not run: web search, because there is no SerpAPI key (D4). The overall
  score counts web search as 0 (20% of the weight).
- Mechanism tools: HF Transformers 5.17.0 forward passes on the exact BFCL
  prompts. The rebuilt prompts match the vLLM token counts (F6, F13; for the
  multi-turn traces, 88% match exactly, F45).

## 3. Results

### 3.1 BFCL-V4, thinking mode (two-sample means, F39, F48)

| Measure | base | wsp (1) | E1b | E4b | E5c (1) |
|---|---|---|---|---|---|
| Overall | 20.46 | 19.04 | 22.80 | 21.49 | 21.14 |
| Non-live AST | 56.87 | 42.67 | 57.22 | 58.68 | 54.73 |
| Live AST | 55.48 | 47.67 | 51.74 | 53.03 | 52.11 |
| Multi-turn | 5.44 | 5.38 | 7.19 | 5.38 | 5.12 |
| Memory | 12.47 | 12.04 | 12.79 | 8.71 | 8.17 |
| Irrelevance detection | 51.04 | 59.83 | 71.94 | 69.69 | 72.80 |

### 3.2 BFCL-V4, non-thinking mode (two-sample means where available, F43, F47)

| Measure | base | wsp (1) | E1b (1) | E4b | E5c (1) |
|---|---|---|---|---|---|
| Overall | 17.68 | 15.66 | 16.73 | 20.29 | 19.72 |
| Non-live AST | 33.98 | 24.31 | 27.65 | 49.55 | 45.15 |
| Live AST | 43.89 | 36.34 | 29.76 | 46.45 | 46.71 |
| Multi-turn | 3.44 | 1.88 | 2.12 | 3.31 | 2.63 |
| Memory | 5.48 | 4.73 | 5.16 | 5.16 | 5.38 |
| Irrelevance detection | 77.72 | 80.90 | 93.18 | 86.66 | 86.70 |
| Text instead of a call (of 2,501) | 685 / 699 | 842 | 1,252 | 332 / 328 | 360 |

### 3.3 Noise (F37, F40)

- Base against base (thinking): overall +0.84, live AST +0.22, irrelevance
  +0.32, memory +4.30.
- Memory is very noisy: one sample has an SD of about 3 points.
- Live categories have a large two-way churn between samples (for example
  243 entries flip in `live_multiple`), but a small net change.

### 3.4 Generalization (F24, F27, F41)

- Held-out rows: the E1b irrelevance gain on the rows that the fit did not
  use (+17.2 points) is as large as on the fitted rows (F24).
- Tool-swap pairs (858, new combinations of real queries and unrelated
  tools):

| Model and mode | call rate, own tools | call rate, swapped tools |
|---|---|---|
| base, thinking | 0.939 | 0.385 |
| E1b, thinking | 0.923 | 0.226 |
| E4b, thinking | 0.920 | 0.228 |
| base, non-thinking | 0.697 | 0.224 |
| E4b, non-thinking | 0.830 | 0.217 |

### 3.5 Side effects (F30, F33)

- KL at generated positions, nats per token (40 rows):

| Edit | reasoning | inside the tool call |
|---|---|---|
| E1b | 0.0025 | 0.0024 |
| E4b (calibration, F33) | 0.0038 | 0.0012 |
| wsp | 0.0138 | 0.0073 |

- The edits change the generated reasoning 3.6 to 5.5 times less than the
  wsp edit.
- The rank-1 change is dense: about 3.5M weights, relative norm 0.307 for
  E4b. The wsp edit changes 131,072 weights.

## 4. How the model decides to call a tool

### 4.1 Non-thinking mode (F6 to F11)

- The decision is made at the first assistant token. `<tool_call>` is one
  token (id 248058). Its probability predicts the sampled outcome with AUROC
  0.914.
- At the same false-positive rate, the output catches 68.5% of the needed
  calls. A linear probe at the output of layer 12 catches 96.5% (F7).
- Tool-swap test: the model gives a higher call probability to the correct
  tools in 97.9% of 858 pairs (F9).
- The call logit is a query prior plus a relevance gap (+2.9). The prior is
  low for questions that the model "can answer itself", for example "What is
  the capital of Brazil?" (F10). The prior is a separate linear signal
  (F11).

### 4.2 Thinking mode (F13 to F20)

- The decision is made inside the reasoning. After `</think>`, `p_call` is
  almost binary (F13).
- 333 of the 525 over-calls occur only in thinking mode. Some traces
  conclude "answer directly", and then the model calls a tool (F14).
- Most irrelevance traces have no clear conclusion. Then the model falls
  back on "tools are present, so call". An explicit "I will answer
  directly" sentence removes 83% of the over-calls (F17).
- The presence penalty adds about 7 points of over-calling: it penalizes
  the text openers that the reasoning used, and `<tool_call>` is never in
  the reasoning (F15). This is a sampling setting.
- Direct logit attribution: the token mixers of layers 15 to 22 write the
  decision. Over-calls use the same circuit as correct calls (F18).
- Mean patching: the Gated DeltaNet of layer 22 has almost all of the causal
  effect (F19).
- Before a call, the call-writing heads read the format instructions in the
  system prompt. Before a refusal, they read the query and the reasoning
  more (F19).
- Among the called rows, a probe separates "fits" from "over-call" with
  AUROC 0.945. The output gives 0.807 (F20).

### 4.3 Multi-turn (F42, F44, F49)

- In the middle of a tool conversation, the "a tool fits" signal stays high,
  also when the task is done (F44, F45; not confirmed).
- E1b makes about 15% fewer multi-turn calls than base and scores higher.
  The robust part of that gain is small: 8 entries gained and 2 lost in
  both samples, of 800, with no clear pattern (F42, F49).

## 5. The edit

- Site: the input of `layers.23.mlp.down_proj` (3,584-d). The output of this
  MLP goes directly to the final norm, so the effect on the logits is
  linear (D12).
- Form: W' = W − k · u · ρᵀ. This is rank 1.
- u (write): the residual direction that raises the `<tool_call>` logit and
  changes the logits of the most common reasoning, query, and tool-call
  tokens least (regularized least squares, D12, D13).
- ρ (read), E4 (D16, F32):
  - v: a logistic direction for "irrelevance" against "needs a call", fitted
    on the decision tokens of both modes (even rows).
  - Bias proxies: the weights have no bias. ρ = v + Σ cᵢ · p̂ᵢ, with p̂ᵢ the
    mean activation directions of the thinking decision tokens, the
    non-thinking decision tokens, and the background (all position types).
  - The cᵢ set ρ·a = 0 at the midpoint between correct calls and over-calls
    in each mode, and the mean background read to 0.
  - ρ is scaled so that the gap between over-calls and correct calls is
    0.196.
- k = 5 (calibration, F33).

## 6. What did not work, and the lessons

| Variant | What it changed | Result | Lesson |
|---|---|---|---|
| E1 with the plain readout as u | u = g ⊙ W_U[`<tool_call>`] | background KL 1.2 at k = 20, 2nd parallel call fell to 0.29 | protect the common tokens in u |
| E1b | fit on the thinking decision token only | +2.34 thinking, −0.90 non-thinking, calls suppressed (F31) | fit every decision context that the edit will see |
| E2 | also protect the tool-call tokens in u | same as E1b (F26) | the in-call side effect was already small |
| E3, attempt 1 | background penalty, user and reasoning positions only | KL 13 times larger (F28) | sample every position type in a penalty |
| E3, attempt 2 | background penalty, all positions | same as E1b (F30) | most "background KL" was at prompt positions; measure only generated positions |
| E5c | constraint at mid-conversation decision tokens | fewest multi-turn calls, no multi-turn gain, −0.35 / −0.57 against E4b (F48) | fewer calls do not cause the E1b multi-turn gain |
| wsp (RikiyaT) | "thinking loop" direction, layers 15 and 19 | worse in both modes (F12) | an edit for one behavior can damage another; test all categories |

## 7. Limitations

- Most results have one or two samples. Memory and the small categories
  (16 to 24 entries) are noisy.
- No web search. The overall numbers are not comparable to the public
  leaderboard. Qwen reports 25.3 for thinking mode. The settings match is
  not confirmed (F12).
- The read vector and thresholds come from BFCL data. The held-out rows and
  the tool-swap pairs reduce this risk, but do not remove it.
- E5 used BFCL multi-turn traces for its fit. Its multi-turn score is not a
  clean test (D18).
- Some runs have context-length errors (0 to 10 entries), which count as
  failures. Two memory "prereq" entries of base were generated again after
  their dependent entries.
- MATH-500 was not tested. This report does not judge the RikiyaT edit on
  its own goal.
- E4b keeps a live AST cost in thinking mode (−2.45, two samples). Most new
  failures keep the call count and change an argument value (F36).

## 8. Open items

1. The live AST cost in thinking mode (F36).
2. E4b in a real agent loop (for example Codex) is not tested.
3. Multi-turn (30% of the BFCL weight) stays near 5%. A study of its
   failure causes is the largest remaining opportunity.

## 9. Reproduction

All commands run in `v1/wsp` unless stated, and in tmux (D15) with
`v1/tmux_run.sh`. The base directory is
`@BASE@`.

1. BFCL runs: `v1/bfcl/serve.sh <model-dir> <name> <port> <frac> 262144 <seed>`,
   then `v1/bfcl/gen.sh <name>-think-FC <port> single_turn,multi_turn,memory`
   (or `-nothink-FC`). Score with `run_bfcl.py evaluate`. Tables:
   `bfcl/compare_table.py`.
2. Prompt dumps: `../bfcl/.venv/bin/python dump_prompts.py qwen35-base-think-FC prompts-base-think.jsonl`
   and `... qwen35-base-nothink-FC prompts-base-nothink-v2.jsonl`.
3. Row selection: `uv run python attribute_post_think.py <base-dir> prompts-base-think.jsonl attr-post-think-base.pt`.
4. Activations, thinking: `BG_MODE=all uv run python collect_sites.py <base-dir> prompts-base-think.jsonl attr-post-think-base.pt sites-base-allbg.pt`.
5. Activations, non-thinking: `MODE=nothink BG_MODE=all uv run python collect_sites.py <base-dir> prompts-base-nothink-v2.jsonl - sites-base-nothink-allbg.pt`.
6. Protected token list (`e2-common-ids.json`): written by
   `EDIT_VARIANT=e2 uv run python edit_e1.py sweep ...` (D13).
7. E4b: `CUDA_VISIBLE_DEVICES="" uv run python edit_e4.py save <base-dir> sites-base-allbg.pt sites-base-nothink-allbg.pt 5 ../models/qwen35-e4b`.
   Check: only `layers.23.mlp.down_proj.weight` changes; σ1 = 5.04,
   σ2 ≈ 0.002.
8. vLLM on the RTX 5090 needs `CUDA_HOME` and `FLASHINFER_EXTRA_LDFLAGS`
   (D5). `serve.sh` sets both.
9. With several vLLM servers on one GPU, keep the sum of the memory
   fractions at or below 0.85 (progress log, 09:54).
