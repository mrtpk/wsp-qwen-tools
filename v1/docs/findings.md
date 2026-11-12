# Findings

## F1. `modelq` is the official base checkpoint

- The decoded `models-text/modelq.base64` has SHA-256 `04b1c301…4696`. This is
  the LFS hash of `model.safetensors-00001-of-00001.safetensors` in
  `Qwen/Qwen3.5-0.8B` at revision `2fc06364`.
- `models-text/modelq.hex` decodes to the same bytes.
- The model has 873,438,784 parameters. No tensor has NaN or Inf.

## F2. Architecture (from the tensor names and `config.json`)

- The text model has 24 decoder layers and a hidden size of 1024.
- The layer types repeat as 3 × `linear_attention` (Gated DeltaNet), then
  1 × `full_attention`. The full-attention layers are 3, 7, 11, 15, 19, and 23.
- Each layer has a dense MLP with an intermediate size of 3584 (SwiGLU).
- The vocabulary has 248,320 tokens.
- The checkpoint also has a vision encoder (12 blocks, width 768) and an MTP
  head.

## F3. The RikiyaT edit

- The edit changes only `model.language_model.layers.15.mlp.down_proj.weight`
  and `model.language_model.layers.19.mlp.down_proj.weight`.
- Each change is rank 1 on 64 columns. The second singular value of the delta
  is about 1/1700 of the first, which is BF16 rounding.
- The relative norm of the delta is 0.106 in layer 15 and 0.081 in layer 19.
- All other tensors are bit-identical to base.

## F4. BFCL-V4, non-thinking mode (one sample, no web search)

| Measure | base | wsp |
|---|---|---|
| Overall (web search = 0) | 17.56 | 15.66 |
| Non-live AST | 32.65 | 24.31 |
| Live AST | 44.19 | 36.34 |
| Multi-turn | 3.00 | 1.88 |
| Memory | 5.81 | 4.73 |
| Relevance | 75.00 | 50.00 |
| Irrelevance | 78.11 | 80.90 |

Qwen reports 25.3 for BFCL-V4 in thinking mode. The thinking-mode runs of this
study are not complete.

## F5. The "text instead of a tool call" defect

- Definition: The entry needs a call, but the model output has no tool call.
  BFCL gives `ast_decoder:decoder_failed` and the raw result is a string.
- In base, 685 of 2,501 call entries (27.4%) have this defect. It is 47.0% of
  the failures in the single-turn call categories.
- In wsp, 842 of 2,501 (33.7%) have this defect. It is 50.2% of the failures.
- The worst category is `parallel`: 58.5% in base, 65.0% in wsp.
- Example: `simple_python_3` has the tool `algebra.quadratic_roots`. The model
  solves the quadratic equation in LaTeX and does not call the tool.
- Script: `v1/bfcl/defect.py`.

## F6. The decision is made at the first assistant token (H2 confirmed)

- Data: 3,625 non-thinking prompts (2,501 need a call, 1,124 irrelevance).
  The prompts are rebuilt with the chat template. The token counts match vLLM
  for all 3,625 prompts.
- `<tool_call>` is one token, id 248058.
- The AUROC of the first-token `p(<tool_call>)` for the sampled outcome
  "called" is 0.914. For the label "needs a call", it is 0.851.
- When the model does not call, the greedy first token is usually "I" (351)
  or "To" (157). These start a prose answer.
- For the prompts that need a call, the values of `p_call` spread over the full
  range from 0 to 1. They do not split into two groups.
- Scripts: `v1/wsp/probe_first_token.py`, output `acts-base.pt`.

## F7. The residual stream knows more than the output uses

- Table: `v1/wsp/layers-base-all.txt`, `layers-base-live.txt`,
  `ceiling-base.txt`.
- A linear probe for "needs a call" (logistic regression, 5-fold CV) reaches
  AUROC 0.974 at `hs13` (the output of layer 12). It stays at about 0.97 to
  `hs24`.
- One difference-of-means direction reaches AUROC 0.917 at `hs16` to `hs24`.
  The model's own `p_call` gives 0.851.
- At the model false-positive rate on irrelevance (15.6% at `p_call > 0.5`):

| Readout | TPR on "needs a call" | Defect cases caught (of 685) |
|---|---|---|
| model `p_call` | 0.685 | 0.260 |
| difference of means, `hs16` | 0.834 | 0.685 |
| logistic probe, `hs13` | 0.965 | 0.917 |

- On the live subset only, the probe gives 0.969 and `p_call` gives 0.900.
- Confound: the irrelevance prompts come from other datasets. The probe can
  partly detect the prompt style. A controlled tool-swap test is the next
  step.
- The `hs` index: `hs[0]` is the embedding output, `hs[L]` for L = 1 to 23 is
  the output of layer L − 1, and `hs[24]` is the layer 23 output after the
  final norm.
- An AUROC bug: the first version broke ties by row order, so the `hs0` row
  was wrong (0.760). The true value is 0.5. The fix uses average ranks.

## F8. The wsp edit is orthogonal to the call decision (H1 rejected in direct form)

- Script: `v1/wsp/wsp_geometry.py`. Output: `wsp-geometry.txt`.
- Let u be the output direction (left singular vector) of the wsp delta.
  - Layer 15: cos(u, relevance direction at `hs16`) = −0.007.
    cos(u, `<tool_call>` − text readout) = −0.008.
  - Layer 19: the same values are +0.006 and +0.005.
- The residual projection on u is the same for "needs a call" and
  irrelevance (layer 15: −0.341 and −0.334). The loop direction is a constant
  part of this token state and does not carry relevance.
- The effect is indirect. The final logit difference `<tool_call>` − text
  falls from +4.05 to +3.60 (needs a call) and from +1.87 to +1.67
  (irrelevance). Mean `p_call` falls from 0.631 to 0.569 and from 0.208 to
  0.181.
- The relevance direction and the readout have a small cosine: +0.10 at
  `hs16` and +0.06 at `hs20`. The later layers must convert the relevance
  signal into the readout. That conversion is a candidate location of the
  loss.

## F9. Tool-swap test: the relevance signal is real (H3 confirmed)

- Data: `prompts-swap.jsonl`, 858 pairs from `simple_python`, `multiple`, and
  `live_simple`. Each query appears with its own tools and with the tools of
  a random entry of the same category, with no common function name.
  Script: `make_swap.py`, `analyze_swap.py`. Output: `swap-base.txt`.
- The model `p_call` is 0.607 with the own tools and 0.194 with the swapped
  tools. In 97.9% of the pairs, `p_call` is higher with the own tools.
- The unpaired AUROC of `p_call` is 0.862.
- The BFCL difference-of-means direction separates the swap pairs:
  AUROC 0.884 at `hs24`, pair win 0.992.
- cos(BFCL direction, swap direction) is 0.70 at `hs16`, 0.75 at `hs20`, and
  0.77 at `hs24`. At `hs9`, it is −0.09. The relevance signal appears between
  layers 12 and 16.

## F10. The call logit is a query prior plus a relevance gap

- Script: inline in the log of 2026-11-06 16:20. Output: `swap-prior.txt`.
- logit(`p_call`): own tools +0.56 (sd 1.77), swapped tools −2.36 (sd 2.16).
- The correlation of the own logit and the swapped logit over the queries
  is 0.631. Each query has a prior that does not depend on the tools.
- The relevance gap (own − swapped) is +2.92 (sd 1.73).
- Defect cases (own tools, no call in BFCL, n = 254): prior −3.67, gap +2.64.
  Called cases (n = 604): prior −1.81, gap +3.04.
- The defect comes mostly from a low query prior, not from a small gap.
- Low prior: knowledge, translation, and simple physics questions, for
  example "What is the capital of Brazil?" (−9.64).
- High prior: actions and real-time data, for example weather, "play wrecking
  ball", food orders (+2 to +3).
- Interpretation (not confirmed): the model learned the policy "use a tool
  for what I cannot do, and answer what I know". BFCL scores the policy "if a
  fitting tool exists, call it".
- Edit implication: lower the weight of the query prior in the readout and
  keep the relevance gap. A constant push also raises the false calls.

## F11. The query prior is a separate linear signal (H4 supported)

- Script: `prior_direction.py`. Output: `prior-base.txt`.
- Method: ridge regression from the residual at layer L of the swapped
  prompts to logit(`p_call`) of the swapped prompts. Train on 429 pairs,
  test on 429 pairs. λ is selected on an inner split, for each layer.
- Results:

| hs | held-out R² (swap) | corr with own-tool logit | corr with BFCL logit (needs call) | cos(prior, relevance) |
|---|---|---|---|---|
| 9 | 0.449 | 0.548 | 0.349 | 0.150 |
| 13 | 0.643 | 0.749 | 0.720 | 0.108 |
| 16 | 0.723 | 0.868 | 0.812 | 0.166 |
| 20 | 0.674 | 0.816 | 0.711 | 0.109 |
| 22 | 0.866 | 0.931 | 0.874 | 0.250 |
| 24 | 0.954 | 0.953 | 0.954 | 0.252 |

- The prior is readable from layer 13. At `hs16`, it is almost orthogonal to
  the relevance direction (cos 0.17).
- The prior direction from the swap data also predicts the logit on the real
  BFCL prompts that need a call (corr 0.81 at `hs16`).
- Limit: the ridge weight vector is a fitted readout, not one clean
  feature. With 429 training pairs and 1,024 dimensions, the vector has
  noise. A low-rank or sparse version is a next step.
- First problem: with one fixed λ = 1000, the middle layers gave R² near 0.1.
  That λ was too large for the small residual norm of the middle layers.

## F12. BFCL-V4: base compared with wsp, both thinking modes

- One sample for each entry. No web search. Scored on 2026-11-06 16:20.
- The base thinking run was stopped at 4,423 entries and resumed with
  `--max-model-len 262144`. See the log of 2026-11-06 15:50.
- "Overall estimate" = overall + 0.2 × memory. It assumes that web search
  scores the same as memory. It is not a BFCL score.

| Measure | base think | wsp think | base nothink | wsp nothink |
|---|---|---|---|---|
| Overall (web search = 0) | 20.04 | 19.04 | 17.56 | 15.66 |
| Overall estimate | 22.1 | 21.4 | 18.7 | 16.6 |
| Non-live AST | 57.42 | 42.67 | 32.65 | 24.31 |
| Non-live simple | 40.67 | 31.17 | 32.08 | 25.75 |
| Non-live multiple | 76.50 | 70.00 | 45.50 | 41.50 |
| Non-live parallel | 57.00 | 31.50 | 21.50 | 9.50 |
| Non-live parallel multiple | 55.50 | 38.00 | 31.50 | 20.50 |
| Live AST | 55.37 | 47.67 | 44.19 | 36.34 |
| Live simple | 51.16 | 42.64 | 39.53 | 35.66 |
| Live multiple | 56.89 | 49.57 | 45.87 | 37.23 |
| Live parallel (n = 16) | 62.50 | 43.75 | 43.75 | 37.50 |
| Live parallel multiple (n = 24) | 29.17 | 20.83 | 20.83 | 4.17 |
| Multi-turn | 5.38 | 5.38 | 3.00 | 1.88 |
| Multi-turn base | 10.00 | 9.50 | 5.50 | 3.50 |
| Multi-turn missing function | 0.50 | 0.50 | 0.00 | 0.00 |
| Multi-turn missing parameter | 5.50 | 5.00 | 2.50 | 2.50 |
| Multi-turn long context | 5.50 | 6.50 | 4.00 | 1.50 |
| Memory | 10.32 | 12.04 | 5.81 | 4.73 |
| Memory KV | 10.97 | 13.55 | 5.16 | 4.52 |
| Memory vector | 15.48 | 10.32 | 4.52 | 4.52 |
| Memory recursive summary | 4.52 | 12.26 | 7.74 | 5.16 |
| Relevance detection | 75.00 | 68.75 | 75.00 | 50.00 |
| Irrelevance detection | 50.88 | 59.83 | 78.11 | 80.90 |
| Text instead of a call (of 2,501) | 156 (6.2%) | 197 (7.9%) | 685 (27.4%) | 842 (33.7%) |
| Context-length errors | 10 | 11 | 17 | 9 |

- Thinking mode almost removes the "text instead of a call" defect, but
  irrelevance detection falls from 78.11 to 50.88.
- The wsp edit lowers the single-turn call scores in both modes. It raises
  irrelevance detection in both modes. Both effects agree with "wsp calls
  tools less often".
- Qwen reports 25.3 for thinking mode. This harness gives 20.04 (web search
  = 0) or about 22.1 (estimate). The settings match is not confirmed.

## F13. Thinking mode: the decision is made inside the reasoning

- Data: `prompts-base-think.jsonl` (3,625 rows, with the saved
  `reasoning_content`). The sequence is the chat template with
  `enable_thinking=True`, then the reasoning, then `</think>\n\n`.
- Rebuild check: the input token count matches vLLM for all 3,625 rows. For
  the text outputs, `reasoning + "</think>" + content + "<|im_end|>"` matches
  `output_token_count` in 654 of 732 cases. The other cases differ by 1 or 2
  tokens (re-tokenization at the joins).
- Script: `probe_post_think.py`, `analyze_post_think.py`. Output:
  `post-think-base.txt`.
- `p_call` after `</think>` is almost binary: 2,164 rows above 0.9, 588 rows
  below 0.1. The AUROC for the sampled outcome is 0.980.
- Sampled call rate: 0.936 (needs a call), 0.467 (irrelevance). There are 525
  over-calls in 1,124 irrelevance prompts.
- The non-thinking relevance direction (`hs16`) still separates "needs a
  call" at the post-think position (AUROC 0.909).

## F14. Most thinking-mode over-calls come from the step from reasoning to action

- Output: `over-call-origin.txt`, `reasoning-action.txt`.
- 333 of 525 over-calls occur only in thinking mode. For these prompts, the
  non-thinking model would not call a tool. 192 occur in both modes.
- Prompts with non-thinking `p_call < 0.05` (n = 442) still have a
  thinking-mode over-call rate of 0.260.
- The over-call rate rises with the non-thinking `p_call`: 0.260, 0.455,
  0.589, and 0.857 for the buckets [0, 0.05), [0.05, 0.2), [0.2, 0.5), and
  [0.5, 1].
- Examples where the reasoning says "answer directly" or "none of these
  tools fit", and the model still calls a tool: `irrelevance_142`
  (`get_element_properties("Albert Einstein")`), `live_irrelevance_824-317-0`
  (boiling point tool for a freezing-point question), `live_irrelevance_742-254-2`
  (`call_uber` for a question about the complexity of bubble sort).
- A regex classification of the last 400 characters of each trace
  (heuristic, not confirmed):

| needs a call | conclusion | n | mean `p_call` | sampled call rate |
|---|---|---|---|---|
| no | answer | 128 | 0.163 | 0.250 |
| no | tool | 184 | 0.750 | 0.853 |
| no | unclear | 812 | 0.299 | 0.414 |
| yes | answer | 55 | 0.423 | 0.545 |
| yes | tool | 1,398 | 0.956 | 0.987 |
| yes | unclear | 1,048 | 0.804 | 0.889 |

- The conclusion usually sets the action. But when the conclusion is
  "answer", the action still leaks: mean `p_call` 0.163, and the sampled call
  rate is 0.250.
- 72% of the irrelevance traces have no clear conclusion. For these, the
  call rate is 0.414.

## F15. The presence penalty pushes toward `<tool_call>` after thinking

- Mechanism: vLLM applies `presence_penalty` to the generated tokens only.
  Source: `vllm/model_executor/layers/utils.py` lines 68 to 69 and 80. The
  order is penalties, then temperature, then top-k and top-p. Source:
  `vllm/v1/sample/sampler.py` lines 38 to 49.
- In thinking mode, the reasoning contains the common text openers ("The",
  "I", "To"), so they lose 1.5 logits. `<tool_call>` is not in the
  reasoning. In non-thinking mode, no token is generated before the decision.
- Script: `probe_sampling_bias.py`. Output: `sampling-bias.txt`.
- Expected call rate on irrelevance: 0.367 with top-k and top-p only, 0.435
  with the penalty, 0.467 observed. Expected over-calls: 412, 489, observed
  525.
- Expected call rate on "needs a call": 0.891, then 0.919, observed 0.936.
- Effect: about 7 points of irrelevance detection. This is a sampling
  setting, not a weight property.

## F16. A sample of the thinking-mode `value_error:string` failures

- Total: 365. By category: `live_multiple` 226, `live_simple` 51,
  `simple_python` 47, `multiple` 15, `simple_java` 15, `simple_javascript` 11.
- Sample: 36 cases. File: `v1/wsp/value-error-sample.txt`. The
  classification is my reading, not a script.
  - Wrong format or label for the tool: about 20. Examples: "New York"
    against "New York, NY"; "2023-04-25" against "04/25/2023"; "Science
    Fiction" against the enum label "Sci-fi"; "10:00 AM" against "10:00".
  - Wrong content: about 14. Examples: "rent" when the user wants to buy;
    "Las Vegas, NV" for Los Angeles; the year 2024 where the data assumes
    2023; optional parameters with invented values.
  - Bad ground truth: 1 or 2. Example: the expected date "20023-04-2".
- These are content errors, not binary decisions. A direction edit is less
  likely to fix them.

## F17. An explicit conclusion mostly sets the action, but not fully

- Script: `conclusion_test.py`. Output: `conclusion-test-base.txt`,
  `conclusion-test-base.json`.
- Method: take the saved trace, add a conclusion sentence at the end, then
  measure the raw `p_call` after `</think>\n\n` (no penalty, no top-k).

| Group | original | + "I will answer directly without using any tools." | + "I will call the tool." |
|---|---|---|---|
| Irrelevance over-calls (525) | 0.675 | 0.249 (83.4% below 0.5) | 0.790 |
| Needs a call, called (400 random) | 0.920 | 0.581 (36.2% below 0.5) | 0.926 |

- An explicit "answer" conclusion removes most over-calls (H7 mostly
  supported).
- 16.6% of the over-call rows stay above 0.5 with the explicit "answer"
  sentence. The action does not fully follow the stated conclusion (H6
  partly supported).
- With the same "answer" sentence, `p_call` is 0.581 when the tool fits and
  0.249 when it does not. The action step reads the relevance signal, also
  against the stated conclusion.
- At the post-think position, the relevance probe gives AUROC 0.97 and
  `p_call` gives 0.877 (F13). The readout gap from non-thinking mode (F7) is
  also present here.

## F18. Direct logit attribution of the action token (thinking mode)

- Script: `attribute_post_think.py`, `analyze_attribution.py`. Output:
  `attribution-base.txt`.
- Rows: all 1,124 irrelevance prompts and 800 random "needs a call"
  prompts. Groups: need (needs a call, called; n = 745), refuse (irrelevance,
  no call; n = 599), over (irrelevance, called; n = 525).
- Metric: logit(`<tool_call>`) − logit(best other token), split into the
  embedding and 48 parts (one token mixer and one MLP for each layer),
  through the final norm with the RMS of each row fixed.
- Check: the correlation of the total attribution with logit(`p_call`) is
  0.996. The maximum absolute reconstruction error of the normed state is
  0.30 (bf16).
- Mean total: need +4.33, refuse −3.64, over +1.67.
- Large constant parts (about the same in all groups): `L23.attn` −4.1,
  `L23.mlp` +2.4, `L00.gdn` +1.8, `L17.mlp` −1.0.
- The parts that carry relevance into the readout (need − refuse):
  `L22.gdn` +2.29, `L19.attn` +1.52, `L21.gdn` +1.33, `L15.attn` +0.92,
  `L15.mlp` +0.63, `L20.gdn` +0.58, `L20.mlp` +0.55.
- The parts that drive the over-calls (over − refuse): the same list, in the
  same order. `L22.gdn` +1.49, `L19.attn` +1.07, `L21.gdn` +0.86,
  `L15.attn` +0.60.
- Cumulative difference by layer: almost 0 to layer 14. Then it rises in
  layers 15 to 22 to +5.4 (over − refuse) and +8.3 (need − refuse).
- Conclusion: the decision is written by the token mixers of layers 15 to 22.
  The over-calls use the same call circuit as the correct calls, at about 65%
  of the strength. There is no separate over-call pathway.

## F19. Mean patching: `L22.gdn` is the causal switch; heads read the format instructions before a call

- Script: `heads_and_patch.py`, `analyze_heads.py`. Output:
  `heads-patch-base.txt`, `heads-patch-base.json`. 300 rows for each group.
- Method: at the action position, add (target-group mean − own-group mean)
  to the output of a component. The need and over rows go toward the refuse
  mean. The refuse rows go toward the need mean. The effect includes the
  downstream layers.

| Shifted component | need | refuse | over |
|---|---|---|---|
| none | 0.922 | 0.074 | 0.681 |
| `L15.attn` | 0.906 | 0.125 | 0.639 |
| `L19.attn` | 0.943 | 0.048 | 0.724 |
| `L20.gdn` | 0.921 | 0.070 | 0.681 |
| `L21.gdn` | 0.905 | 0.083 | 0.662 |
| `L22.gdn` | 0.714 | 0.333 | 0.505 |
| `L15` + `L19` attn | 0.926 | 0.083 | 0.677 |
| `L20`–`L22` gdn | 0.683 | 0.373 | 0.489 |
| all five | 0.685 | 0.408 | 0.491 |
| control `L17.mlp` | 0.925 | 0.086 | 0.679 |

- `L22.gdn` has almost all of the causal effect. `L15.attn` and `L19.attn`
  have a large direct attribution (F18) but a small total effect. The later
  layers compensate for them.
- Head attribution (largest need − refuse): `L23.h3` −1.64, `L23.h6` +1.12,
  `L23.h4` +0.66, `L23.h2` −0.66, `L15.h7` +0.62, `L23.h7` +0.55,
  `L19.h1` +0.49, `L19.h7` +0.44.
- Attention mass from the action token (for example `L15.h7`):
  instructions 0.84 (need), 0.79 (over), 0.57 (refuse). Reasoning 0.01,
  0.02, 0.05. For `L19.h1`: instructions 0.64, 0.49, 0.28; reasoning 0.08,
  0.19, 0.29.
- The "instructions" segment is the system text after `</tools>`. It has the
  example `<tool_call>` block.
- Before a call, the call-writing heads read the format template. Before a
  refusal, they read the user query and the reasoning more. This agrees
  with H6.
- First problem: a prompt with no `<tools>` block stopped the first run at
  row 500. The fix makes the segment search robust.

## F20. Among the called rows, the residual still separates "fits" from "over-call"

- Output: `over-vs-need-separability.txt`. Rows: 2,867 called rows (2,342
  need, 525 over).
- `p_call` after `</think>`: AUROC 0.807 for need against over.
- Post-think residual: logistic probe 0.933 to 0.947 (`hs9` to `hs24`).
  Difference of means 0.86 (`hs16` and later).
- Non-thinking first-token residual (before any reasoning): logistic probe
  0.95.
- Conclusion: the model has the "this tool does not fit" information when
  it over-calls. The action step does not use it.

## F21. Candidate edit sites: all separate well, but a rank-1 edit has no bias

- Script: `collect_sites.py`, `analyze_sites.py`. Output: `sites-base.txt`,
  `site-constant.txt`. Data: `sites-base.pt`, `site-dirs-base.pt`.
- Sites: inputs of `mlp.down_proj` (layers 20 to 23, 3,584-d) and
  `linear_attn.out_proj` (layers 20 to 22, 2,048-d), at the action position
  and at 8 random background positions (user text or reasoning) of 400
  rows.
- A raw-space logistic direction v (train on even rows, test on odd rows)
  gives held-out AUROC 0.949 to 0.963 for irrelevance against need, and
  0.918 to 0.938 for over against need (called rows only).
- `mlp23`: mean v·a is −0.177 (need), −0.006 (over), +0.081 (refuse).
  Background: mean +0.012, sd 0.146.
- Problem: v·a is not centered between the groups, and the weights have no
  bias. A plain −k·(v·a) push raises the call logit in all groups.
- A bias proxy: the mean-activation direction m̂. At the action position,
  m̂·a has a coefficient of variation of 0.08 to 0.11 (`mlp23`: mean 12.81,
  CV 0.081). At background positions, the mean is lower (`mlp23`: 6.36).

## F22. E1 calibration on held-out rows (thinking mode, raw `p_call`)

- Script: `edit_e1.py sweep`. Output: `e1-sweep.json`, `e1-sweep.log`.
- Rows: 120 need, 120 refuse, 120 over (held-out odd rows), 80 parallel
  rows for the second call, 30 background sequences for the KL (all
  positions).
- τ values: −0.190 (need mean), −0.092 (midpoint), +0.006 (over mean).

| k | τ | need | refuse | over | 2nd parallel call | background KL |
|---|---|---|---|---|---|---|
| 0 | — | 0.910 | 0.076 | 0.669 | 1.000 | 0 |
| 2 | −0.190 | 0.888 | 0.010 | 0.462 | 1.000 | 0.0019 |
| 2 | −0.092 | 0.935 | 0.024 | 0.559 | 1.000 | 0.0014 |
| 2 | +0.006 | 0.965 | 0.053 | 0.652 | 1.000 | 0.0013 |
| 5 | −0.190 | 0.845 | 0.001 | 0.276 | 1.000 | 0.0112 |
| 5 | −0.092 | 0.951 | 0.006 | 0.450 | 1.000 | 0.0096 |
| 5 | +0.006 | 0.987 | 0.046 | 0.628 | 1.000 | 0.0116 |
| 10 | −0.190 | 0.764 | 0.000 | 0.136 | 0.996 | 0.0756 |
| 10 | −0.092 | 0.953 | 0.001 | 0.364 | 1.000 | 0.0960 |
| 10 | +0.006 | 0.991 | 0.056 | 0.605 | 1.000 | 0.1493 |
| 20 | −0.190 | 0.639 | 0.000 | 0.078 | 0.637 | 0.5219 |
| 20 | −0.092 | 0.945 | 0.000 | 0.281 | 0.992 | 0.7155 |
| 20 | +0.006 | 0.991 | 0.078 | 0.577 | 1.000 | 1.0970 |

- With the midpoint threshold, the edit raises the correct calls and lowers
  the over-calls. It sharpens the decision. It does not only change the call
  rate.
- The background KL grows fast with k (about k²).
- Selected for BFCL: E1a (k = 2, τ = −0.092) and E1b (k = 5, τ = −0.092).
- Saved checkpoints: `v1/models/qwen35-e1a`, `v1/models/qwen35-e1b`.
  Check: only `layers.23.mlp.down_proj.weight` changes. The delta is rank 1
  (σ1 = 2.00 and 5.00, σ2 ≈ 0.002 from bf16). The change is dense: about 3.3M
  and 3.5M weights, relative norm 0.122 and 0.305. For comparison, the wsp
  edit changes 131,072 weights in total, with relative norms 0.106 and 0.081.

## F23. E1 on BFCL thinking mode, single-turn only (early result)

- Scored at 18:55 with `--test-category single_turn`. Multi-turn and memory
  were not complete. One sample for each entry.

| Measure | base | wsp | E1a | E1b |
|---|---|---|---|---|
| Irrelevance detection | 50.88 | 59.83 | 63.68 | 70.98 |
| Relevance detection (n = 16) | 75.00 | 68.75 | 81.25 | 75.00 |
| Non-live AST | 57.42 | 42.67 | 57.71 | 57.35 |
| Non-live simple | 40.67 | 31.17 | 39.83 | 39.42 |
| Non-live multiple | 76.50 | 70.00 | 73.00 | 72.50 |
| Non-live parallel | 57.00 | 31.50 | 65.50 | 66.00 |
| Non-live parallel multiple | 55.50 | 38.00 | 52.50 | 51.50 |
| Live AST | 55.37 | 47.67 | 53.74 | 51.74 |
| Live simple | 51.16 | 42.64 | 48.06 | 47.29 |
| Live multiple | 56.89 | 49.57 | 55.37 | 52.99 |
| Live parallel (n = 16) | 62.50 | 43.75 | 50.00 | 56.25 |
| Live parallel multiple (n = 24) | 29.17 | 20.83 | 45.83 | 41.67 |
| Text instead of a call (of 2,501) | 156 | 197 | 139 | 135 |

- E1 raises irrelevance detection by 12.8 (E1a) and 20.1 (E1b) points. It
  keeps non-live AST. It lowers the "text instead of a call" count.
- Cost: live AST falls by 1.6 (E1a) and 3.6 (E1b) points.
- Estimated effect on the overall score from the single-turn part only:
  +1.2 (E1a) and +1.6 (E1b).
- Failure types, `live_multiple`, base against E1b (output:
  `v1/wsp/e1-live-failure-types.txt`): `value_error:string` +29,
  `unexpected_param` +12, `value_error:others` +6. 144 entries newly fail
  and 103 newly pass. The churn is single-sample noise. The net +41
  (3.9 points) is about 2.6 standard errors.
- `live_simple`: "text instead of a call" +14. The edit causes some false
  refusals there.
- Hypothesis (not confirmed): the write direction u was protected only for
  the reasoning and query tokens. Inside a tool call, the edit can move the
  decisions on parameters and on closing the call. This can explain the
  `unexpected_param` rise.

## F24. The E1 irrelevance gain holds on the held-out rows

- Question: the read direction v was fitted on the even rows of the
  irrelevance prompts, and BFCL scores all rows. Is the gain only on the
  fitted rows?
- Output: `v1/wsp/e1-irrelevance-train-test.txt`. The accuracy is
  pooled over `irrelevance` and `live_irrelevance`. The BFCL column
  "Irrelevance Detection" is the unweighted mean of the two categories, so
  the numbers differ (base 50.88 there).

| Model | train rows (562) | held-out rows (562) |
|---|---|---|
| base | 53.0 | 53.6 |
| wsp | 59.1 | 60.9 |
| E1a | 62.5 | 65.3 |
| E1b | 68.0 | 70.8 |

- The held-out gain of E1b is +17.2 points. It is as large as the train gain
  (+15.0). The edit does not overfit the fitted rows.
- Limit: the held-out rows come from the same BFCL datasets. An
  out-of-distribution test is still necessary (the tool-swap set, in
  thinking mode).

## F25. E1 on the full BFCL-V4, thinking mode

- Scored at 19:25. One sample for each entry. No web search. All four runs
  have 0 context errors. Base was scored again after the repair of its 10
  error entries (the repaired entries still fail, so base did not change).
- Output: `v1/wsp/e1-full-table.txt`.

| Measure | base | wsp | E1a | E1b |
|---|---|---|---|---|
| Overall | 20.04 | 19.04 | 22.26 | 22.91 |
| Non-live AST | 57.42 | 42.67 | 57.71 | 57.35 |
| Live AST | 55.37 | 47.67 | 53.74 | 51.74 |
| Multi-turn | 5.38 | 5.38 | 6.38 | 6.75 |
| Multi-turn base | 10.00 | 9.50 | 10.50 | 11.50 |
| Multi-turn missing function | 0.50 | 0.50 | 1.00 | 0.50 |
| Multi-turn missing parameter | 5.50 | 5.00 | 4.50 | 6.00 |
| Multi-turn long context | 5.50 | 6.50 | 9.50 | 9.00 |
| Memory | 10.32 | 12.04 | 14.19 | 14.41 |
| Memory KV | 10.97 | 13.55 | 10.97 | 7.74 |
| Memory vector | 15.48 | 10.32 | 25.16 | 23.87 |
| Memory recursive summary | 4.52 | 12.26 | 6.45 | 11.61 |
| Relevance detection | 75.00 | 68.75 | 81.25 | 75.00 |
| Irrelevance detection | 50.88 | 59.83 | 63.68 | 70.98 |

- E1b: +2.87 overall. The parts: single-turn about +1.6 (solid, F24),
  multi-turn about +0.4 (+1.37 points × 0.3, within the noise of about 2
  points), memory about +0.8 (+4.09 points × 0.2, noisy).
- Multi-turn and memory were not used for the fit. They did not get worse.
- Noise: each multi-turn category has 200 entries and each memory backend
  has 155. The memory vector gain (+8 to +10) may be real. Memory KV falls by
  3.2 for E1b. That can be noise.

## F26. E2 calibration, and the E3 read vector

- E2 (k = 5, midpoint): need 0.951, refuse 0.0056, over 0.444, background KL
  0.0106, in-call KL 0.0028. E1b with the same metric: 0.951, 0.0062, 0.450,
  0.0096, 0.0036. Output: `e2-sweep.json`, `e1-k5-incall.json`. The E1b
  numbers reproduce F22 exactly.
- The in-call KL is small for both. E2 is not tested on BFCL (D14).
- E3 read vector at `mlp23` (script `e3_read.py`, output
  `e3-read-mlp23.txt`). Fit on even rows, evaluated on odd rows. Each read
  vector is scaled to the E1 gap (0.196 on the train rows).

| Read vector | need | over | refuse | AUROC over/need | background mean | background sd | background \|p99\| |
|---|---|---|---|---|---|---|---|
| E1 | −0.088 | +0.088 | +0.171 | 0.930 | +0.058 | 0.146 | 0.407 |
| E3, γ = 0 | −0.026 | +0.075 | +0.048 | 0.757 | 0.000 | 0.651 | 1.815 |
| E3, γ = 0.1 | −0.063 | +0.063 | +0.090 | 0.880 | 0.000 | 0.072 | 0.195 |
| E3, γ = 1 | −0.071 | +0.063 | +0.098 | 0.902 | 0.000 | 0.031 | 0.084 |
| E3, γ = 10 | −0.075 | +0.068 | +0.112 | 0.906 | 0.000 | 0.012 | 0.035 |
| E3, γ = 30 | −0.077 | +0.070 | +0.120 | 0.906 | 0.000 | 0.009 | 0.025 |

- With γ = 10, the background sd is 12 times smaller than for E1, and the
  background mean is 0. The held-out AUROC falls a little (0.906 against
  0.930).
- Without the background penalty (γ = 0), the discriminant overfits (AUROC
  0.757).
- The two bias proxies (mean action activation, mean background activation)
  give a 2 × 2 system with condition number 2.8, so the solution is stable.

## F27. Out-of-distribution test: E1b on the tool-swap prompts (thinking mode)

- Script: `swap_gen.py` through vLLM, with the BFCL sampling. 858 pairs.
  Output: `swap-gen-base.jsonl`, `swap-gen-e1b.jsonl`.
- The swapped-tool prompts are new combinations. The fit never saw them.

| Model | call rate, own tools | call rate, swapped tools |
|---|---|---|
| base | 0.939 | 0.385 |
| E1b | 0.923 | 0.226 |

- E1b removes 41% of the false calls on new prompts (0.385 to 0.226). The
  correct-call rate falls by 1.6 points. The sampling noise is about 1.6
  points for each number.
- Conclusion: the E1b irrelevance gain generalizes beyond the BFCL
  irrelevance datasets.

## F28. E3 failed: the side effect is larger, not smaller

- Calibration (`edit_e3.py sweep`, γ = 10, E2 write direction), output
  `e3-sweep.log`. Stopped after k = 5. The higher k values can only be worse.

| Edit (k = 5) | need | refuse | over | 2nd parallel call | background KL | in-call KL |
|---|---|---|---|---|---|---|
| E1b | 0.951 | 0.006 | 0.450 | 1.000 | 0.0096 | 0.0036 |
| E3 | 0.939 | 0.018 | 0.451 | 0.997 | 0.1215 | 0.0811 |

- The prediction was about 100 times smaller. The result is about 13 times
  larger (background) and 22 times larger (in-call).
- Cause (probable, not confirmed): the background sample of F21 has only
  user-text and reasoning positions. The KL test uses all positions,
  including the system prompt (tool schemas, format instructions) and the
  tool calls. The discriminant made the read vector quiet where it was
  sampled. The variance moved to the positions that were not sampled.
- Lesson: a background penalty must sample every type of position that the
  edit acts on.

## F29. E3 read vectors with the full background

- Data: `sites-base-allbg.pt` (`collect_sites.py`, `BG_MODE=all`). 3,972
  background rows: 8 positions from the whole sequence of 400 rows, plus 4
  positions inside the model's own first tool call where it exists.
- Output: `e3-read-mlp23-allbg.txt`.

| Read vector | need | over | refuse | AUROC over/need | background mean | background sd | background \|p99\| |
|---|---|---|---|---|---|---|---|
| E1 | −0.080 | +0.085 | +0.164 | 0.926 | +0.099 | 0.185 | 0.579 |
| E3, γ = 1 | −0.076 | +0.060 | +0.101 | 0.907 | 0.000 | 0.088 | 0.217 |
| E3, γ = 10 | −0.080 | +0.072 | +0.115 | 0.931 | 0.000 | 0.065 | 0.154 |
| E3, γ = 30 | −0.082 | +0.077 | +0.125 | 0.937 | 0.000 | 0.052 | 0.124 |

- With the full background, E1 is noisier than F26 showed (sd 0.185
  against 0.146). The E1 row uses m̂ from the new background, so it differs a
  little from the E1b checkpoint.
- E3 with γ = 30: AUROC 0.937 (better than E1), background mean 0, and a
  mean-square background read of 0.0027 against 0.044 for E1 (16 times
  smaller).
- Next: the E3 calibration with γ = 30 and k = 5, 10, 20 (`e3b-sweep.json`).

## F30. E3 calibration and the KL by segment

- E3 calibration (γ = 30, full background, E2 write direction), output
  `e3b-sweep.json`:

| k | need | refuse | over | 2nd parallel call | background KL | in-call KL |
|---|---|---|---|---|---|---|
| 0 | 0.910 | 0.076 | 0.669 | 1.000 | 0 | 0 |
| 5 | 0.946 | 0.007 | 0.450 | 1.000 | 0.0104 | 0.0021 |
| 10 | 0.939 | 0.002 | 0.363 | 0.991 | 0.0750 | 0.0470 |
| 20 | 0.933 | 0.002 | 0.279 | 0.711 | 0.5506 | 0.1858 |

- At k = 5, E3 is almost the same as E1b. A 16 times smaller mean-square
  background read did not lower the background KL.
- KL by segment (`segment_kl.py`, 40 rows, output `segment-kl.txt`). Mean KL
  per position:

| Edit | system (n 31,070) | user (n 1,386) | reasoning (n 10,146) | action (n 20) | call (n 1,137) | largest single position |
|---|---|---|---|---|---|---|
| E1a | 0.0013 | 0.0058 | 0.0007 | 0.054 | 0.0004 | 0.57 (action) |
| E1b | 0.0091 | 0.0334 | 0.0025 | 0.326 | 0.0024 | 4.02 (system) |
| E3 k = 5 | 0.0070 | 0.0791 | 0.0023 | 0.177 | 0.0048 | 18.86 (system) |
| wsp | 0.1341 | 0.0525 | 0.0138 | 0.011 | 0.0073 | 7.76 (system) |

- Most of the "background KL" is at prompt positions. The model reads them
  but does not generate them, so they do not change the outputs.
- At the generated positions, E1b changes the reasoning by 0.0025 nats per
  token (about 0.6 nats over 250 tokens). The wsp edit changes it 5.5 times
  more.
- E3 is not better than E1b: the same reasoning KL, twice the in-call KL,
  and one extreme position. E3 is not tested on BFCL.
- The live AST cost of E1b probably does not come from changed reasoning.
  Candidates: the decision change itself (`live_simple` false refusals +14)
  and single-sample noise in the value errors. Not confirmed.
- Correction to D14: the background KL of the sweep is a poor side-effect
  metric, because it averages over prompt positions. Use the reasoning and
  in-call KL.

## F31. E1b fails the non-thinking side-effect check

- Run: BFCL-V4 non-thinking for E1b, 5,017 entries, 0 errors. Base
  non-thinking was scored again after its repair (overall 17.56 to 17.63,
  multi-turn 3.00 to 3.25). Output: `v1/wsp/e1b-nothink-table.txt`.

| Measure | base | wsp | E1b |
|---|---|---|---|
| Overall | 17.63 | 15.66 | 16.73 |
| Non-live AST | 32.65 | 24.31 | 27.65 |
| Non-live simple | 32.08 | 25.75 | 15.58 |
| Non-live multiple | 45.50 | 41.50 | 30.00 |
| Non-live parallel | 21.50 | 9.50 | 24.50 |
| Non-live parallel multiple | 31.50 | 20.50 | 40.50 |
| Live AST | 44.19 | 36.34 | 29.76 |
| Live simple | 39.53 | 35.66 | 16.28 |
| Live multiple | 45.87 | 37.23 | 33.05 |
| Live parallel (n = 16) | 43.75 | 37.50 | 25.00 |
| Live parallel multiple (n = 24) | 20.83 | 4.17 | 33.33 |
| Multi-turn | 3.25 | 1.88 | 2.12 |
| Memory | 5.81 | 4.73 | 5.16 |
| Relevance detection (n = 16) | 75.00 | 50.00 | 43.75 |
| Irrelevance detection | 78.11 | 80.90 | 93.18 |
| Text instead of a call (of 2,501) | 685 | 842 | 1,252 |

- In non-thinking mode, E1b suppresses calls: "text instead of a call"
  rises from 27.4% to 50.1% of the call entries. Irrelevance detection rises
  to 93.18. By D10, this is only a lower call rate.
- Overall: −0.9 points. E1b fails the side-effect condition of D11.
- Cause (probable, not confirmed): v, m̂, and τ were fitted only at the
  thinking-mode action token (after the reasoning and `</think>\n\n`). The
  non-thinking decision token (after `<think>\n\n</think>\n\n` with no
  reasoning) has other activations. The threshold is wrong there.
- Note: the parallel categories rise (non-live parallel multiple 31.50 to
  40.50). The edit lowers the wrong extra calls there.

## F32. The E4 read vector works in both modes; E1b does not

- Data: `sites-base-allbg.pt` (thinking, 1,924 rows) and
  `sites-base-nothink-allbg.pt` (non-thinking, 3,625 rows, from
  `prompts-base-nothink-v2.jsonl`). Both have the full background.
- Script: `edit_e4.py read`. Output: `e4-read.txt`, `tmuxlogs/e4-read.log`.
- Held-out mean read values at the decision token:

| Read vector | mode | need | over | refuse | AUROC over/need | bg mean | bg sd |
|---|---|---|---|---|---|---|---|
| E1b | thinking | −0.088 | +0.088 | +0.171 | 0.930 | +0.067 | 0.181 |
| E1b | non-thinking | +0.094 | +0.217 | +0.267 | 0.880 | +0.069 | 0.191 |
| E4 | thinking | −0.086 | +0.083 | +0.154 | 0.952 | +0.020 | 0.191 |
| E4 | non-thinking | −0.091 | +0.074 | +0.148 | 0.953 | −0.019 | 0.195 |

- This confirms the cause of F31: in non-thinking mode, E1b reads +0.094 for
  the prompts that need a call. That is above its threshold, so the edit
  pushes them toward "no call".
- E4 puts the threshold between need and over in both modes. Its AUROC is
  also higher (0.95 in both modes).
- The 3 × 3 system for the three bias proxies has condition number 35.7.

## F33. E4 calibration: both modes improve

- Script: `edit_e4.py calibrate`. Output: `e4-cal.json`,
  `tmuxlogs/e4-cal.log`. 120 held-out rows for each group and mode. The KL
  uses only the generated segments (F30): 40 rows.

| k | think need | think over | think refuse | nothink need | nothink over | nothink refuse | reasoning KL | in-call KL |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.910 | 0.683 | 0.070 | 0.779 | 0.522 | 0.117 | 0 | 0 |
| 2 | 0.935 | 0.577 | 0.025 | 0.869 | 0.410 | 0.069 | 0.0009 | 0.0002 |
| 5 | 0.953 | 0.458 | 0.008 | 0.926 | 0.312 | 0.049 | 0.0038 | 0.0012 |
| 8 | 0.958 | 0.386 | 0.004 | 0.949 | 0.267 | 0.046 | 0.0152 | 0.0131 |

- In both modes, E4 raises the correct calls and lowers the over-calls.
- In non-thinking mode at k = 5, the correct-call probability rises from
  0.779 to 0.926. E1b lowered it (F31).
- Selected: E4b, k = 5. Checkpoint `v1/models/qwen35-e4b`. Check: only
  `layers.23.mlp.down_proj.weight` changes, rank 1 (σ1 = 5.04, σ2 = 0.002),
  relative norm 0.307.

## F34. E4b passes the non-thinking side-effect check

- Run: BFCL-V4 non-thinking for E4b, 5,017 entries, 5 context errors
  (counted as failures). Scored at 03:16. Output:
  `v1/wsp/e4b-nothink-table.txt`.

| Measure | base | wsp | E1b | E4b |
|---|---|---|---|---|
| Overall | 17.63 | 15.66 | 16.73 | 20.02 |
| Non-live AST | 32.65 | 24.31 | 27.65 | 48.58 |
| Non-live simple | 32.08 | 25.75 | 15.58 | 37.33 |
| Non-live multiple | 45.50 | 41.50 | 30.00 | 60.00 |
| Non-live parallel | 21.50 | 9.50 | 24.50 | 44.00 |
| Non-live parallel multiple | 31.50 | 20.50 | 40.50 | 53.00 |
| Live AST | 44.19 | 36.34 | 29.76 | 45.82 |
| Live simple | 39.53 | 35.66 | 16.28 | 38.37 |
| Live multiple | 45.87 | 37.23 | 33.05 | 48.15 |
| Live parallel (n = 16) | 43.75 | 37.50 | 25.00 | 43.75 |
| Live parallel multiple (n = 24) | 20.83 | 4.17 | 33.33 | 25.00 |
| Multi-turn | 3.25 | 1.88 | 2.12 | 3.25 |
| Memory | 5.81 | 4.73 | 5.16 | 4.73 |
| Relevance detection (n = 16) | 75.00 | 50.00 | 43.75 | 56.25 |
| Irrelevance detection | 78.11 | 80.90 | 93.18 | 86.59 |
| Text instead of a call (of 2,501) | 685 | 842 | 1,252 | 332 |

- E4b: +2.39 overall in non-thinking mode. The "text instead of a call"
  count falls by half, and irrelevance detection rises by 8.5 points. This
  is the sharper-decision pattern, not the call-less pattern of D10.
- Memory −1.08: within the noise (155 entries for each backend).
- The thinking-mode run of E4b is not complete.

## F35. E4b in thinking mode, and the two-mode comparison

- Run: BFCL-V4 thinking for E4b, 5,017 entries, 0 errors. Scored at 03:47.
  Output: `v1/wsp/e4b-think-table.txt`.

| Measure | base | wsp | E1b | E4b |
|---|---|---|---|---|
| Overall | 20.04 | 19.04 | 22.91 | 21.40 |
| Non-live AST | 57.42 | 42.67 | 57.35 | 59.23 |
| Live AST | 55.37 | 47.67 | 51.74 | 52.85 |
| Live simple | 51.16 | 42.64 | 47.29 | 44.57 |
| Live multiple | 56.89 | 49.57 | 52.99 | 54.89 |
| Multi-turn | 5.38 | 5.38 | 6.75 | 5.63 |
| Multi-turn base | 10.00 | 9.50 | 11.50 | 8.00 |
| Multi-turn long context | 5.50 | 6.50 | 9.00 | 8.50 |
| Memory | 10.32 | 12.04 | 14.41 | 8.60 |
| Relevance detection (n = 16) | 75.00 | 68.75 | 75.00 | 87.50 |
| Irrelevance detection | 50.88 | 59.83 | 70.98 | 67.84 |
| Text instead of a call (of 2,501) | 156 | 197 | 135 | 97 |

- Change of the overall score against base:

| Edit | thinking | non-thinking |
|---|---|---|
| wsp | −1.00 | −1.97 |
| E1b | +2.87 | −0.90 |
| E4b | +1.36 | +2.39 |

- E4b is the only edit that improves both modes.
- In thinking mode, E4b is equal to or better than E1b on non-live AST,
  live AST, and "text instead of a call". It is 3.1 points lower on
  irrelevance detection.
- The overall gap between E1b and E4b in thinking mode (1.51) comes mostly
  from memory (5.81 × 0.2 = 1.16) and multi-turn (1.12 × 0.3 = 0.34). These
  categories have 155 to 200 entries each and one sample. The gap is not
  confirmed.

## F36. The E4b live AST cost (thinking) is not a decision change

- Output: `v1/wsp/e4b-live-failure-types.txt`. Base thinking against
  E4b thinking, one sample each.
- `live_simple`: failures 126 → 143. 35 entries newly fail and 18 newly pass.
  Of the 35, 32 have the same number of calls as base (a changed argument or
  function name), and 3 have no call now. The largest change is
  `value_error:string` +15.
- `live_multiple`: failures 454 → 475. 148 newly fail and 127 newly pass. Of
  the 148, 131 have the same number of calls, 9 have no call now, and 8 have
  another call count.
- Conclusion: the cost is not a change of the call decision. It is a change
  of the argument content. The large two-way churn suggests sampling noise.
  Test: compare with the base-against-base change of the second sample
  (D17).

## F37. The noise floor: base thinking, sample 1 against sample 2

- Sample 2 uses vLLM seed 1 (D17). 1 context error. Output:
  `v1/wsp/base-noise.txt`.

| Measure | base s1 | base s2 | s2 − s1 |
|---|---|---|---|
| Overall | 20.04 | 20.88 | +0.84 |
| Non-live AST | 57.42 | 56.31 | −1.11 |
| Live AST | 55.37 | 55.59 | +0.22 |
| Live simple | 51.16 | 50.00 | −1.16 |
| Live multiple | 56.89 | 57.17 | +0.28 |
| Multi-turn | 5.38 | 5.50 | +0.12 |
| Memory | 10.32 | 14.62 | +4.30 |
| Memory KV | 10.97 | 9.68 | −1.29 |
| Memory vector | 15.48 | 18.06 | +2.58 |
| Memory recursive summary | 4.52 | 16.13 | +11.61 |
| Irrelevance detection | 50.88 | 51.20 | +0.32 |

- Memory is very noisy: +4.30 between two runs of the same model. The E1b
  memory gain (+4.09, F25) cannot be separated from the noise.
- Live churn between the two base runs: `live_simple` 34 newly fail and 31
  newly pass. `live_multiple` 120 and 123. From this churn, the standard
  deviation of a net change is about 8 entries (3.1 points) in
  `live_simple` and about 16 entries (1.5 points) in `live_multiple`.
- The E4b thinking live costs (−6.59 in `live_simple`, −2.00 in
  `live_multiple`, F35) are about 2.1σ and 1.3σ. They are borderline.
- Irrelevance detection is stable (+0.32), so the gains of E1b and E4b
  there are real.
- Incident: the third server (`qwen35-e4b-s2`) failed at startup with CUDA
  out of memory. Each of the first two servers used about 10.6 GB, more than
  0.30 × 31.36 GB, because of the CUDA context. The E4b server was started
  again after the base-s2 run, at 09:56.

## F38. Two-sample means, thinking mode: base and E1b (E4b pending)

- E1b sample 2: 0 errors. Output: `v1/wsp/s2-partial.txt`.

| Measure | base s1 | base s2 | base mean | E1b s1 | E1b s2 | E1b mean | E1b − base |
|---|---|---|---|---|---|---|---|
| Overall | 20.04 | 20.88 | 20.46 | 22.91 | 22.70 | 22.80 | +2.35 |
| Non-live AST | 57.42 | 56.31 | 56.87 | 57.35 | 57.08 | 57.22 | +0.35 |
| Live AST | 55.37 | 55.59 | 55.48 | 51.74 | 51.74 | 51.74 | −3.74 |
| Multi-turn | 5.38 | 5.50 | 5.44 | 6.75 | 7.62 | 7.19 | +1.75 |
| Memory | 10.32 | 14.62 | 12.47 | 14.41 | 11.18 | 12.79 | +0.33 |
| Irrelevance detection | 50.88 | 51.20 | 51.04 | 70.98 | 72.91 | 71.94 | +20.90 |

- The E1b memory gain of F25 (+4.09) was noise. The two-sample difference is
  +0.33.
- The E1b live AST cost is real: 51.74 in both samples, against 55.37 and
  55.59 for base.
- The E1b multi-turn gain is probably real: both E1b samples (6.75, 7.62)
  are above both base samples (5.38, 5.50).

## F39. Two-sample means, thinking mode: base, E1b, E4b

- E4b sample 2: 1 context error. Output: `v1/wsp/s2-final.txt`
  (`bfcl/two_sample_table.py`).

| Measure | base s1 | base s2 | E1b s1 | E1b s2 | E4b s1 | E4b s2 | base mean | E1b mean | E4b mean | E1b − base | E4b − base |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Overall | 20.04 | 20.88 | 22.91 | 22.70 | 21.40 | 21.59 | 20.46 | 22.80 | 21.49 | +2.34 | +1.03 |
| Non-live AST | 57.42 | 56.31 | 57.35 | 57.08 | 59.23 | 58.13 | 56.87 | 57.22 | 58.68 | +0.35 | +1.81 |
| Live AST | 55.37 | 55.59 | 51.74 | 51.74 | 52.85 | 53.22 | 55.48 | 51.74 | 53.03 | −3.74 | −2.45 |
| Live simple | 51.16 | 50.00 | 47.29 | 50.39 | 44.57 | 48.06 | 50.58 | 48.84 | 46.31 | −1.74 | −4.27 |
| Live multiple | 56.89 | 57.17 | 52.99 | 52.52 | 54.89 | 55.18 | 57.03 | 52.76 | 55.03 | −4.27 | −2.00 |
| Multi-turn | 5.38 | 5.50 | 6.75 | 7.62 | 5.63 | 5.12 | 5.44 | 7.19 | 5.38 | +1.75 | −0.06 |
| Memory | 10.32 | 14.62 | 14.41 | 11.18 | 8.60 | 8.82 | 12.47 | 12.79 | 8.71 | +0.33 | −3.76 |
| Relevance (n = 16) | 75.00 | 75.00 | 75.00 | 87.50 | 87.50 | 68.75 | 75.00 | 81.25 | 78.12 | +6.25 | +3.12 |
| Irrelevance detection | 50.88 | 51.20 | 70.98 | 72.91 | 67.84 | 71.55 | 51.04 | 71.94 | 69.69 | +20.90 | +18.65 |

- Both edits are above base in both samples.
- E1b is stronger in thinking mode (+2.34 against +1.03). Its multi-turn
  gain is consistent (both samples above both base samples).
- E4b is lower in memory in both samples (8.60, 8.82 against 10.32,
  14.62). Correction (F40): this is about 1.3σ. Not confirmed.
- Live AST cost: E1b −3.74 (51.74 in both samples), E4b −2.45. Both are
  consistent over the samples.
- Across the two modes (non-thinking from one sample, F34, F31): E4b +1.03
  and +2.39, E1b +2.34 and −0.90. E4b is the better general edit. E1b fails
  the non-thinking check of D11.
- Open question: why does E4b lose memory and multi-turn gain in thinking
  mode, where E1b does not?

## F40. The E4b memory gap is not a call-rate effect, and it is not confirmed

- Output: `v1/wsp/memory-calls.txt`, `memory-backends.txt`.
- Mean tool calls per prereq conversation (the storing phase): base 37.8 /
  40.0 (kv), 31.5 / 33.9 (vector), 9.5 / 11.7 (rec_sum). E4b: 37.5 / 34.3,
  33.2 / 29.5, 18.0 / 16.4. E1b: 40.7 / 46.9, 30.7 / 32.5, 11.8 / 11.6. No
  prereq row has 0 calls, except one E1b rec_sum row.
- Mean tool calls per question row: base 4.9 / 4.6 (kv), E4b 4.5 / 4.2, E1b
  3.3 / 3.4. E4b does not call less than base. E1b calls less and keeps the
  base accuracy.
- Accuracy per backend, two-sample mean: base kv 10.3, vector 16.8, rec_sum
  10.3. E1b 7.1, 20.3, 11.0. E4b 7.1, 12.6, 6.5.
- Noise: base s1 against s2 differs by 4.30 points in memory, so the SD of
  one memory sample is about 3.0 points, and the SD of a difference of two
  two-sample means is also about 3.0 points. The E4b gap (−3.76) is about
  1.3σ. Not confirmed.
- By the same method for multi-turn (800 entries, base s1 against s2
  differs by 0.12): the E1b gain (+1.75) is about 2.2σ. Probably real.

## F41. Tool-swap test for E4b in both modes (out of distribution)

- Script: `swap_gen.py` (mode argument), 858 pairs, BFCL sampling, vLLM seed
  0. Output: `swap-gen-e4b-think.jsonl`, `swap-gen-e4b-nothink.jsonl`,
  `swap-gen-base-nothink.jsonl`, `tmuxlogs/swap-e4b.log`,
  `tmuxlogs/swap-base-nothink.log`.

| Model and mode | call rate, own tools | call rate, swapped tools |
|---|---|---|
| base, thinking (F27) | 0.939 | 0.385 |
| E1b, thinking (F27) | 0.923 | 0.226 |
| E4b, thinking | 0.920 | 0.228 |
| base, non-thinking | 0.697 | 0.224 |
| E4b, non-thinking | 0.830 | 0.217 |

- Thinking: E4b removes 41% of the false calls (0.385 to 0.228), the same as
  E1b. The correct calls fall by 1.9 points.
- Non-thinking: E4b raises the correct calls by 13.3 points (0.697 to
  0.830). The false calls do not rise (0.224 to 0.217).
- E4b fixes both target faults on prompts that the fit never saw:
  over-calling in thinking mode, and "answer it myself" in non-thinking mode.

## F42. Multi-turn: E1b makes fewer calls and scores higher; E4b makes more

- Output: `v1/wsp/multiturn-compare.txt`. Thinking mode, two samples.

| Category | base acc (s1, s2) | E1b acc | E4b acc | base calls/entry | E1b calls/entry | E4b calls/entry |
|---|---|---|---|---|---|---|
| `multi_turn_base` | 10.0, 10.0 | 11.5, 12.0 | 8.0, 9.5 | 12.2 | 10.4 | 13.1 |
| `multi_turn_miss_func` | 0.5, 1.0 | 0.5, 0.5 | 1.0, 0.5 | 13.9 | 11.3 | 15.0 |
| `multi_turn_miss_param` | 5.5, 4.5 | 6.0, 9.5 | 5.0, 5.0 | 15.3 | 12.4 | 15.1 |
| `multi_turn_long_context` | 5.5, 6.5 | 9.0, 8.5 | 8.5, 5.5 | 11.8 | 10.2 | 12.8 |

- E1b makes about 15% fewer calls than base in all four categories, and it
  has the higher accuracy. E4b makes about 5% more calls than base.
- The BFCL multi-turn score file has no `error_type` field, so the failure
  types are not available from this script.
- Hypothesis H9 (not confirmed): most multi-turn decisions are made in the
  middle of a conversation, after a tool result. Both fits used only the
  first decision of each prompt. E1b pushes the mid-conversation decisions
  toward "stop". E4b pushes them toward "call again".
- Test: the read values of the E1b and E4b read vectors at the
  mid-conversation decision points of the base multi-turn traces.

## F43. Two-sample means, non-thinking mode: the E4b gain holds

- E4b sample 2: 10 context errors (counted as failures). Output:
  `v1/wsp/s2-nothink-final.txt`.

| Measure | base s1 | base s2 | E4b s1 | E4b s2 | base mean | E4b mean | E4b − base |
|---|---|---|---|---|---|---|---|
| Overall | 17.63 | 17.74 | 20.02 | 20.56 | 17.68 | 20.29 | +2.61 |
| Non-live AST | 32.65 | 35.31 | 48.58 | 50.52 | 33.98 | 49.55 | +15.57 |
| Live AST | 44.19 | 43.60 | 45.82 | 47.08 | 43.89 | 46.45 | +2.56 |
| Live simple | 39.53 | 36.82 | 38.37 | 42.25 | 38.17 | 40.31 | +2.14 |
| Live multiple | 45.87 | 45.68 | 48.15 | 48.53 | 45.77 | 48.34 | +2.57 |
| Multi-turn | 3.25 | 3.62 | 3.25 | 3.38 | 3.44 | 3.31 | −0.12 |
| Memory | 5.81 | 5.16 | 4.73 | 5.59 | 5.48 | 5.16 | −0.32 |
| Irrelevance detection | 78.11 | 77.33 | 86.59 | 86.72 | 77.72 | 86.66 | +8.94 |

- "Text instead of a call": base 685 / 699, E4b 332 / 328.
- Both E4b samples are far above both base samples. The non-thinking gain
  (+2.61) is confirmed.
- E4b, two samples in each mode: +1.03 (thinking), +2.61 (non-thinking).

## F44. Multi-turn decision tokens: E1b pushes toward "stop", E4b toward "call again" (H9 supported)

- Script: `mt_reads.py` on 1,000 random rows of `mt-decisions-base.jsonl`
  whose rebuilt prompt matches the vLLM token count. Output:
  `tmuxlogs/mt-reads.log`.
- A read value > 0 lowers the `<tool_call>` logit (toward "stop"). A value
  < 0 raises it.

| Group | n | E1b mean (share > 0) | E4b mean (share > 0) |
|---|---|---|---|
| step 0, called | 220 | −0.036 (0.323) | −0.050 (0.164) |
| step 0, stopped | 14 | +0.058 (0.929) | +0.007 (0.571) |
| later step, called | 546 | +0.050 (0.758) | +0.025 (0.645) |
| later step, stopped | 220 | +0.039 (0.741) | −0.053 (0.268) |

- Base `p_call` at the later steps: 0.870 (called), 0.138 (stopped).
- E1b pushes almost all later steps toward "stop" and does not separate
  called from stopped. This agrees with its 15% fewer multi-turn calls and
  its higher multi-turn score (F42).
- E4b pushes the later steps where base stopped (the end-of-turn text
  answers) toward "call again". This agrees with its 5% more calls (F42).
- Neither fit saw a mid-conversation position. Both effects are side
  effects.

## F45. E5 read vectors: control the mid-conversation reads

- Data: `mt-reads-base.pt` (all 2,757 exact-match rows, with the `mlp23`
  activations). The full run reproduces F44 (for example E4b, later stopped:
  −0.054, share > 0 0.256).
- First design (one proxy, the mean of all later steps, target t_mid):
  wrong. E4 already has a mean of about 0 there. Its fault is the spread:
  later called +0.021, later stopped −0.056. A higher t_mid moves both
  groups, and the stopped steps stay negative (−0.026 at t_mid = 0.045).
  Output: `e5-read.txt`.
- Second design (two proxies: the mean of later-called and of
  later-stopped activations, each with a target). Output: `e5-read2.txt`.
  The 5 × 5 system has condition number 70.4.

| Targets (called : stopped) | held-out later called | held-out later stopped | AUROC think / nothink |
|---|---|---|---|
| E4 (none) | +0.025 | −0.053 | 0.952 / 0.953 |
| 0 : 0 | +0.006 | +0.002 | 0.950 / 0.949 |
| 0.045 : 0.045 | +0.049 | +0.047 | 0.949 / 0.946 |
| 0.03 : 0.06 | +0.035 | +0.061 | 0.945 / 0.941 |

- The targets transfer to the held-out conversations (odd entry IDs). The
  single-turn behavior is almost the same as E4.
- Interpretation (not confirmed): in a tool conversation, the "a tool
  fits" feature is high, also when the task is done. So the E4 read says
  "needs a call" at the steps where the model should stop.

## F46. E5 calibration: the mid-conversation fault of E4, and its repair

- Script: `edit_e5.py calibrate`, k = 5. Output: `e5-cal.json`,
  `tmuxlogs/e5-cal.log`. 100 held-out rows for each single-turn group and
  mode, 100 held-out later-step rows (odd entry IDs) for each multi-turn
  group.

| Edit | think need | think over | nothink need | nothink over | mt later called | mt later stopped | reasoning KL | in-call KL |
|---|---|---|---|---|---|---|---|---|
| base | 0.902 | 0.661 | 0.754 | 0.503 | 0.890 | 0.131 | 0 | 0 |
| E4 | 0.950 | 0.429 | 0.907 | 0.287 | 0.860 | 0.356 | 0.0038 | 0.0017 |
| E5 0 : 0 | 0.950 | 0.447 | 0.907 | 0.296 | 0.867 | 0.235 | 0.0040 | 0.0018 |
| E5 0.045 : 0.045 | 0.946 | 0.441 | 0.907 | 0.294 | 0.780 | 0.141 | 0.0044 | 0.0018 |
| E5 0.03 : 0.06 | 0.946 | 0.449 | 0.906 | 0.298 | 0.797 | 0.126 | 0.0044 | 0.0018 |

- E4 raises the call probability at the steps where base stops from 0.131
  to 0.356. This is the cause of its extra multi-turn calls (F42).
- E5 0 : 0 does not fully repair this (0.235). A read with mean 0 but a
  spread still raises the low probabilities.
- E5 0.03 : 0.06 restores the stopped steps to base (0.126). It lowers the
  mid-conversation calls (0.890 to 0.797), like E1b.
- All E5 variants keep the single-turn effect of E4 in both modes.
- Selected: E5c = E5 0.03 : 0.06, k = 5. Checkpoint
  `v1/models/qwen35-e5c`. Check: only
  `layers.23.mlp.down_proj.weight` changes, rank 1 (σ1 = 5.13, σ2 = 0.002),
  relative norm 0.312.

## F47. E5c in non-thinking mode (one sample)

- 4 context errors. Output: `v1/wsp/e5c-nothink-table.txt`
  (`bfcl/compare_table.py`).

| Measure | base (2) | E1b (1) | E4b (2) | E5c (1) |
|---|---|---|---|---|
| Overall | 17.68 | 16.73 | 20.29 | 19.72 |
| Non-live AST | 33.98 | 27.65 | 49.55 | 45.15 |
| Live AST | 43.89 | 29.76 | 46.45 | 46.71 |
| Live simple | 38.17 | 16.28 | 40.31 | 40.31 |
| Live multiple | 45.77 | 33.05 | 48.34 | 48.81 |
| Multi-turn | 3.44 | 2.12 | 3.31 | 2.63 |
| Memory | 5.48 | 5.16 | 5.16 | 5.38 |
| Irrelevance detection | 77.72 | 93.18 | 86.66 | 86.70 |

- E5c: +2.04 against base, −0.57 against E4b. Non-live AST is below both
  E4b samples (45.15 against 48.58 and 50.52). Multi-turn is 0.68 lower.
- Probable cause (not confirmed): the mid-conversation fit of E5 used only
  thinking-mode traces.
- "Text instead of a call": 360 (E4b: 332, 328).

## F48. E5c in thinking mode: fewer multi-turn calls, but no multi-turn gain

- One sample, 0 context errors. The first evaluation failed: one entry of a
  category was missing (199 of 200). A second `gen.sh` run generated only
  that entry. Output: `v1/wsp/e5c-think-table.txt`.

| Measure | base (2) | E1b (2) | E4b (2) | E5c (1) |
|---|---|---|---|---|
| Overall | 20.46 | 22.80 | 21.49 | 21.14 |
| Non-live AST | 56.87 | 57.22 | 58.68 | 54.73 |
| Live AST | 55.48 | 51.74 | 53.03 | 52.11 |
| Live simple | 50.58 | 48.84 | 46.31 | 53.10 |
| Live multiple | 57.03 | 52.76 | 55.03 | 52.23 |
| Multi-turn | 5.44 | 7.19 | 5.38 | 5.12 |
| Memory | 12.47 | 12.79 | 8.71 | 8.17 |
| Irrelevance detection | 51.04 | 71.94 | 69.69 | 72.80 |

- Mean calls per `multi_turn_base` entry (sample 1 of each): base 12.4, E1b
  10.3, E4b 12.8, E5c 9.8.
- E5c makes the fewest multi-turn calls, but its multi-turn score is not
  higher than base. So "fewer mid-conversation calls" does not cause the E1b
  multi-turn gain. The causal part of H9 is not supported.
- E5c is not better than E4b (thinking 21.14 against 21.49, non-thinking
  19.72 against 20.29, one sample each).
- "Text instead of a call": 127 (base 156, E4b 97).

## F49. The E1b multi-turn gain is narrow and has no clear pattern

- Output: `v1/wsp/mt-e1b-gain.txt`. Thinking mode, two samples of base
  and E1b.
- Entries that E1b passes in both samples and base fails in both:
  `multi_turn_base` 1, `miss_func` 0, `miss_param` 3, `long_context` 4
  (8 of 800). The reverse: 1, 0, 1, 0 (2 of 800).
- On the 8 gained entries, E1b does not make fewer calls in a consistent
  way. For example `multi_turn_long_context_123`: base 6 and 7 calls, E1b
  11 and 8 calls.
- Conclusion: the robust part of the E1b multi-turn gain is about 6 entries
  net. It does not show one mechanism to copy. No more GPU time on it.

## Hypotheses

- H1 (rejected in the direct form): The loop direction that the wsp edit
  removes also carries a part of the "act now" signal. See F8. An indirect
  effect through the later layers remains possible.
- H2 (confirmed): The decision "call or answer" is mostly made at the first
  assistant token. See F6.
- H3 (confirmed): The relevance direction is mostly a real "tool fits the
  query" signal. See F9. About a quarter of the BFCL direction is not shared
  with the swap direction.
- H4 (supported): A direction in the residual stream encodes the query
  prior "I can answer this myself". It is mostly separate from the relevance
  direction. See F11.
- H5 (not confirmed, lower priority after D11): One or a few components in
  layers 12 to 16 write the prior. Test: direct attribution of each
  attention block and each MLP to the prior direction.
- H6 (partly supported, F17): In thinking mode, the action token reads the
  reasoning conclusion incompletely. 16.6% of the over-call rows stay above
  0.5 with an explicit "answer" conclusion.
- H7 (mostly supported, F17): Most irrelevance traces have no clear
  conclusion. Then the model falls back on a "tools are present, so call"
  default.
- H9 (F42, F44, F48): the read pattern part is supported (E1b pushes the
  mid-conversation steps toward "stop", E4b toward "call again"). The causal
  part ("fewer mid-conversation calls give the E1b multi-turn gain") is not
  supported: E5c makes fewer calls and does not gain.
- H8 (not confirmed): An edit that makes the post-think `<tool_call>` logit
  follow the relevance direction more strongly reduces the over-calls and
  keeps the correct calls. Test: a rank-1 edit in a late layer, measured on
  all BFCL categories in both modes (D10).
