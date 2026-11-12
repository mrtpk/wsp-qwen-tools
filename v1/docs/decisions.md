# Decisions

## D1. Use BFCL-V4 with `bfcl-eval==2026.3.23` in a separate venv

- Date: 2026-11-06.
- Decision: Install `bfcl-eval` in `v1/bfcl/.venv`. Do not add it to the
  project `uv.lock`.
- Reason: The project lock pins `vllm==0.30.0` for other work. BFCL is only a
  measurement tool for this study.
- Extra dependency: `soundfile`. `qwen_agent` imports it at startup.

## D2. Use the vLLM OpenAI chat API with the `qwen3_coder` parser

- Decision: Send BFCL requests to `vllm serve` through
  `/v1/chat/completions`. Use `--tool-call-parser qwen3_coder` and
  `--reasoning-parser qwen3`.
- Reason: BFCL 2026.3.23 has no Qwen3.5 handler. The Qwen3 local handler
  writes the old Hermes JSON tool format, but Qwen3.5 uses the XML format of
  `qwen3_coder`. The model card recommends the vLLM command with this parser.
  Codex would also use this API path.
- Implementation: `v1/bfcl/run_bfcl.py` registers new model entries. The
  handler is a subclass of `OpenAICompletionsHandler`.

## D3. Sampling settings

- Decision: Use `temperature=1.0`, `top_p=0.95`, `top_k=20`,
  `presence_penalty=1.5` in both thinking modes. Use `max_tokens=16384`.
- Reason: The model card note under the language table, which reports
  BFCL-V4, gives these values.
- Limit: There is one sample for each entry. The sampling noise on 2,501
  entries is about 1 percentage point (about 25 entries).

## D4. Do not run web search

- Decision: Run `single_turn`, `multi_turn`, and `memory`. Do not run
  `web_search`.
- Reason: Web search needs `SERPAPI_API_KEY`, and no key is set.
- Effect: The BFCL overall score counts web search as 0. Web search has 20% of
  the overall weight. Compare the parts of the score, not only the overall
  number.

## D5. vLLM on the RTX 5090

- Decision: Set `CUDA_HOME` to the pinned `nvcc` 13.0. Set
  `FLASHINFER_EXTRA_LDFLAGS` to `.cudart`. See the root `README.md`.
- Reason: Without these, the FlashInfer sampler JIT fails with "FlashInfer
  requires GPUs with sm75 or higher".
- Script: `v1/bfcl/serve.sh`.

## D6. The study goal

- Decision: Do not aim for a BFCL score now. First, learn how the model makes
  the tool-call decision in its weights.
- Reason: The user asked for this order on 2026-11-06.

## D7. The first target: "text instead of a tool call"

- Decision: Study the first assistant token in non-thinking mode. The model
  either starts `<tool_call>` or starts text.
- Reason: This failure is 47% of the single-turn failures of the base model.
  It is one binary decision at a known token position.
- Risk: A constant push to call tools can make irrelevance detection worse.
  Measure irrelevance with each change.

## D8. Study the query prior next

- Decision: Find the residual direction that sets the query prior ("I can
  answer this myself"). Find which layers write it, and how it reaches the
  `<tool_call>` readout.
- Reason: F10. The relevance gap is almost the same in the defect cases and
  the called cases. The prior is different.

## D10. Use the wsp edit as a negative reference

- Decision: Treat the wsp edit as an example of what an edit must not do.
  Use the wsp model as a negative control in each later comparison.
- Reason: F12. For tool calls, wsp is worse than base in both modes. Its
  small gains (irrelevance detection, memory in thinking mode) agree with
  a general fall in the call rate. F8 shows that its directions are
  orthogonal to the call decision, but the call rate still falls.
- Rules for our own edits:
  1. Measure all BFCL categories before and after the edit.
  2. Measure irrelevance detection with each change. A gain in calls with
     an equal loss in irrelevance detection is only a change of the call
     rate.
  3. Do not accept an edit because its direction is orthogonal to a
     signal. Test the behavior.
- Limit: This does not show that the wsp edit fails at its own goal (math
  loops in thinking mode). This study did not test MATH-500.

## D11. Focus on thinking mode

- Date: 2026-11-06 16:25. The user proposed this. I agreed.
- Decision: Study and edit for thinking mode. Use non-thinking mode only as a
  side-effect check after each edit.
- Reasons:
  1. Base in thinking mode is the best configuration (F12). Qwen reports
     BFCL-V4 in thinking mode.
  2. Thinking mode already removes most of the "text instead of a call"
     defect (156 cases against 685).
  3. Thinking loops do not occur in BFCL single-turn. The median output is 312
     tokens, p95 is 800, and none of 3,625 is near the 16,384 cap.
- Thinking-mode failures in the single-turn call categories (base, 1,056):
  `value_error:string` 365, text instead of a call 156, `value_error:others`
  120, parallel no match 94, wrong function name 76, parallel wrong count 73.
- Irrelevance detection is 50.88%. The model over-calls.
- Targets: (1) over-calling at the token after `</think>`; (2) wrong
  argument values. First check if (2) is a model error or BFCL strictness.
- Method change: the decision point is after a sampled reasoning trace.
  Rebuild "prompt + reasoning + `</think>`" from the saved traces.

## D12. Edit E1: a thresholded rank-1 edit of `L23.mlp.down_proj`

- Formula: W' = W − k · u · (v − (τ/μ) · m̂)ᵀ.
  - v: the read direction from F21 (irrelevance > need) at the action
    position.
  - m̂, μ: the mean-activation direction and its mean at the action position.
    The term (τ/μ) · m̂ acts as a threshold τ on v·a.
  - u: the residual direction that raises the `<tool_call>` logit and changes
    the logits of the 3,000 most common reasoning and query tokens least.
    It is a regularized least-squares solution (λ = 0.1 × mean diagonal).
- Reason for the site: `L23.mlp` is the last part before the final norm, so
  the effect on the logits is linear. Its read direction separates over
  from need with AUROC 0.931 (F21).
- Reason for the threshold: F21 (no bias in the weights).
- Reason for this u: the first u was the plain `<tool_call>` readout
  direction g ⊙ W_U[tc]. At k = 20, the background KL was 1.2 nats and the
  second parallel call fell from 1.0 to 0.29. That u changed many common
  tokens. With the least-squares u, the common-token |gain| is 0.031 on
  average (max 0.287), against 1.621 for `<tool_call>`.
- Calibration: `edit_e1.py sweep` on held-out (odd) rows. It measures the
  three groups, the second call of parallel calls (80 rows), and the KL at
  all positions of 30 background sequences.
- Limit: v, m̂, and τ come from the BFCL irrelevance data. The calibration rows
  are held out, but they come from the same distribution. BFCL is the real
  test.

## D13. Edit E2: E1 with the tool-call tokens protected

- Decision: Keep the E1 form. Add the 2,000 most common tokens of the
  rendered tool calls (XML structure, parameter names, values) to the set of
  tokens that u must not change.
- Implementation: `EDIT_VARIANT=e2 uv run python edit_e1.py ...`. The list is
  in `e2-common-ids.json`. E1 still builds from `e1-common-ids.json` with
  `EDIT_VARIANT=e1` (the default).
- New calibration metric: the KL at the positions inside the first tool call
  of 30 rows that need a call (`in_call_kl`).
- Reason: F23. `unexpected_param` and `value_error:string` rose with E1b.
- Grid: k = 2, 5, 8 with the midpoint threshold, and k = 5 with the need-mean
  threshold.
- When: after the E1 BFCL runs, because the GPU is full.

## D14. Do not run BFCL for E2; design E3 against the background push

- Data: at k = 5 with the midpoint threshold, E2 and E1b give almost the same
  calibration. E1b: need 0.951, refuse 0.006, over 0.450, background KL
  0.0096, in-call KL 0.0036. E2: 0.951, 0.006, 0.444, 0.0106, 0.0028.
- The E1b rerun reproduced its first calibration numbers exactly.
- Decision: do not spend a 60-minute BFCL run on E2. The difference would be
  within the noise.
- Reason for E3: the in-call KL is small for both. A more likely cause of the
  live AST cost is the push at ordinary positions. There, the E1 read value
  r = v·a − (τ/μ)·m̂·a has a mean of about +0.058 (v·a 0.012, plus 0.092 × 6.36
  / 12.81). The edit changes every reasoning token a little. At 0.01 nats
  per token over about 300 tokens, the sampled trace changes.
- E3 design: fit the read vector with a generalized discriminant
  v ∝ (Σ_action + γ·Σ_background)⁻¹ (μ_irrelevance − μ_need). Then set the
  threshold and center the background read at 0 with two bias proxies.

## D15. Run every shell command and script in tmux

- Date: 2026-11-07 01:57. The user asked for this.
- Decision: run all commands in session `wsp` with
  `v1/tmux_run.sh <name> <wait-seconds> <command>`. The helper writes
  the command to `v1/tmuxlogs/<name>.sh`, runs it in a new window, and
  writes the output to `v1/tmuxlogs/<name>.log` with an `__EXIT__`
  marker.
- First problem: the first version used `bash -c '<command>'`, and a command
  with single quotes gave an empty log. The fix writes the command to a
  script file.

## D16. Edit E4: fit the read vector on both modes

- Decision: fit v, the threshold, and the bias proxies on the decision tokens
  of thinking mode and non-thinking mode together.
- Reason: F31. E1b, fitted on thinking mode only, suppresses the calls in
  non-thinking mode.
- First step: collect the `mlp23` inputs at the non-thinking decision token
  for the 3,625 single-turn prompts.

## D17. A second BFCL sample in thinking mode

- Decision: run base, E1b, and E4b again in thinking mode with vLLM seed 1.
  Use new registry names (`qwen35-base-s2`, `qwen35-e1b-s2`,
  `qwen35-e4b-s2`) so that the results of the two samples stay separate.
- Reason: F35. The overall gap between E1b and E4b in thinking mode comes
  from memory and multi-turn, which are noisy with one sample.
- Analysis: report each sample and the mean of the two samples. Compare the
  sample-to-sample difference of base with the edit effects.
- Not run: a second non-thinking sample. The non-thinking E4b gain (+2.39)
  comes mostly from single-turn, which has less noise.

## D18. Edit E5: E4 with a constraint at the mid-conversation decision tokens

- Decision: add a fourth bias proxy, the mean `mlp23` activation at the
  later-step decision tokens of multi-turn traces. Add a constraint: the
  mean read there equals a target t_mid.
- t_mid = 0 makes the edit neutral in the middle of conversations. A small
  positive t_mid (E1b reads about +0.045 there) copies the E1b "stop" push.
- Data: the base thinking `multi_turn_base` traces, all 2,757 exact-match
  rows. Split by entry ID: even IDs for the fit, odd IDs for the check.
- Reason: F44. Both E1b and E4b change the mid-conversation decisions by
  accident.
- Risk: the multi-turn traces come from the BFCL evaluation data. The
  BFCL multi-turn score of E5 is then not a clean test. Use the held-out
  entry IDs, and report the multi-turn score with this note.

## D9. Context length for the remaining BFCL generation

- Decision: Restart the vLLM servers with `--max-model-len 262144`. Generate
  again the entries with the context error.
- Reason: The 65,536 limit caused 400 errors on long multi-turn histories.
  The model card serves the model with 262,144.
- Effect: The other entries do not change, because they did not reach the
  limit.
