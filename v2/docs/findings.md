# Findings (v2, run 1)

Run 1 follows the strict protocol (`TASK.md`). Every number below comes
from practice data, except where a line says "BFCL". The cell for each
number is in `cells/run1/`; the pass numbers refer to `work/passes.jsonl`.

## G1. No single MLP neuron writes the tool-call decision

- Cell `c01_readout.py` (weights only, no pass). For each neuron of layers 16
  to 23, the direct write of its down_proj column to the `<tool_call>`
  logit, as a z-score against the vocabulary spread.
- The largest |z| is 5.0 (layer 23). No neuron is a dedicated
  `<tool_call>` writer. This agrees with v1 F18 (the token mixers of layers
  15 to 22 write the decision).

## G2. Base behavior on practice data (passes 1 to 4)

| Set | Mode | Group | Call rate | A call is correct |
|---|---|---|---|---|
| single-turn `st_A` | think | cannot (no tool fits) / ask (a value is missing) / call | 0.54 / 0.88 / 0.96 | no / no / yes |
| single-turn `st_A` | nothink | cannot / ask / call | 0.54 / 0.69 / 0.81 | no / no / yes |
| multi-turn step `mt_A` | think | call step / text step | 0.88 / 0.35 | yes / no |
| multi-turn step `mt_A` | nothink | call step / text step | 0.74 / 0.23 | yes / no |

- The faults of v1 appear also on practice data: over-calls, invented
  values instead of a question, and missed calls in non-thinking mode.

## G3. Reads at the input of the layer-23 MLP (passes 5 to 8)

- The rebuilt decision positions reproduce the sampled behavior: AUROC of
  `p_call` against the sampled call 0.89 to 0.99.
- The model's own `p_call` separates "a call is correct" from "no call is
  correct" with AUROC 0.72 (single-turn) and 0.82 (multi-turn).
- r (LDA, even rows, all four contexts): held-out AUROC 0.97 to 0.98
  (single-turn) and 0.81 to 0.83 (multi-turn). In multi-turn, r is not
  better than the model.
- q (decision position against all other positions): held-out AUROC 1.0.
  `q·x` is 11.9 at decision positions in all four contexts (CV 0.04 to
  0.06), and at most 6 elsewhere.
- Context score e (multi-turn against single-turn decision positions):
  AUROC 1.0. e is 0 at single-turn and 6 at multi-turn decision positions
  (sd 0.3).
- Neuron 1488 of layer 23 has the lowest use (8.3e-3; median 3.5e-2).

## G4. The program: neuron 1488 of layer 23

- h = silu(G·x) · (U·x); the output is D[:, 1488] · h. Three vectors of 1,024
  values: 3,072 weights.
- Gate row: 2 · (ridge regression: +4 at decision positions, −8 elsewhere)
  − 2.33 · (c·x − τ · q·x / 11.9) + 0.5 · q.
  - The first term opens the gate only at decision positions.
  - The second term closes it at multi-turn decision positions.
  - The third term also opens it at "second chance" positions after a text
    preamble.
- Up row: (r − 1.47 · q / 11.9) / 1.74. At decision positions `q·x / 11.9`
  is about 1, so it is the bias proxy for the threshold θ = 1.47.
- Down column: 0.35 · u. u raises the `<tool_call>` logit and changes the
  4,000 most common practice tokens least (gain 1.65; common-token |gain|
  0.036 on average).
- Layer 23 is the last layer. Its MLP output changes only the logits at the
  same position. No later layer reads it.

## G5. Calibration (passes 9 to 15, held-out odd rows, teacher-forced)

- The gate works: mean |h| 6.4 at decision positions, 0.0000 elsewhere
  (largest 0.13).
- Removing neuron 1488 alone (K = 0): no change at decision positions; mean
  |Δ log p(`<tool_call>`)| 0.007 elsewhere.
- K = 1 saturates. Single-turn accuracy at 0.5: 0.76 → 0.92 (think), 0.71 →
  0.96 (nothink). Multi-turn: no gain. Larger K adds only leak (largest
  Δ 10.8 at K = 16).

## G6. Thinking mode: the reasoning sets the action

- Validation by generation (passes 16 to 19, first version, no δ term):
  single-turn think over-calls fell only from 0.59 to 0.47.
- Cause: after a suppressed decision token, 46 of 256 rows wrote a short
  preamble ("I'll …", "I can…") and then called. The reasoning had
  concluded "Let me call …" (v1 F17).
- At the position before such a late call, r still reads "no call" (r·x
  −1.19), and `q·x` is 7.5 (between 0.4 and 12). Pass 20.
- The δ term (+0.5 · q) opens the gate there. A larger δ gives little more
  veto and much more leak (cell `c29_delta2.py`).
- A separate veto neuron was rejected: its gate also opens before 76 to
  90% of the natural non-thinking preambles before correct calls (pass 27).
  Natural preamble calls are 12% (single-turn) and 23 to 26% (multi-turn)
  of the base non-thinking calls.
- Limit: neuron 1488 also pushes "no call" at natural single-turn
  non-thinking preambles (mean h −5.5). In the validation the non-thinking
  call rate still rose, because the decision push makes the model call at
  once.

## G7. Parallel calls are not affected

- Pass 22: 128 ToolACE rows with two or more calls (not in any fit). Before
  the second `<tool_call>`, h = 0.00 in both modes.

## G8. Held-out validation by generation (passes 23 to 26, final edit)

| Set | Mode | Decision accuracy, base → edited | Details |
|---|---|---|---|
| single-turn `st_B` | think | 0.641 → 0.785 | cannot 0.64 → 0.36, ask 0.84 → 0.56, call 0.97 → 0.98 |
| single-turn `st_B` | nothink | 0.652 → 0.930 | cannot 0.45 → 0.06, ask 0.69 → 0.12, call 0.81 → 0.94 |
| multi-turn `mt_B` | think | 0.715 → 0.703 | call 0.82 → 0.80, text step 0.39 → 0.39 |
| multi-turn `mt_B` | nothink | 0.746 → 0.750 | call 0.70 → 0.73, text step 0.21 → 0.23 |

- The base arm itself varies by about 0.05 between passes (vLLM batch
  nondeterminism). The multi-turn changes are within this noise.

## G9. Damage on generated tokens (passes 28, 29)

| Context | approx. KL mean | KL p99 | top-1 agreement |
|---|---|---|---|
| single-turn think | 0.0022 | 0.0024 | 0.996 |
| single-turn nothink | 0.0002 | 0.0027 | 0.997 |
| multi-turn think | 0.0001 | 0.0020 | 0.999 |

- KL on the original top-20 support of each position. The large single
  changes (largest |Δ log p| 41) are at decision positions, where the change
  is intended.

## G10. The deliverable

- `edits/run1-n1488` (openwsp format v1, `edit_sha256`
  `173071b6e0e66456cab5eccf3a630e32a9d33d49e3c104da0fd371fbd40df12e`).
- 3,072 changed weights in 3 tensors of layer 23: one row of `gate_proj`,
  one row of `up_proj`, one column of `down_proj`. For comparison: v1 E4b
  changes about 3.5M weights; the RikiyaT edit changes 131,072.
- Budget used: 29 of 50 passes, 43 of 300 code runs.

## G11. BFCL: the edit fails on single-call questions

BFCL-V4, `qwen35-wsp2`, seeds 0 and 1. Every value is the mean of two seeds.
v1 base and E4b come from v1 (two seeds each).

| Measure | base think | E4b think | wsp2 think | base nothink | E4b nothink | wsp2 nothink |
|---|---|---|---|---|---|---|
| Overall | 20.46 | 21.49 | 21.32 | 17.68 | 20.29 | 17.74 |
| Non-live simple | 39.96 | 41.21 | 30.46 | 30.41 | 38.45 | 8.21 |
| Non-live multiple | 74.25 | 70.75 | 62.75 | 47.50 | 59.50 | 10.25 |
| Non-live parallel | 58.75 | 67.00 | 53.75 | 22.75 | 47.00 | 40.50 |
| Live | 55.48 | 53.03 | 48.97 | 43.89 | 46.45 | 34.23 |
| Multi-turn | 5.44 | 5.38 | 5.94 | 3.44 | 3.31 | 3.50 |
| Memory | 12.47 | 8.71 | 10.43 | 5.48 | 5.16 | 8.71 |
| Relevance | 75.00 | 78.12 | 81.25 | 71.88 | 53.12 | 65.62 |
| Irrelevance | 51.04 | 69.69 | 76.05 | 77.72 | 86.66 | 88.36 |

- The overall think gain (+0.86) comes from irrelevance (+25). All AST
  categories fall. The non-thinking overall is unchanged for the same reason.
- No log has a context-length error.

- Non-thinking, seed 0: in `simple_python`, 325 of 400 rows fail with `decoder_failed` (v1 base:
  107). The model answers in text and solves the task itself. It makes no
  call.
- Overall is almost unchanged only because irrelevance, parallel and memory
  rise while single-call accuracy falls.

## G12. Cause: r reads the dataset, not the decision

- In `st_A` and `st_B`, every "no call" row is from When2Call and every
  "call" row is from ToolACE (`cells/run1/make_sets.py`). The label equals
  the source.
- r separates the two sources (AUROC 0.97). It does not separate "a call is
  correct" from "no call is correct". The model's own `p_call` gives 0.72.
- In multi-turn, both labels come from one source (APIGen-MT). There, r gives
  0.82, the same as the model. This agrees with the confound.
- BFCL single-call questions read as "When2Call-like" to r. The neuron then
  pushes "no call".
- G3 and the claim "the model knows the decision but does not act on it" are
  withdrawn. The practice validation (G8) has the same confound and does not
  show a real gain.
- Correction for a next run: fit and validate r only on rows where the call
  and no-call labels come from the same source and the same format.

## G13. Run 2: on clean data, the model's own p_call is the best read

- Set `cA` (`cells/run2/make_sets.py`): all rows from ToolACE. 128 call rows,
  128 swap rows (the same question with unrelated tools of another row),
  64 text rows (mostly a question for a missing value).
- Passes 30 to 32, cells `c36` and `c37`. AUROC for "a call is correct" at the
  decision position. The reads use 8-fold CV; a call row and its swap row
  are in the same fold.

| Score | think | nothink |
|---|---|---|
| model `p(<tool_call>)` | **0.937** | **0.860** |
| run-1 r (layer-23 MLP input) | 0.837 | 0.773 |
| CV read, residual before layer 8 | 0.694 | 0.590 |
| CV read, before layer 12 | 0.823 | 0.768 |
| CV read, before layer 16 | 0.879 | 0.831 |
| CV read, before layer 20 | 0.777 | 0.829 |
| CV read, layer-23 MLP input | 0.824 | 0.801 |

- No linear read beats the model. The model does not hold a better decision
  than it shows. G12 is confirmed: run-1 r is worse than the model here.
- Mean `p_call` at the decision position (think / nothink): call 0.94 /
  0.69, swap 0.24 / 0.27, text 0.39 / 0.31.
- Sampled call rate in thinking mode (pass 30): call 0.98, swap 0.43, text
  0.52. This is higher than `p_call` for swap and text, because some calls
  come later, after a text preamble (G6).
- The best constant shift of the `<tool_call>` logit at the decision
  position (cell `c38`): expected decision accuracy 0.801 → 0.835 (think,
  b = −2.0) and 0.706 → 0.719 (nothink, b = −0.75). Both shifts lower the
  call rate on call rows.
- Conclusion: an edit of the call decision at the decision token has little
  room. The decision-neuron route stops here (plan step 2, "if no").
- **Corrected by G16.** The reads above used λ = 0.01 (too little ridge).
  With λ = 1 to 30 the reads beat the model. The conclusion above is
  withdrawn.

## G14. Codex: the model is not usable (all three versions)

Setup: V2-D8. 16 small coding tasks. Pass = the check script passes.

| Model | Mode | Runs | Pass | Shell commands per run | Failed commands | Rejected tool calls per run |
|---|---|---|---|---|---|---|
| base | think | 48 | 6 (12%) | 6.5 | 44% | 40.1 |
| E4b | think | 48 | 6 (12%) | 5.2 | 45% | 10.6 |
| wsp2 | think | 48 | 5 (10%) | 5.8 | 30% | 36.1 |
| base | nothink | 26 (stopped) | 4 (15%) | 4.8 | 30% | 217.6 |
| E4b | nothink | 48 | 2 (4%) | 5.5 | 43% | 23.0 |
| wsp2 | nothink | 30 (stopped) | 3 (10%) | 4.8 | 38% | 214.8 |

- The non-thinking run was stopped at the user's request (focus on BFCL).
- The main fault is tool use, not code. The model calls `write_stdin` on a
  session that does not exist (base, think: 1,126 times in 48 runs). It
  also calls tools that do not exist (`run_command`, `execute_command`,
  `read_file`) and Codex tools it does not need (goals, multi-agent).
- E4b cuts the rejected tool calls by about 4×, but it does not pass more
  tasks.
- Codex sends 9 tools (`exec_command`, `write_stdin`, `request_user_input`,
  `view_image`, `multi_agent_v1`, `get_goal`, `create_goal`, `update_goal`,
  `web_search`), about 17.8k characters of definitions. A turn has 30k to
  70k input tokens.

## Open items

- Codex path (G14). The model does not work in Codex. The user will decide
  an approach later. Run 2 (decision edit) is not the answer: the decision
  is not the main Codex fault.

## G15. BFCL single-turn: a harness retry does not help (base)

- `eval/run_bfcl_retry.py`: after the first answer, the handler checks each
  call with harness knowledge only (no answer key): unreadable call markup,
  unknown tool, not a JSON object, missing required parameter, wrong JSON
  type, value outside the enum. On an error it returns tool-error messages
  and queries once more. A text answer gets no retry. Base model, seed 0,
  `single_turn`.
- Retries: 226 of 3,641 entries (think), 126 (nothink). After the retry:
  think 102 valid calls, 38 no call, 86 still invalid; nothink 39, 62, 25.
- Scores (retry against the base mean of two seeds): all AST categories
  change within the seed spread. Relevance 75.0 → 87.5 (think; 16 entries).
- The remaining failures (think): `value_error:string` 388, `decoder_failed`
  152, `value_error:others` 113. Nothink: `decoder_failed` 744,
  `value_error:string` 346.
- Samples of `value_error:string`: invented values for optional parameters
  ("any", "regulare"), wrong values ("Rohnert Park, TX" for "CA"), wrong
  conventions ("London" for "London, UK").
- Conclusion: format errors are a small share. The failures are about
  values and about no call. A harness retry cannot fix them.

## G16. With enough ridge, the reads beat the model (corrects G13)

- Cells `c43` to `c45`: the G13 test with λ from 0.01 to 10⁴. 8-fold CV.
  Folds by question (single-turn) and by conversation (multi-turn).
- Multi-turn set: `mt_A` (APIGen-MT only, 128 call steps, 128 text steps),
  non-thinking, pass 34.

| Decision | Model `p_call` | Read λ = 0.01 | Read λ = 3, layer-23 MLP input | Read λ = 3, before layer 16 |
|---|---|---|---|---|
| Single-turn think (`cA`) | 0.937 | 0.824 | 0.951 | 0.955 |
| Single-turn nothink (`cA`) | 0.860 | 0.801 | **0.934** | 0.929 |
| Multi-turn nothink (`mt_A`) | 0.818 | 0.789 | **0.905** | 0.909 |

- The AUROC is flat from λ = 1 to 30. Even λ = 10⁴ (almost a plain
  difference of means) beats the model in nothink single-turn (0.924) and
  multi-turn (0.869). The gain does not come from the choice of λ.
- In non-thinking mode the model holds a better call decision than it
  shows, in single-turn and in multi-turn. In thinking mode the gap is
  small.
- Not yet tested on held-out rows (`cB`, `mt_B`).

## G17. Multi-turn: the call decision is the main per-step error

- Pass 33, cell `c40`: 93 gold APIGen-MT call steps (not in `mt_A`, `mt_B`),
  non-thinking, teacher-forced. Mean 15.3 tools, mean step 4.5. Rows with a
  reference longer than 500 characters are left out (cut in `convert.py`).

| Span | Top-1 right |
|---|---|
| Call decision (`<tool_call>`) | 0.70 |
| Function name | 0.93 |
| Parameter names | 1.00 |
| Parameter values | 0.95 |
| Whole call (greedy) | 0.60 |

- At 0.6 per step, a task with five calls succeeds about 8% of the time.
  The per-step errors compound.

## G18. Held-out: the reads hold inside a source, not across sources

- Passes 35 (`cB`) and 36 (`mt_B`), non-thinking, decision position. Cells
  `c47`, `c48`. Read: LDA, λ = 3, layer-23 MLP input, fitted on A.

| Read fitted on | `cB` (ToolACE single-turn) | `mt_B` (APIGen-MT multi-turn) |
|---|---|---|
| model `p_call` | 0.839 | 0.845 |
| `cA` | **0.947** | 0.675 |
| `mt_A` | 0.505 | **0.933** |
| joint (`cA` + `mt_A`, each source centred) | 0.943 | 0.910 |

- On `cB`: swap rows 0.971 (model 0.848), text rows 0.899 (model 0.822).
- G16 holds on held-out rows of the same source.
- A read from one source does not work on the other source. The joint read
  works on both (pooled, one threshold: 0.929) only because it saw both.
- Risk: BFCL is a third distribution. Run 1 failed in this way (G12). A
  test on a third practice source must come before an edit.

## G19. Third source (glaive): the single-turn read transfers

- Practice source: `glaiveai/glaive-function-calling-v2`, revision
  `e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac`, apache-2.0
  (`practice/convert_glaive.py`). First user turn and first assistant turn.
  A text answer directly before a call is dropped (label not clear). Leak
  filter: 1 row dropped. 70,528 rows; questions deduplicated for the set.
- Set `gT` (`cells/run2/make_glaive_set.py`): 128 call, 128 swap, 64 text.
  Test only. Pass 37, cell `c50`. Non-thinking.

| Score | All | call/swap | call/text |
|---|---|---|---|
| model `p_call` | 0.934 | 0.920 | 0.963 |
| read fitted on `cA` (ToolACE) | 0.958 | 0.955 | 0.965 |
| joint read (`cA` + `mt_A`) | 0.955 | 0.948 | 0.969 |
| read fitted on `mt_A` | 0.678 | 0.581 | 0.871 |

- The single-turn read works on a source that it never saw. The gain over
  the model is small (+0.02); glaive rows have only 1 or 2 tools.
- Mean `p_call` (nothink): call 0.64, swap 0.21, text 0.09. The model
  under-calls on call rows.

## G20. Edit v3 (neuron 1488, clean joint read)

- Cells `c51` to `c54`. Gate row 2 G + 0.5 q (no multi-turn "off" term). Up
  row (r₂ − θ q / 11.9), r₂ = joint read (`cA` + `mt_A`, nothink, λ = 3,
  scaled to sd 1), θ = 0.75. Down column 0.5 u. 3,072 weights.
- Calibration (pass 38, edited probe on A): the predicted neuron activation
  matches the measured one (correlation 1.0). Log-odds gain per unit of
  K·h: median 4.6 (IQR 3.8 to 5.0). The effect saturates from K = 0.5.
- Validation by generation (passes 39 to 43, both arms, seed 0, held-out).
  `cB` and `gT`: 96 call, 96 swap, 64 text rows (512-sample limit).

| Set | Mode | Decision accuracy, base → v3 | Call rate: call / swap / text (or call step / text step) |
|---|---|---|---|
| `cB` | nothink | 0.707 → 0.805 | 0.77 / 0.30 / 0.38 → 0.82 / 0.16 / 0.28 |
| `gT` (glaive, never fitted) | nothink | 0.738 → 0.871 | 0.69 / 0.30 / 0.12 → 0.88 / 0.18 / 0.06 |
| `mt_B` | nothink | 0.734 → 0.809 | 0.70 / 0.23 → 0.84 / 0.23 |
| `cB` | think | 0.730 → 0.754 | 0.97 / 0.36 / 0.48 → 1.00 / 0.32 / 0.50 |
| `mt_B` | think | 0.746 → 0.676 | 0.86 / 0.37 → 0.88 / 0.53 |

- Function-name accuracy of the calls does not change (0.86 to 1.00).
- Non-thinking improves on all three sources. Thinking multi-turn gets
  worse: more calls on text steps. The read was fitted on non-thinking
  states.
- Two failed requests (not counted): 640 samples > 512; tool-call arguments
  as a dict instead of a JSON string.

## G21. Edit v4: v3 with a thinking-mode "off" term

- Cells `c58` to `c62`. Mode read m: LDA (λ = 3), thinking against
  non-thinking decision states (`cA` passes 31, 32; `mt_B` passes 44, 36).
  CV AUROC 1.0.
- Gate row: 2 G + 0.5 q − 24 (m − τ q / 11.9), τ = −0.008. K = 0.1825: the
  larger non-thinking gate is compensated, so the non-thinking effect equals
  v3 (mean |K h| 5.9 against 5.8).
- Gate open (pre-activation > 0) at decision positions: thinking 1 to 2%,
  non-thinking 100%. At background positions: 0.2 to 0.7%; mean |K h| 0.001
  to 0.006, largest 2.9 (v3: 0.7).
- Validation by generation (passes 45 to 47, both arms, seed 0, held-out):

| Set | Mode | Decision accuracy, base → v4 | Call rate |
|---|---|---|---|
| `cB` | nothink | 0.684 → 0.809 | call 0.74 → 0.83, swap 0.32 → 0.18, text 0.39 → 0.25 |
| `mt_B` | nothink | 0.727 → 0.809 | call step 0.69 → 0.84, text step 0.23 → 0.22 |
| `mt_B` | think | 0.750 → 0.723 | call step 0.88 → 0.85, text step 0.38 → 0.41 |

- No `<tool_call>` markup left in text. Fewer calls after a preamble
  (`mt_B` nothink 31 → 17).
- The thinking change (−0.03) is inside the pass-to-pass noise (G8: about
  0.05). v3 was −0.07.

## G22. BFCL: edit v4 improves both modes (no BFCL data in the fit)

BFCL-V4, `qwen35-wsp4`, seeds 0 and 1. v1 base and E4b from v1 (two seeds).
One `memory_vector` entry of seed 1 nothink was missing after the run; it
was generated again alone before the scoring. No context-length errors.

| Measure | base nothink | E4b nothink | **v4 nothink** | base think | E4b think | **v4 think** |
|---|---|---|---|---|---|---|
| Overall | 17.68 | 20.29 | **19.81** | 20.46 | 21.49 | **21.02** |
| – seeds | 17.63 / 17.74 | 20.02 / 20.56 | 19.32 / 20.31 | 20.04 / 20.88 | 21.40 / 21.59 | 20.93 / 21.10 |
| Non-live AST | 33.98 | 49.55 | 39.09 | 56.87 | 58.68 | 54.99 |
| Live | 43.89 | 46.45 | 44.45 | 55.48 | 53.03 | 53.85 |
| Multi-turn | 3.44 | 3.31 | 4.38 | 5.44 | 5.38 | 6.44 |
| Memory | 5.48 | 5.16 | 7.42 | 12.47 | 8.71 | 15.16 |
| Relevance (16 entries) | 71.88 | 53.12 | 53.12 | 75.00 | 78.12 | 84.38 |
| Irrelevance | 77.72 | 86.66 | 86.66 | 51.04 | 69.69 | 51.67 |

- Non-thinking: +2.13. Both v4 seeds are above both base seeds. This is the
  first BFCL gain of v2 that used no BFCL data in the fit.
- Thinking: +0.56. Irrelevance does not change (the neuron is off at most
  thinking decisions). Memory, multi-turn and relevance rise; non-live AST
  falls. So the neuron is partly active in thinking mode on BFCL contexts:
  the mode read (G21) does not fully hold there. Not yet checked.
- v4 is 0.5 (nothink) and 0.5 (think) below E4b. E4b was fitted on BFCL
  data and changes about 3.5M weights; v4 changes 3,070.

## G23. Run 3: thinking mode has a read gap too; edit v5 failed

Run 3 (V2-D10): work directory `work3`, cells `cells/run3/`, fresh sets
(`cells/run3/make_sets.py`: `tA`, `tB` ToolACE; `mA`, `mB` APIGen-MT; `gB`
glaive). The store starts from edit v4.

- Thinking decision (passes 3, 4; CV, λ = 3): `tA` model 0.916, read 0.939
  to 0.941; `mA` model 0.806, read 0.901 to 0.910. Mean `p_call` on `tA`:
  call 0.93, swap 0.31, text 0.40 (over-calls).
- v4 in thinking mode on these sets: gate open at 1.6 to 4.3% of decisions.
- Edit v5 = v4 + neuron T (slot 1152, gate 2 G + 0.5 q + 24 (m − τ q /
  11.9), up r_T − θ q / 11.9). Generation (passes 6 to 9): thinking `tB`
  0.695 → 0.527, `mB` 0.801 → 0.641, `gB` 0.793 → 0.648; function-name
  accuracy `tB` 0.92 → 0.48.
- Cause (pass 10, cell `c10`): the T gate opens at 72 to 91% of positions
  that are not decisions (inside tool calls, in the text), in both modes.
  The mode read m is high on all thinking-mode text, not only at decisions.
  An additive gate "decision + mode" cannot do an AND here. v3 and v4 do
  not have this fault (inside calls: 0%; other positions: at most 1%).

## G24. Edit v6: a gate direction for thinking decisions only

- Cell `c12`: G_T = ridge regression without intercept at the layer-23 MLP
  input. Target +4 at thinking decisions, −8 at nothink decisions, inside
  tool calls, other positions and background positions (both modes). Each
  group has equal weight. Data: run-3 passes 3, 4, 10 (even rows); run-2
  passes 31, 32, 34, 35, 36 (copied to `work3/imports`).
- Held-out (pass 10, odd rows): thinking decisions min 2.58; every other
  kind max −3.38.
- Neuron T6 (slot 1152): gate 2 G_T, up r_T − 1.0 q / 11.9 (r_T: joint
  thinking read on `tA` + `mA`, λ = 3, sd 1), down 1.0 u. θ and K: the best
  expected accuracy on A with call-row `p_call` falling by at most 0.02.
- T6 on pass 10: open at 100% of thinking decisions, 0% at every other
  position kind (both modes).
- Edit v6 = v4 + T6: 6,144 weights (two neurons of layer 23).
- Validation by generation (passes 11 to 14, both arms, seed 0, held-out):

| Set | Mode | Decision accuracy, base → v6 | Call rate |
|---|---|---|---|
| `tB` | think | 0.723 → 0.805 | call 0.98 → 0.92, swap 0.45 → 0.25, text 0.41 → 0.28 |
| `mB` | think | 0.781 → 0.805 | call step 0.88 → 0.92, text step 0.32 → 0.31 |
| `gB` (glaive) | think | 0.801 → 0.922 | call 1.00 → 1.00, swap 0.38 → 0.12, text 0.23 → 0.12 |
| `tB` | nothink | 0.691 → 0.824 | call 0.76 → 0.81, swap 0.39 → 0.15, text 0.30 → 0.20 |

- Function-name accuracy does not change (0.83 to 1.00). No stray markup.
- Thinking `tB`: calls after a text preamble rise from 6 to 20 (G6). The
  call rate on call rows falls by 0.06.

## G25. BFCL: edit v6 gives the best thinking score; non-live falls

- BFCL-V4, `single_turn`, `multi_turn`, `memory`, two seeds (0, 1). Means.
  Overall uses the leaderboard formula (web search counts 0).

| Mode | Metric | Base | E4b (v1) | v4 | v6 |
|---|---|---|---|---|---|
| think | Overall | 20.46 | 21.49 | 21.02 | **22.05** |
| think | Non-live AST | 56.87 | 58.68 | 54.99 | 53.11 |
| think | Live | 55.48 | 53.03 | 53.85 | 53.84 |
| think | Irrelevance | 51.04 | 69.69 | 51.67 | 72.12 |
| think | Multi-turn | 5.44 | 5.38 | 6.44 | 6.19 |
| think | Memory | 12.47 | 8.71 | 15.16 | 11.39 |
| nothink | Overall | 17.68 | 20.29 | 19.81 | 19.57 |
| nothink | Non-live AST | 33.98 | 49.55 | 39.09 | 39.66 |
| nothink | Irrelevance | 77.72 | 86.66 | 86.66 | 87.24 |
| nothink | Memory | 5.48 | 5.16 | 7.42 | 5.48 |

- Seeds, think overall: v6 21.88 / 22.21. Base 20.04 / 20.88.
- Nothink: T6 is off (G24). The v4 and v6 difference is seed noise. Memory
  varies by up to 4.3 points between seeds of one model.
- Thinking non-live: `ast_decoder` failures rise (base, v4, v6, two seeds
  summed): `simple_python` 28, 37, 92; `parallel` 33, 37, 75. The v6 output
  is a text answer, for example a worked math solution.
- Cause (corrected in G32): the `tA` "text" group is a "no call" target in
  the T read. Those rows are missing-parameter or no-function cases. T6
  learned to refuse, and the refusal spread to self-answerable tasks.

## G26. Edit v7: the T read without the "text" group

- Cell `c15`: r_T7 = LDA (λ = 3) on `tA` call vs `tA` swap, plus `mA` call
  vs `mA` text. The `tA` text group is not in the fit (G25). cos(r_T6,
  r_T7) = 0.903. Gate as T6 (2 G_T), slot 1152.
- θ and K: the best accuracy on `tA` (call, swap) and `mA`. Limits: mean
  `p_call` on `tA` call, `mA` call and `tA` text falls by at most 0.02.
  Result: θ = 0.875, K = 1.0 (K is at the grid edge).
- Edit v7 = v4 + T7: 6,144 weights.
- Validation by generation (passes 15 to 18, both arms, seed 0, held-out):

| Set | Mode | Decision accuracy, base → v7 | Call rate |
|---|---|---|---|
| `tB` | think | 0.738 → 0.785 | call 0.96 → 0.95, swap 0.38 → 0.23, text 0.42 → 0.44 |
| `mB` | think | 0.797 → 0.812 | call step 0.89 → 0.96, text step 0.30 → 0.34 |
| `gB` (glaive) | think | 0.797 → 0.809 | call 1.00 → 1.00, swap 0.38 → 0.19, text 0.25 → 0.48 |
| `tB` | nothink | 0.691 → 0.832 | call 0.83, swap 0.14, text 0.22 (v4 only) |

- v7 keeps calls on call rows and on text rows. It removes about half of
  the swap calls. Decision accuracy is lower than v6 because the text rows
  count as "no call" in this measure. This is the intended change.
- Calls after a text preamble: `tB` think 7 → 16. Name accuracy: no change.

## G27. BFCL: edit v7 is the best thinking result

- BFCL-V4, two seeds, means. Same setup as G25.

| Mode | Metric | Base | E4b (v1) | v4 | v6 | v7 |
|---|---|---|---|---|---|---|
| think | Overall | 20.46 | 21.49 | 21.02 | 22.05 | **22.54** |
| think | Non-live AST | 56.87 | 58.68 | 54.99 | 53.11 | 55.78 |
| think | Live | 55.48 | 53.03 | 53.85 | 53.84 | 54.19 |
| think | Irrelevance | 51.04 | 69.69 | 51.67 | 72.12 | 67.22 |
| think | Multi-turn | 5.44 | 5.38 | 6.44 | 6.19 | 7.88 |
| think | Memory | 12.47 | 8.71 | 15.16 | 11.39 | 12.26 |
| nothink | Overall | 17.68 | 20.29 | 19.81 | 19.57 | 19.55 |

- Seeds, think overall: v7 22.65 / 22.42. Both seeds are above every seed
  of base, E4b, v4 and v6.
- v7 recovers most of the v6 non-live loss (53.11 → 55.78) and keeps most
  of the irrelevance gain (51.04 → 67.22).
- Thinking multi-turn rises in all four subcategories (base → v7): base
  10.0 → 12.5, miss func 0.75 → 1.75, miss param 5.0 → 7.25, long context
  6.0 → 10.0.
- Nothink: T7 is off. v4, v6 and v7 are equal within seed noise.

## G28. Edit v8: neuron N for nothink decisions (V2-D11)

- Gate G_N (cells `c18`, `c20`, `c21`): ridge without intercept, as G_T, but
  +4 at nothink decisions. Held-out (pass 10, odd rows): nothink decisions
  min 2.94; every other kind max −3.45. Open at 100% of nothink decisions on
  every probe set.
- Up read, first try (`c18`): LDA on `tA` (call + text) vs swap. It does not
  split multi-turn call steps from text steps. Only K = 0.1 meets the
  limits: no effect.
- Second try (`c20`): joint LDA on `tA` + `mA`. In-sample `tA` text
  `p_call` 0.16 → 0.67. Generation (passes 21 to 23, held-out) fails:
  swap calls `gB` 0.20 → 0.56, `tB` 0.36 (v4 alone: 0.14). The read
  overfits (96 swap rows).
- Third try (`c21`, kept): joint LDA on `tA`, `cA` (run-2 pass 32), `mA`,
  `mtA` (run-2 pass 34). Limits: swap `p_call` rises by at most 0.03, call
  falls by at most 0.02, multi-turn text rises by at most 0.03. θ = −0.75,
  K = 0.75, slot 2088. Held-out probe `cB`: call 0.78 → 0.87, text 0.18 →
  0.28, swap 0.055 → 0.073. `mt_B`: no change.
- Edit v8 = v7 + N: 9,216 weights (three neurons of layer 23).
- Generation, nothink, held-out (passes 24 to 26, base → v8):

| Set | Call | Swap | Text | Note |
|---|---|---|---|---|
| `tB` | 0.77 → 0.89 | 0.39 → 0.21 | 0.33 → 0.34 | v4 alone (pass 17): 0.83, 0.14, 0.22 |
| `mB` | 0.77 → 0.85 | − | 0.23 → 0.20 | |
| `gB` | 0.76 → 0.95 | 0.20 → 0.34 | 0.08 → 0.27 | |

- Against v4 alone, N adds calls on call and text rows. It also adds calls
  on swap rows.

## G29. BFCL: edit v8 nothink trades irrelevance for call accuracy

- BFCL-V4, nothink, two seeds, means (v8 = v7 in thinking mode).

| Metric | Base | E4b (v1) | v4 | v7 | v8 |
|---|---|---|---|---|---|
| Overall | 17.68 | 20.29 | 19.81 | 19.55 | 19.76 |
| Non-live AST | 33.98 | 49.55 | 39.09 | 39.11 | 47.03 |
| Live | 43.89 | 46.45 | 44.45 | 44.23 | **48.00** |
| Irrelevance | 77.72 | 86.66 | 86.66 | 87.11 | 78.93 |
| Relevance | 71.88 | 53.12 | 53.12 | 43.75 | 68.75 |
| Multi-turn | 3.44 | 3.31 | 4.38 | 4.31 | 3.56 |
| Memory | 5.48 | 5.16 | 7.42 | 6.02 | 6.46 |

- Neuron N gives +7.9 non-live and +3.8 live. Live is the best of all models.
  Irrelevance falls by 8.2 (back to the base level). The overall score does
  not change.
- The seed-1 job hung once (progress log, 04:30). The resumed run has all
  5,017 entries.

## G30. Edit v9: neuron T8 (V2-D11 targets in thinking mode)

- Cell `c23`: T8 replaces T7 in slot 1152. Gate 2 G_T. Up read: joint LDA
  (λ = 3) on thinking decisions of `tA` (pass 3), `cA` (run-2 pass 31), `mA`
  (pass 4). Call: call and text (single-turn), call (multi-turn). No call:
  swap, multi-turn text. Limits as `c21`. θ = −0.125, K = 2.0 (grid edge).
- In-sample (base → T7 → T8): `tA` text 0.40 → 0.38 → 0.65, swap 0.31 →
  0.18 → 0.14; `mA` text 0.30 → 0.24 → 0.19; `cA` text 0.40 → 0.42 → 0.69.
- Generation, thinking, held-out (passes 27 to 29, base → v9; v7 from
  passes 15, 16, 18):

| Set | Call | Swap | Text |
|---|---|---|---|
| `tB` | 0.94 → 0.92 (v7 0.95) | 0.46 → 0.23 (v7 0.23) | 0.45 → 0.52 (v7 0.44) |
| `mB` | 0.92 → 0.91 (v7 0.96) | − | 0.41 → 0.23 (v7 0.34) |
| `gB` | 0.98 → 1.00 (v7 1.00) | 0.35 → 0.33 (v7 0.19) | 0.25 → 0.83 (v7 0.48) |

- Mixed against v7. Better multi-turn stops and more text-row calls. Worse
  swap on glaive.

## G31. BFCL: edit v9 loses to v7 in thinking mode

- BFCL-V4, thinking, two seeds, means (v9 = v8 in nothink).

| Metric | Base | E4b (v1) | v7 | v9 |
|---|---|---|---|---|
| Overall | 20.46 | 21.49 | **22.54** | 20.96 |
| Non-live AST | 56.87 | 58.68 | 55.78 | **59.46** |
| Live | 55.48 | 53.03 | 54.19 | 55.59 |
| Irrelevance | 51.04 | 69.69 | 67.22 | 49.58 |
| Relevance | 75.00 | 78.12 | 78.12 | 90.62 |
| Multi-turn | 5.44 | 5.38 | 7.88 | 5.81 |
| Memory | 12.47 | 8.71 | 12.26 | 13.77 |

- T8 makes the best single-turn caller of all models, but it loses the
  irrelevance gain and the multi-turn gain of T7.
- In both modes, the text-row push (V2-D11) raises AST and lowers
  irrelevance by about the same overall weight. Multi-turn (weight 30) sets
  the ranking. T7 has the higher call rate on multi-turn call steps (`mB`
  0.96 against 0.91).
- Decision: keep T7 for thinking mode.

## G32. Multi-turn: a call where the reference asks is a main failure

- v7, thinking, two seeds: 1,474 failed multi-turn entries. The first
  differing turn (model calls against reference calls):
  more calls 701, call where the reference has none 328, fewer calls 160,
  other functions 122, no decoded result 95, no call 53.
- All 328 "call where the reference has none" cases are in `miss_func` (192)
  and `miss_param` (156). The model must ask or refuse there.
- The practice text rows are this same case. Examples (`tA`): "the required
  'period' parameter is missing", "there are no available functions that
  can ...". G25 called them "answer in text when you can". That was wrong.
- So the V2-D11 push (text → call) raises single-turn AST and lowers
  `miss_param` (v7 7.25 → v9 4.75; nothink v7 6.25 → v8 4.25). The
  single-turn gain and the multi-turn loss come from one direction.
- "More calls than the reference" (701) counts extra exploration calls (for
  example `ls`, `cd`). BFCL checks state and results, so many of these are
  not errors. This measure does not separate them.

## G33. Three seeds: v7 thinking gain holds

- BFCL-V4, thinking, seeds 0, 1, 2 (E4b: seeds 0, 1 from v1).

| Metric | Base | v7 | E4b (v1) |
|---|---|---|---|
| Overall | 20.04 / 20.88 / 21.59 = 20.84 | 22.65 / 22.42 / 22.56 = **22.54** | 21.40 / 21.59 = 21.49 |
| Non-live AST | 57.50 | 55.21 | 58.68 |
| Live | 54.95 | 54.16 | 53.03 |
| Multi-turn | 5.54 | 7.38 | 5.38 |
| Memory | 13.90 | 13.62 | 8.71 |
| Irrelevance | 51.52 | 66.70 | 69.69 |

- The lowest v7 seed (22.42) is above the highest base seed (21.59). Gain:
  +1.70 overall. The gain comes from irrelevance (+15.2) and multi-turn
  (+1.8). v7 loses 2.3 non-live AST.
- v7 spread is 0.23; base spread is 1.55.

## Run 3 summary (2026-11-08)

- Final edit: v7 = v4 (neuron 1488) + T7 (neuron 1152), layer 23 only, 6,144
  weights. `edits/run3-v7`, `models/qwen35-wsp7`, `edit_sha256` 6b456447….
  No BFCL data in any fit (run 3 is "not blind" for choices, V2-D7, V2-D10).
- BFCL-V4 overall: thinking 20.84 → 22.54 (three seeds); nothink 17.68 →
  19.55 (two seeds). v1 E4b (fitted on BFCL data): 21.49 and 20.29.
- Tried and not kept: v5 (gate opened off-decision, G23), v6 (refusal spread
  to self-answer tasks, G25), v8 and v9 (text → call push, wrong premise,
  G29 to G32), v10 ("ask" rows, refusal spread to matched tools, G35).
- Limit: a decision neuron sets "call or no call". Most remaining BFCL
  failures are multi-turn state errors and memory errors. A decision neuron
  cannot fix them. 60 is out of reach with this method.
- Budget run 3: 35 of 100 passes (V2-D13), 29 of 300 code runs.

## G34. Edit v10: neuron T11 with when2call "ask" rows

- Sets `wA` (fit), `wB` (held-out): 256 when2call "ask" rows each (a required
  value is missing). Base thinking: `wA` mean `p_call` 0.79; `wB` call rate
  0.84. The base invents the missing value.
- Cell `c27`: r_T11 = LDA (λ = 3) on tA call vs swap, mA call vs text, plus
  `wA` as no call. Limits: call falls by at most 0.02 (tA, mA), tA text falls
  by at most 0.10. θ = 1.25, K = 0.75. cos(r_T7, r_T11) = 0.675. Edit v10 =
  v4 + T11 (slot 1152); slot 2088 (N) back to the original weights.
- In-sample: `wA` 0.79 → 0.07; tA call 0.93 → 0.97, swap 0.31 → 0.18, text
  0.40 → 0.30; mA call 0.83 → 0.91, text 0.30 → 0.31.
- Generation, thinking, held-out (passes 32 to 35, base → v10):

| Set | Accuracy | Call rate |
|---|---|---|
| `tB` | 0.684 → 0.820 (v7 0.785) | call 0.98 → 0.96, swap 0.50 → 0.26, text 0.48 → 0.27 |
| `mB` | 0.797 → 0.797 | call 0.93 → 0.96, text 0.34 → 0.37 |
| `gB` | 0.789 → 0.875 | call 1.00 → 0.99, swap 0.36 → 0.16, text 0.30 → 0.25 |
| `wB` | 0.160 → 0.242 | ask 0.84 → 0.76; 176 calls after a text preamble (base 2) |

- On `wB`, the decision neuron moves the first token away from
  `<tool_call>`. The model then writes "I'll fetch ..." and calls. The call
  is planned in the reasoning. One neuron at the decision position cannot
  stop it.

## G35. Edit v10 on BFCL-V4 (thinking, two seeds): not kept

- BFCL-V4 thinking, seeds 0 and 1 (means; base and E4b from v1):

| Metric | base | E4b | v7 | v10 |
|---|---|---|---|---|
| Overall | 20.46 | 21.49 | 22.54 | 21.99 |
| Non-live AST | 56.87 | 58.68 | 55.78 | 51.88 |
| Live | 55.48 | 53.03 | 54.19 | 52.92 |
| Multi-turn | 5.44 | 5.38 | 7.88 | 6.12 |
| Memory | 12.47 | 8.71 | 12.26 | 11.40 |
| Irrelevance | 51.04 | 69.69 | 67.22 | 73.90 |
| MT miss param | 5.00 | 5.00 | 7.25 | 6.25 |

- Both v10 seeds (22.26, 21.72) are below the lowest v7 seed (22.42).
- Irrelevance rises by 6.7 points over v7. Non-live AST falls by 3.9 points.
- `ast_decoder` errors (text answer, no call), two seeds, base / v7 / v10:
  simple_python 28 / 60 / 124, simple_java 65 / 57 / 98, live_simple
  60 / 76 / 101, live_multiple 29 / 36 / 70.
- Cause: the "ask" rows teach "no call when a value is not given". On BFCL
  AST, the model often thinks a value is missing when the reference fills it
  from context. The no-call read spreads to matched tools (same pattern as G25).
- MT miss_param does not improve (7.25 → 6.25). The missing-value refusal
  does not carry over into multi-turn, as G34 predicted (a preamble, then a call).
- Conclusion: v7 stays the final edit. Every decision-neuron edit in run 3
  moves along one axis: irrelevance against AST. v7 is the best point found.
