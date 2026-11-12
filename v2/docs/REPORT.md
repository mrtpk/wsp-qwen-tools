# Report: weight edits for tool use in Qwen3.5-0.8B (v1 and v2)

Status: FINAL, 2026-11-08 (UTC+4).

Sources:

- v1: `v1/docs/` (`REPORT.md`, findings F1 to F49, decisions
  D1 to D18, `progress-log.md`).
- v2: `v2/docs/` (findings G1 to G35, decisions V2-D1 to
  V2-D13, `progress-log.md`, `TASK.md`).

Each statement gives its F, G, D, or V2-D number. Numbers come from those
documents. A statement marked "not confirmed" has no direct test.

## 1. Verdict

1. Small weight edits improve BFCL-V4 for Qwen3.5-0.8B by 1 to 2.6 points
   overall. They do not come near 60. The best scores are about 22.5
   (thinking) and 20.3 (non-thinking).
2. Every edit in both studies changes one thing: the decision "call a tool
   or answer in text". The gains come from irrelevance detection and a
   small part of multi-turn. The edits do not fix wrong argument values,
   multi-turn state errors, or memory errors. These failures are most of
   the remaining score loss.
3. All edits move along one trade-off: fewer false calls (irrelevance up)
   against fewer correct calls (AST down). An edit is good when it finds a
   better point on this trade-off. No edit removed the trade-off.
4. v1 recommends **E4b**: one rank-1 change of
   `layers.23.mlp.down_proj` (about 3.5M weights). The fit used BFCL data.
5. v2 recommends **v7**: two programmed neurons in the layer-23 MLP (6,144
   weights). No fit used BFCL data. But BFCL scores guided the choice
   between v2 candidates. So v2 is not a blind test (V2-D7).
6. No single edit wins in both modes:

| BFCL-V4 overall (web search = 0) | thinking | non-thinking | mean of the modes |
|---|---|---|---|
| base | 20.46 (2 seeds); 20.84 (3 seeds) | 17.68 | 19.07 |
| v1 E1b | **22.80** | 16.73 (1 seed) | 19.77 |
| v1 E4b | 21.49 | **20.29** | 20.89 |
| v2 v4 | 21.02 | 19.81 | 20.42 |
| v2 v7 | 22.54 (2 and 3 seeds) | 19.55 | **21.05** |

- v7 has the best mean of the two modes. The lead over E4b (+0.16) is
  smaller than the seed noise (about 0.5 to 1.5 overall). The two edits are
  equal within the measurement.
- E1b has the best thinking score, but it damages non-thinking mode (F31).
- v7 is better in thinking mode (+1.05 against E4b). E4b is better in
  non-thinking mode (+0.74 against v7).

## 2. Common setup

### 2.1 Model

- `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`.
  The v1 input `models-text/modelq.base64` is this checkpoint
  (SHA-256 match, F1).
- 873,438,784 parameters. 24 decoder layers, hidden size 1,024, MLP size
  3,584 (SwiGLU), vocabulary 248,320 (F2).
- Layers 3, 7, 11, 15, 19, and 23 use full attention. The other layers use
  Gated DeltaNet (linear attention) (F2).
- `<tool_call>` is one token (id 248058) (F6).
- The checkpoint also holds a vision encoder and an MTP head. No edit
  touches them.

### 2.2 Benchmark and harness

- BFCL-V4 with `bfcl-eval==2026.3.23` in a separate venv (D1). Categories:
  `single_turn`, `multi_turn`, `memory`. 5,017 entries per run.
- Requests go to `vllm serve` (`vllm==0.30.0`) with the `qwen3_coder` tool
  parser and the `qwen3` reasoning parser (D2).
- Sampling: temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5,
  maximum 16,384 output tokens (D3). Seeds: vLLM seed 0, 1, and (for base
  and v7 thinking) 2.
- Web search is not run (no SerpAPI key, D4). The overall score counts it
  as 0. Web search is 20% of the overall weight. So the overall numbers are
  not comparable to the public leaderboard. Qwen reports 25.3 (thinking)
  for this model (F12). The settings match is not confirmed.
- Overall weights: non-live 10, live 10, irrelevance 10, multi-turn 30,
  agentic 40 (memory and web search are the agentic part).
- GPU: one RTX 5090. vLLM needs `CUDA_HOME` and `FLASHINFER_EXTRA_LDFLAGS`
  (D5).

### 2.3 Noise

- Base thinking, seed 0 against seed 1: overall +0.84, memory +4.30 (F37).
- Base thinking, three seeds: 20.04, 20.88, 21.59. Spread 1.55 (G33).
- Memory varies by up to 4.3 points between seeds of one model (G25).
- Relevance has only 16 entries. One entry is 6.25 points.
- Rule used in both studies: a gain counts only when every seed of the edit
  is above every seed of the base.

## 3. The base model: what fails

### 3.1 Base scores (two seeds, F39, F43)

| Measure | thinking | non-thinking |
|---|---|---|
| Overall | 20.46 | 17.68 |
| Non-live AST | 56.87 | 33.98 |
| Live AST | 55.48 | 43.89 |
| Multi-turn | 5.44 | 3.44 |
| Memory | 12.47 | 5.48 |
| Relevance | 75.00 | 71.88 |
| Irrelevance | 51.04 | 77.72 |
| Text instead of a call (of 2,501 call entries) | 156 (one seed) | 685 / 699 |

### 3.2 The two decision faults

- Non-thinking mode under-calls. The model answers in text when a fitting
  tool exists. Example: `simple_python_3` gives `algebra.quadratic_roots`,
  and the model solves the equation in LaTeX. This is 47% of the
  single-turn failures (F5).
- Thinking mode over-calls. The model calls a tool for 46.7% of the prompts
  where no tool fits (525 of 1,124) (F13). Irrelevance is 51%.

### 3.3 The other faults (not decision faults)

- Wrong argument values are the largest single-turn failure in thinking
  mode: `value_error:string` 365, `value_error:others` 120 of 1,056
  failures (D11). Examples: invented values for optional parameters
  ("any"), wrong values ("Rohnert Park, TX" for "CA"), wrong conventions
  ("London" for "London, UK") (G15).
- Multi-turn is near 5%. At about 0.6 correct steps per step, a task with
  five calls succeeds about 8% of the time (G17).
- In multi-turn failures of v7, the first differing turn is: more calls
  than the reference 701, a call where the reference has none 328, fewer
  calls 160, other functions 122, no decoded result 95, no call 53 (G32).
- A harness retry that returns format errors to the model does not help.
  Format errors are a small share (G15).

## 4. How the model decides (v1 and v2 mechanism results)

### 4.1 Non-thinking mode

- The decision is made at the first assistant token. `p(<tool_call>)` there
  predicts the sampled outcome with AUROC 0.914 (F6).
- The call logit is a query prior plus a relevance gap (+2.9). The prior is
  low for questions that the model "can answer itself" (F10). The prior is
  a separate linear signal (F11, H4).
- Tool-swap test: with the right tools, the call probability is higher than
  with unrelated tools in 97.9% of 858 pairs (F9). The relevance signal is
  real.

### 4.2 Thinking mode

- The decision is made inside the reasoning. After `</think>`, `p_call` is
  almost binary (AUROC 0.980 for the sampled outcome) (F13).
- Most irrelevance traces have no clear conclusion. Then the model falls
  back on "tools are present, so call". An explicit "I will answer
  directly" sentence removes 83% of the over-calls (F17).
- The presence penalty adds about 7 points of over-calling (F15).
- Direct logit attribution: the token mixers of layers 15 to 22 write the
  decision (F18). The Gated DeltaNet of layer 22 holds almost all of the
  causal effect under mean patching (F19).
- No single MLP neuron in layers 16 to 23 is a dedicated `<tool_call>`
  writer (largest |z| 5.0) (G1).
- When a call is planned in the reasoning, a push at the decision token
  does not stop it. The model writes a short preamble ("I'll fetch ...")
  and then calls (G6, G34). In v2 run 3, the `wB` set had 176 such
  preamble calls after the edit (base 2).

### 4.3 Can a linear read beat the model?

- v1: yes, on BFCL data. A probe at layer 12 catches 96.5% of the needed
  calls at the false-positive rate where the output catches 68.5% (F7).
- v2 corrected two wrong results on the way:
  - G12: the run-1 read separated the data sources (When2Call against
    ToolACE), not the decision. Each label came from one source.
  - G13: with too little ridge (λ = 0.01), no read beat the model. G16
    corrected this: with λ = 1 to 30, the reads beat the model.
- v2 final numbers (AUROC, held-out or 8-fold CV, same source):

| Decision | model `p_call` | linear read (λ = 3, layer-23 MLP input) |
|---|---|---|
| Single-turn, non-thinking (ToolACE `cB`, held-out) | 0.839 | 0.947 |
| Multi-turn, non-thinking (APIGen-MT `mt_B`, held-out) | 0.845 | 0.933 |
| Single-turn, thinking (`tA`, CV) | 0.916 | 0.939 to 0.941 |
| Multi-turn, thinking (`mA`, CV) | 0.806 | 0.901 to 0.910 |
| Glaive (third source, never fitted), non-thinking | 0.934 | 0.955 to 0.958 |

- A read from one source does not transfer to the other source (single-turn
  read on multi-turn: 0.675; multi-turn read on single-turn: 0.505). A joint
  read works on both (G18). The single-turn read transfers to glaive (G19).

## 5. v1: the BFCL-fitted rank-1 edit

### 5.1 Context

- Agent: Cursor. Dates: 2026-11-06 14:13 to 2026-11-07 14:50.
- The starting point was the RikiyaT "weight-space programming" edit
  (Experiment 8, MATH-500 thinking loops). It changes 64 columns of
  `down_proj` in layers 15 and 19 (131,072 weights, rank 1 each) (F3).
- For tool calls, that edit is worse than base in both modes: thinking
  20.04 → 19.04, non-thinking 17.56 → 15.66 (one seed each, F12). v1 used
  it as a negative reference (D10). MATH-500 was not tested.
- Goal order (D6): first learn the mechanism, then edit.

### 5.2 The edit form

- Site: the input of `layers.23.mlp.down_proj` (3,584-d). Its output goes
  straight to the final norm. So the effect on the logits is linear (D12).
- Form: W' = W − k · u · ρᵀ (rank 1).
- u (write): the residual direction that raises the `<tool_call>` logit and
  changes the logits of the common tokens least (regularized least squares,
  D12, D13).
- ρ (read): a logistic direction for "irrelevance against needs a call",
  plus bias proxies. The weights have no bias, so the mean-activation
  directions act as the bias (F21).
- E4 fits ρ on the decision tokens of both modes (even rows of BFCL), with
  a separate threshold for each mode (D16, F32). k = 5 (F33).

### 5.3 v1 edits

| Edit | Change | Result | Lesson |
|---|---|---|---|
| E1 | u = plain readout | background KL 1.2 at k = 20 | protect the common tokens in u |
| E1b | fit on the thinking decision token only | thinking +2.34, non-thinking −0.95 (F31, F47) | fit every decision context that the edit sees |
| E2 | also protect the tool-call tokens | same as E1b (F26) | the in-call side effect was already small |
| E3 | background penalty | KL 13 times larger, then no change (F28, F30) | measure KL only at generated positions |
| **E4b** | ρ fitted on both modes, k = 5 | thinking +1.03, non-thinking +2.61 (F39, F43) | **recommended** |
| E5c | constraint at mid-conversation decisions | fewest multi-turn calls, no multi-turn gain (F48) | fewer calls do not cause the E1b multi-turn gain |

### 5.4 v1 results (two seeds unless stated)

| Measure | base think | E1b think | E4b think | base nothink | E4b nothink |
|---|---|---|---|---|---|
| Overall | 20.46 | 22.80 | 21.49 | 17.68 | 20.29 |
| Non-live AST | 56.87 | 57.22 | 58.68 | 33.98 | 49.55 |
| Live AST | 55.48 | 51.74 | 53.03 | 43.89 | 46.45 |
| Multi-turn | 5.44 | 7.19 | 5.38 | 3.44 | 3.31 |
| Memory | 12.47 | 12.79 | 8.71 | 5.48 | 5.16 |
| Irrelevance | 51.04 | 71.94 | 69.69 | 77.72 | 86.66 |

- E4b halves the non-thinking "text instead of a call" failures (685 / 699
  → 332 / 328) (F43).
- Tool-swap pairs (858, not in the fit): E4b removes 41% of the false calls
  in thinking mode (0.385 → 0.228). In non-thinking mode it raises the
  correct calls 0.697 → 0.830 with no more false calls (F41).
- Side effects: KL per generated token 0.0038 (reasoning), 0.0012 (inside
  the call). The RikiyaT edit: 0.0138 and 0.0073 (F30, F33).
- The change is dense: about 3.5M weights, relative norm 0.307.
- Costs: thinking live AST −2.45, memory −3.76. Most new live failures keep
  the call and change an argument value (F36). The memory loss is not
  confirmed as real (F40).

### 5.5 v1 limits

- The read and the thresholds come from BFCL data. The tool-swap pairs and
  held-out rows reduce the risk of overfit, but do not remove it.
- E5 used BFCL multi-turn traces. Its multi-turn score is not a clean test
  (D18).
- Some runs have 0 to 17 context-length errors, which count as failures.

## 6. v2: weight-space programming under a strict protocol

### 6.1 Rules

- Method: weight-space programming only. The editor (Claude) reads and
  writes the weights directly. No LoRA, no training, no gradients (V2-D1).
- Protocol: openwsp (commit `8d09276`), Experiment 8 rules with a tool-use
  goal (V2-D2, `TASK.md`).
  - The editor writes Python cells. The kernel runs them (`./wsp run`).
    The kernel has `load`, `load_original`, `save_edit`, and other helpers.
    It does not run model layers.
  - Model outputs come only from forward passes (`./wsp forward`):
    `generate` (vLLM, tools in the chat) or `teacher_forced` / `probe`
    (HF Transformers, activations and log probabilities).
  - A failed pass and an identical repeat do not count (V2-D5).
  - The kernel refuses edits of the final norm, the embedding, the output
    layer, and whole-tensor rescaling.
- Data: practice data only. BFCL data, prompts, and results must not go
  into a fit. The v1 activation files are not allowed. The v1 documents are
  allowed as prior knowledge (V2-D2).
- Budgets: 50 passes and 300 code runs per run. For run 3 the user raised
  the pass limit to 100 (V2-D13).
- Evaluation: BFCL-V4 after the export, with the v1 harness and sampling.

### 6.2 Tooling (V2-D3, V2-D4)

- `backend/tooluse_backend.py`: a local forward backend. Two `vllm serve`
  processes (original on port 8201, edited on port 8202). The edited server
  loads a checkpoint from the edit store and restarts when the edit
  changes.
- The `probe` request type reads MLP inputs, down_proj inputs, the residual
  stream, and next-token log probabilities at chosen positions.
- `eval/build_checkpoint.py` builds a full checkpoint from an exported edit
  with openwsp `apply_edit` (hash checks).
- The project `uv.lock` did not change. openwsp runs from source.

### 6.3 Practice data (V2-D6, V2-D9)

| Source | Rows | Use |
|---|---|---|
| `nvidia/When2Call` (train_sft) | 14,999 | no-call cases: no fitting tool ("cannot"), missing value ("ask"), no tools |
| `Team-ACE/ToolACE` | 10,782 | first assistant turn: call or text |
| `Salesforce/APIGen-MT-5k` | 46,127 steps | multi-turn agent steps: call or text |
| `glaiveai/glaive-function-calling-v2` | 70,528 | third source, test only in run 2 and in `gB` |

- Leak filter: a row is dropped when a user message equals a BFCL user
  message after normalization. One When2Call row and one glaive row were
  dropped.
- Set types used in the fits:
  - "call": a fitting tool, a call reference.
  - "swap": the same question with the unrelated tools of another row (no
    fitting tool).
  - "text": a text reference with tools present. G32 shows these are almost
    all missing-parameter or no-function cases.
  - "call step" and "text step": multi-turn steps from APIGen-MT.
- Sets end in `A` (fit) or `B` (held-out). Each has 256 to 320 rows.

### 6.4 The program form (neuron programming, G4)

Each v2 edit overwrites unused neurons of the layer-23 MLP. One neuron j
has three vectors of 1,024 values (3,072 weights):

- Gate row `gate_proj[j]`: a direction that is positive only at decision
  positions (the token where the model starts `<tool_call>` or text).
  silu(gate · x) is then about 0 at all other positions.
- Up row `up_proj[j]`: r − θ · q / 11.9. r is a decision read (LDA, λ = 3,
  scaled to sd 1). q is the "decision position" direction. q · x is 11.9 at
  decision positions (CV 0.04 to 0.06), so q / 11.9 acts as a bias of 1 and
  θ is the threshold (G3).
- Down column `down_proj[:, j]`: K · u. u raises the `<tool_call>` logit
  and changes the 4,000 most common practice tokens least.
- The neuron output is K · silu(gate · x) · (up · x) · u. Layer 23 is the
  last layer. No later layer reads its output.
- Slots: neuron 1488 (lowest use, 8.3e-3), then 1152 and 2088 (next lowest
  use).
- θ and K come from a grid search on the fit set, with a model of the
  call probability: p = sigmoid(logit(p₀) + c · K · h), where c (median 4.6,
  IQR 3.8 to 5.0) comes from an edited probe pass (G20).

### 6.5 Run 1: one neuron, a confounded read (G1 to G12)

- Time: 2026-11-07 15:27 to 16:02. 29 of 50 passes, 43 of 300 code runs.
- Edit `run1-n1488` (model `qwen35-wsp2`): neuron 1488, gate for single-turn
  decisions, a multi-turn "off" term, and a δ term for preamble positions.
- Practice validation looked good: single-turn non-thinking accuracy 0.652
  → 0.930 (G8). KL per generated token at most 0.0022 (G9).
- BFCL result (two seeds): thinking 20.46 → 21.32, non-thinking 17.68 →
  17.74 (G11). Non-thinking `simple_python`: 325 of 400 rows fail with no
  call (base 107). Non-live simple 30.41 → 8.21.
- Cause (G12): in the single-turn sets, every "no call" row came from
  When2Call and every "call" row came from ToolACE. The read learned the
  source. BFCL single-call questions look "When2Call-like" to it, so the
  neuron pushed "no call". The practice validation had the same confound.
- Lesson: fit and validate a read only where the call and no-call labels
  come from the same source and format.

### 6.6 Run 2: clean reads, edit v4 (G13 to G22)

- Time: 2026-11-07 20:15 to 23:30. Not blind: Claude saw the run-1 BFCL
  results (V2-D7). It continued the run-1 budget: 47 of 50 passes, 78 of
  300 code runs in total.
- Clean set `cA`: all rows from ToolACE (call, swap, text) (G13).
- G13 → G16: the ridge correction (section 4.3).
- G17: multi-turn per-step accuracy (teacher-forced, 93 gold steps): call
  decision 0.70, function name 0.93, parameter names 1.00, parameter values
  0.95, whole call 0.60.
- Edit v3 (G20): neuron 1488 with a joint read (`cA` + `mt_A`, non-thinking).
  Held-out generation, non-thinking: `cB` 0.707 → 0.805, glaive 0.738 →
  0.871, `mt_B` 0.734 → 0.809. Thinking multi-turn got worse (0.746 →
  0.676).
- Edit v4 (G21): v3 plus a mode read in the gate. The gate is open at 100%
  of non-thinking decisions and 1 to 2% of thinking decisions. Model
  `qwen35-wsp4`, 3,070 changed weights.
- BFCL (G22, two seeds):

| Measure | base nothink | v4 nothink | base think | v4 think |
|---|---|---|---|---|
| Overall | 17.68 | **19.81** | 20.46 | 21.02 |
| – seeds | 17.63 / 17.74 | 19.32 / 20.31 | 20.04 / 20.88 | 20.93 / 21.10 |
| Non-live AST | 33.98 | 39.09 | 56.87 | 54.99 |
| Live | 43.89 | 44.45 | 55.48 | 53.85 |
| Multi-turn | 3.44 | 4.38 | 5.44 | 6.44 |
| Memory | 5.48 | 7.42 | 12.47 | 15.16 |
| Irrelevance | 77.72 | 86.66 | 51.04 | 51.67 |

- v4 is the first v2 BFCL gain with no BFCL data in the fit. Non-thinking
  +2.13: both v4 seeds are above both base seeds.

### 6.7 The Codex test (G14)

- Codex CLI 0.167.0 in a Docker container, vLLM through the Responses API.
  16 small coding tasks, 3 repeats (V2-D8).
- Pass rate: base 12% (thinking), E4b 12%, wsp2 10%. Non-thinking: 4% to
  15% (runs stopped at the user's request).
- The main fault is tool use, not code. The model calls `write_stdin` on a
  session that does not exist (base thinking: 1,126 times in 48 runs). It
  calls tools that do not exist (`run_command`, `read_file`).
- E4b cuts the rejected tool calls by about 4 times, but it does not pass
  more tasks.
- The user parked the Codex path.

### 6.8 Run 3: a second neuron for thinking mode (G23 to G35)

- Time: 2026-11-08 01:04 to 14:10. Work directory `work3`, a new budget
  (V2-D10). The store starts from v4. 35 of 100 passes, 29 of 300 code
  runs. Not blind.
- New sets: `tA`, `tB` (ToolACE), `mA`, `mB` (APIGen-MT), `gB` (glaive),
  later `wA`, `wB` (When2Call "ask").

| Edit | Content | Practice result | BFCL thinking (2 seeds) | Kept |
|---|---|---|---|---|
| v5 | v4 + T (gate = decision + mode read) | gate open at 72 to 91% of non-decision positions; `tB` name accuracy 0.92 → 0.48 (G23) | not run | no |
| v6 | v4 + T6 (gate G_T, read on call, swap, text) | `tB` 0.723 → 0.805, glaive 0.801 → 0.922 (G24) | 22.05 | no |
| **v7** | v4 + T7 (read without the text group) | `tB` 0.738 → 0.785; swap calls 0.38 → 0.23 (G26) | **22.54** | **yes** |
| v8 | v7 + N (non-thinking neuron, text → call) | `gB` swap calls 0.20 → 0.34 (G28) | = v7 (N is off) | no |
| v9 | v8 with T8 (text → call in thinking) | `gB` text calls 0.25 → 0.83 (G30) | 20.96 | no |
| v10 | v4 + T11 (read with When2Call "ask" rows) | `tB` 0.684 → 0.820; `wB` preamble calls 2 → 176 (G34) | 21.99 | no |

Details:

- **G_T (G24).** A ridge regression without intercept at the layer-23 MLP
  input. Target +4 at thinking decisions, −8 at all other position kinds
  in both modes. Held-out: thinking decisions min 2.58, every other kind
  max −3.38. This fixed the v5 fault: an additive gate cannot do an AND of
  "decision" and "thinking mode".
- **v6 (G25).** BFCL thinking 22.05, irrelevance 72.12. But non-live AST
  fell 56.87 → 53.11. The `ast_decoder` failures rose (`simple_python`
  28 → 92). The model wrote worked solutions in text. The "text" group in
  the read taught a refusal, and the refusal spread to self-answer tasks.
- **v7 (G26, G27, G33).** The read uses `tA` call against swap and `mA`
  call step against text step. θ = 0.875, K = 1.0. Edit `run3-v7`, model
  `qwen35-wsp7`, `edit_sha256` 6b456447….
- **V2-D11 and V2-D12.** After v7, Claude assumed that the text rows are
  "a tool matches, answer in text" and pushed them toward a call (v8, v9).
  This premise was wrong. The text rows are missing-parameter and
  no-function cases (G32). V2-D12 withdrew V2-D11.
- **v8 (G29).** The non-thinking neuron N raised non-live AST 39.11 → 47.03
  and live 44.23 → 48.00 (the best live of all models). Irrelevance fell
  87.11 → 78.93. Overall 19.76 (v7 19.55): no net change.
- **v9 (G31).** Thinking non-live AST 59.46 (the best of all models), but
  irrelevance 49.58 and multi-turn 5.81. Overall 20.96.
- **v10 (G34, G35).** The read adds 256 When2Call "ask" rows as "no call".
  In-sample `wA` p_call 0.79 → 0.07. BFCL: irrelevance 73.90 (+6.7 against
  v7), non-live AST 51.88 (−3.9). `ast_decoder` errors doubled against v7
  (`simple_python` 60 → 124). Multi-turn `miss_param` did not improve
  (7.25 → 6.25).

### 6.9 v7 in detail

- Weights: 6,144 values in three tensors of layer 23. Rows 1488 and 1152 of
  `gate_proj` and `up_proj`, columns 1488 and 1152 of `down_proj`.
- Neuron 1488 (from v4) acts at non-thinking decisions. Neuron 1152 (T7)
  acts at thinking decisions. Each is closed (silu ≈ 0) at other positions.
- BFCL-V4, thinking, three seeds (G33):

| Metric | base | v7 | E4b (2 seeds) |
|---|---|---|---|
| Overall | 20.04 / 20.88 / 21.59 = 20.84 | 22.65 / 22.42 / 22.56 = **22.54** | 21.40 / 21.59 = 21.49 |
| Non-live AST | 57.50 | 55.21 | 58.68 |
| Live | 54.95 | 54.16 | 53.03 |
| Multi-turn | 5.54 | 7.38 | 5.38 |
| Memory | 13.90 | 13.62 | 8.71 |
| Irrelevance | 51.52 | 66.70 | 69.69 |

- The lowest v7 seed (22.42) is above the highest base seed (21.59).
- Gain +1.70: irrelevance +15.2 and multi-turn +1.8. Cost: non-live AST
  −2.3.
- Thinking multi-turn, two seeds, base → v7: base 10.0 → 12.5, missing
  function 0.75 → 1.75, missing parameter 5.0 → 7.25, long context 6.0 →
  10.0 (G27).
- Non-thinking (two seeds): overall 19.55, non-live AST 39.11, live 44.23,
  irrelevance 87.11, relevance 43.75, multi-turn 4.31, memory 6.02 (G29).
  Relevance (16 entries) falls from 71.88. This is a real cost or noise; it
  is not confirmed.

## 7. v1 and v2 compared

| Item | v1 (E4b) | v2 (v7) |
|---|---|---|
| Agent | Cursor | Claude |
| Method | rank-1 edit, read and write directions fitted by regression | programmed MLP neurons (gate, up, down set by hand from reads) |
| Changed weights | about 3.5M (dense rank 1) | 6,144 (two neurons) |
| Fit data | BFCL prompts and activations | practice data only (ToolACE, APIGen-MT, When2Call) |
| Blind | no (fit on the test set) | no (BFCL scores chose between candidates) |
| Forward passes | not limited | 47 (runs 1, 2) + 35 (run 3) |
| Modes | one edit, one threshold per mode | one neuron per mode |
| BFCL thinking | 21.49 | 22.54 |
| BFCL non-thinking | 20.29 | 19.55 |
| Main gain | irrelevance (both modes), non-thinking AST | irrelevance (thinking), multi-turn (thinking) |
| Main cost | thinking live AST, memory | thinking non-live AST, non-thinking relevance |

- Both studies find the same structure: the decision is linear and readable
  at layer 23, and a late push can move it.
- The v2 form is about 570 times smaller and is local. It acts only at
  decision positions.
- v2 is the stronger result on data hygiene: no BFCL row went into a fit.
  It is weaker than a clean test, because BFCL guided the choices of runs 2
  and 3 (section 9).

## 8. Lessons

### 8.1 About the model and the method

1. A decision edit trades irrelevance against AST. Count both before an
   edit is accepted (D10, G31, G35).
2. A label that equals the data source makes a read learn the source
   (G12). Fit call and no-call rows from one source.
3. Use enough ridge. λ = 0.01 hid a real signal (G13 → G16).
4. A read does not transfer across sources unless it saw both (G18). Test
   on a third source before an edit (G19).
5. A gate must be checked at positions that are not decisions. The additive
   "decision + mode" gate opened almost everywhere (G23).
6. A ridge gate trained with negative targets at every other position kind
   gives a clean AND (G24).
7. A "no call" target spreads. Missing-value and refusal rows teach the
   model to answer in text on BFCL tasks that have a matching tool and all
   values (G25, G35).
8. A call planned in the reasoning text cannot be stopped at the decision
   token. The model writes a preamble and calls (G6, G34).
9. Fewer multi-turn calls do not give a multi-turn gain (F48).
10. Practice validation can mislead in both directions. Run 1 looked good
    on practice data and failed on BFCL (G8, G11). v10 had the best
    practice accuracy on `tB` and lost on BFCL (G34, G35).

### 8.2 About the operation

1. The openwsp kernel keeps its own `vllm serve` processes (20 GB) after
   passes. Stop them by PID before a BFCL run.
2. A BFCL generation job can hang with an idle server. One hang cost about
   5.7 hours (2026-11-08 04:30 to 10:17). Use a stall check: the log is
   idle for more than 900 s while the process lives.
3. Never use `pkill -f` with a pattern that is also in your own command.
4. Probe passes fail with CUDA OOM while BFCL servers run. Failed passes do
   not count, but they lose time.
5. Keep the sum of vLLM memory fractions on one GPU at or below 0.85.

## 9. Limits and threats to validity

1. **Selection bias in v2.** Runs 2 and 3 saw BFCL scores. v7 was chosen
   from v6, v7, v9, and v10 by BFCL thinking score. Its 22.54 is an
   optimistic estimate. The third seed (seed 2) was run after the choice.
   It tests the sampling noise, but not the item selection.
2. **v1 is fitted on BFCL.** E4b numbers are in-sample for the read and the
   thresholds.
3. **No web search.** Overall numbers are not leaderboard numbers.
4. **Few seeds.** Most results have two seeds. Memory and relevance are
   noisy.
5. **Causes not tested by intervention.** The G35 cause ("the model thinks
   a value is missing when BFCL fills it from context") is not confirmed.
   G25 had a wrong cause first, corrected by G32.
6. **No real agent use.** In Codex, no model passes more than 15% of small
   tasks (G14). The edits do not change this.
7. **Other abilities.** MATH-500 and general chat quality were not tested.
   The v2 KL checks (G9) cover practice prompts only.
8. **The pass budget changed.** Run 3 used a limit of 100, not 50. Run 3
   used 35, so it stayed inside the original 50.

## 10. Why 60 is out of reach with this method

- BFCL-V4 overall is 40% agentic (memory and web search) and 30%
  multi-turn. The base is near 5% on multi-turn and near 12% on memory
  (thinking).
- A decision neuron sets "call or no call" at one token. The best decision
  edit adds about 1.7 to 2.6 points overall.
- To reach 60, the model needs correct argument values, multi-step state
  tracking, and memory use. These are model capability limits at 0.8B. The
  public scores of larger Qwen3.5 models (not confirmed: 2B 43.6, 4B 50.3,
  9B 66.1) suggest that 60 needs a model near 9B.

## 11. Open items

1. A clean blind test of the v2 method: a new run that never sees BFCL, on
   a new seed set, with one pre-registered export.
2. Argument values: the largest single-turn failure (G15, F16). No edit in
   either study targets them.
3. Multi-turn state errors and "more calls than the reference" (G32).
4. A combined edit: E4b's non-thinking strength with v7's thinking
   neuron. Not tested.
5. MATH-500 and general quality checks for v7 and E4b.

## 12. Artifacts

| Item | Path |
|---|---|
| v1 docs | `v1/docs/` |
| v1 E4b model | `v1/models/qwen35-e4b` |
| v1 scripts | `v1/wsp/` (`edit_e4.py`, `collect_sites.py`) |
| v1 BFCL harness | `v1/bfcl/` |
| v2 docs | `v2/docs/` |
| v2 cells | `v2/cells/run1/`, `run2/`, `run3/` |
| v2 pass ledgers | `v2/work/passes.jsonl`, `work3/passes.jsonl` |
| v2 exported edits | `v2/edits/` (`run1-n1488`, `run2-v4`, `run3-v6` to `run3-v10`) |
| v2 checkpoints | `v2/models/` (`qwen35-wsp2`, `wsp4`, `wsp6` to `wsp10`) |
| v2 final edit | `v2/edits/run3-v7`, `models/qwen35-wsp7` |
| v2 BFCL results and scores | `v2/eval/result/`, `eval/score/` |
| v2 run scripts | `v2/tmuxlogs/` (`bfclN.sh`, `scoreN.sh`) |
| Codex test | `v2/codex/` |

Disk use: `v1` 23 GB, `v2` 22 GB (12 GB of each
is checkpoints).

### 12.1 Reproduce v7 on BFCL

1. Build the checkpoint: `PYTHONPATH=openwsp/src ../.venv/bin/python
   eval/build_checkpoint.py <base-dir> edits/run3-v7 models/qwen35-wsp7`.
2. Serve: `eval/serve.sh models/qwen35-wsp7 qwen35-wsp7 8301 0.40 262144 0`.
3. Generate: `QWEN35_EXTRA_MODELS=qwen35-wsp7 eval/gen.sh
   qwen35-wsp7-think-FC 8301 single_turn,multi_turn,memory`.
4. Score: see `tmuxlogs/score10.sh` for the command and the model list.

## 13. Timeline

| Date and time (UTC+4) | Event |
|---|---|
| 2026-11-06 14:13 | v1 starts (Cursor). |
| 2026-11-07 14:25 | v1 final report (E4b). |
| 2026-11-07 15:15 | v2 starts (Claude, openwsp). |
| 2026-11-07 16:02 | v2 run 1 export. BFCL fails on single calls (G11). |
| 2026-11-07 20:15 | v2 run 2 starts. |
| 2026-11-07 21:00 to 22:37 | Codex test (G14). |
| 2026-11-07 23:30 | Edit v4 export. |
| 2026-11-08 01:04 | BFCL v4 scored (G22). Run 3 starts. |
| 2026-11-08 04:15 | BFCL v7 scored (G27). |
| 2026-11-08 04:30 to 10:17 | Hung BFCL job, 5.7 hours lost. |
| 2026-11-08 11:24 | v9 scored. V2-D11 withdrawn (V2-D12). |
| 2026-11-08 12:28 | Third seed confirms v7 (G33). |
| 2026-11-08 14:09 | v10 scored, not kept (G35). |
| 2026-11-08 14:10 | v2 concluded. Final edit v7. |
