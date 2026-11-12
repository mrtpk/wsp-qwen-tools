# Decisions (v2)

## V2-D1. Method: weight-space programming only

- Date: 2026-11-07. The user asked for this.
- Decision: an LLM editor (Claude) reads and writes the weights directly.
  No LoRA, no fine-tuning, no other method.

## V2-D2. Strict protocol, openwsp tooling

- Date: 2026-11-07. The user asked for this.
- Decision: follow the openwsp protocol. Practice data only, a fixed
  forward-pass budget, BFCL-V4 as the test only. Use the openwsp kernel,
  refusals, and edit format. See `TASK.md`.
- The v1 documents are allowed as prior knowledge. The v1 data files made
  from BFCL are not allowed.

## V2-D3. openwsp source and environment

- openwsp: `https://github.com/openwsp/openwsp`, commit `8d09276`, cloned to
  `v2/openwsp`. Not changed.
- Python: the project `.venv` (torch 2.13.0+cu130, transformers 5.17.0,
  vllm 0.30.0). openwsp runs from source through `PYTHONPATH`. The project
  `uv.lock` does not change.
- openwsp pins `vllm==0.33.0` for its Modal backend. The local backend uses
  `vllm serve` 0.30.0, the same version as the v1 BFCL runs.

## V2-D4. Local forward backend

- openwsp ships only a Modal backend, and its public `generate` request takes
  plain user prompts without tools. Tool use needs tools in the chat.
- Decision: `backend/tooluse_backend.py` implements
  `forward(request, store)`.
  - `generate`: chats with tools through two `vllm serve` processes
    (original on port 8201, edited on port 8202), with the `qwen3_coder` tool
    parser and the `qwen3` reasoning parser. This is the BFCL path.
  - `teacher_forced`: the Experiment 8 schema, run by
    `replications.exp8.internals.capture` on HF Transformers.
- The edited server loads a checkpoint written from the store (bf16 values).
  It restarts when the edit changes.

## V2-D5. Budget accounting

- One accepted request is one pass. An identical request on the same weights
  returns the stored result and does not count. A failed pass does not count.
  This follows the openwsp README rule.
- The ledger is `work/passes.jsonl`. The code-run count and every cell are in
  `work/kernel-log.jsonl`.
- `backend/wsp.py` keeps the code-run count across invocations, because the
  openwsp CLI counts only inside one process.

## V2-D6. Practice data

| File | Source (HF dataset, revision) | Rows | Use |
|---|---|---|---|
| `when2call.jsonl` | `nvidia/When2Call` train_sft, `0582f7749d` | 14,999 | "no call": cannot (no fitting tool), ask (missing value), no tools |
| `toolace.jsonl` | `Team-ACE/ToolACE`, `6bda777c88` | 10,782 | first assistant turn: call or text |
| `apigen_mt.jsonl` | `Salesforce/APIGen-MT-5k`, `abc4a517d6` | 46,127 steps | multi-turn agent steps: call or text |

- Leak filter: a row is dropped when a user message equals a BFCL user
  message after normalization (`eval/bfcl_user_messages.py`). One When2Call
  row was dropped.
- Some rows use function names that also occur in BFCL (When2Call 690 rows,
  APIGen-MT 3,411 rows). Their questions differ. These rows stay.

## V2-D7. Run 2 is not blind

- Date: 2026-11-07. The user asked for this.
- Run 1 failed on BFCL (G11, G12). Claude saw the BFCL scores and the raw
  `simple_python` outputs before run 2.
- Decision: run 2 continues with the run-1 budget (21 of 50 passes left).
  Run 2 is recorded as not blind. Its BFCL result is not a clean test.
- Run 2 uses practice data only, as in run 1. No BFCL data goes into a fit.

## V2-D8. Codex test (real agent loop)

- Date: 2026-11-07. The user asked for this: test the model in Codex.
- Codex CLI 0.167.0 (`@openai/codex`, npm). It runs in a container
  (`codex/Dockerfile`, image `codex-eval:0.167.0`). Reason: the host blocks
  unprivileged user namespaces (AppArmor), so the Codex sandbox (bubblewrap)
  cannot start. Each shell command then fails with "Permission denied". The
  container is the sandbox: it mounts only the task work directory and
  `codex/home` (CODEX_HOME).
- Provider: `vllm serve` (0.30.0) through the Responses API
  (`wire_api = "responses"`). `codex/serve.sh` sets the thinking mode with
  `--default-chat-template-kwargs`, because Codex cannot send
  `enable_thinking`.
- Task suite: 16 small coding tasks (`codex/make_tasks.py`). Each task has a
  start repo, a prompt, and a check script outside the work directory. Each
  check fails on the start repo and passes on a reference solution.
- Models: base, v1 E4b, v2 run-1 edit (wsp2). Three repeats per task.
- Measures (`codex/summarize.py`): task pass rate, shell commands, failed
  commands, rejected tool calls, runs with no action, timeouts.

## V2-D9. Third practice source: glaive

- Date: 2026-11-07. The user agreed.
- `glaiveai/glaive-function-calling-v2`, revision `e7f4b645`, apache-2.0.
  Use: an out-of-source test of the reads (set `gT`). No fit uses it.

## V2-D10. Autonomous work overnight

- Date: 2026-11-08, about 00:55. The user gave permission to continue the
  experiment alone for about 8 hours.
- Plan: finish and score BFCL for edit v4 (two seeds, both modes). Then a
  run 3 under the same protocol rules: weight-space programming only,
  practice data only for fits, a new budget of 50 passes and 300 code runs,
  recorded as not blind (V2-D7). Target: thinking mode, where base
  irrelevance is 51% (over-calls) and v4 is switched off.

## V2-D11. Target convention: call when a tool matches

- Date: 2026-11-08, 03:06.
- Context: the practice sets have a "text" group (a text reference with
  matching tools). Edits v4 and v6 learned to answer in text there. BFCL
  counts a text answer as a failure when a tool matches (G25).
- Decision: from v7, the fits use the BFCL convention. A matching tool means
  "call". A swap (no matching tool) means "no call". A multi-turn text step
  after tool results keeps its "no call" label.
- This choice comes from BFCL results. Run 3 stays "not blind" (V2-D7).

## V2-D12. Correction of V2-D11

- Date: 2026-11-08, 11:26.
- V2-D11 has a wrong premise. The `tA` and `tB` text rows are not "matching
  tool, answer in text". Almost all of them say "a required parameter is
  missing" or "no function fits" (G32). Both are correct "no call" cases,
  and BFCL `miss_param` and `miss_func` want the same behavior.
- V2-D11 is withdrawn. Text rows keep their "no call" label. Edits v8 and v9
  (built on V2-D11) are not candidates.
- The real BFCL single-turn failure (G25) is a self-answer: the model solves
  the task in text when a matching tool exists and all parameters are given.
  The practice sets have no labeled rows of this kind.

## V2-D13. Pass limit 100 for run 3

- Date: 2026-11-08, 12:54. The user asked for this.
- The user did not set the 50-pass limit. It came from the openwsp
  Experiment 8 setup (V2-D2). The user wants a limit, but a larger one.
- Decision: run 3 has a limit of 100 forward passes (`wsp` default changed
  from 50 to 100). The code-run limit stays at 300 (25 used).
- Run 3 results after pass 50 are not comparable with the openwsp 50-pass
  setup. Findings state the pass number.
