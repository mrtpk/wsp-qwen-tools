# Progress log (v2)

All times are UTC+4. Times come from the file timestamps where available.

## 2026-11-07

- Before 15:15. The user gave v2 to Claude. v1 is in `v1/`
  (read only). Read the weight-space programming blog post and the openwsp
  docs. The user chose: weight-space programming only, strict protocol,
  openwsp tooling, v1 as reference (V2-D1, V2-D2).
- 15:15. Cloned openwsp `8d09276`.
- 15:16 to 15:20. Downloaded and converted the practice data. Leak check
  against BFCL (V2-D6).
- 15:17 to 15:26. Wrote `backend/tooluse_backend.py`, `backend/wsp.py`, and
  the `wsp` command (V2-D4, V2-D5). Smoke test in a separate work directory
  (not counted in the budget):
  - The first vLLM start failed: `ninja` was not on `PATH`. v1 used
    `uv run`, which adds `.venv/bin`. Fix: the backend adds `.venv/bin` to
    `PATH`.
  - Then: generation in both arms, a free repeat of an identical request, a
    teacher-forced read, an edit with a restarted edited server, and the
    refusal of an embedding edit all worked.
  - Removed the smoke work directory.
- 15:26. Added the `probe` request type (compact reads at chosen positions:
  MLP inputs, down_proj inputs, residual stream, next-token log
  probabilities). Smoke test outside the work directory.
- 15:27. Run 1 started. Passes 1 to 4: base behavior on the practice sets
  `st_A` and `mt_A`, in both modes (`cells/run1/make_sets.py`).
- 15:27. Code run 1 (`c01_readout.py`): no single MLP neuron in layers 16 to
  23 is a strong `<tool_call>` writer (largest |z| 5.0, layer 23). G1.
- 15:29. Passes 1 to 4 complete. G2.
- 15:30 to 15:32. Passes 5 to 8: reads at the layer-23 MLP input. The first
  version of `c03_probe_requests.py` failed: some APIGen-MT arguments are
  JSON strings inside JSON strings. Fixed. G3.
- 15:33 to 15:44. Program design, calibration passes 9 to 15. G4, G5.
- 15:45 to 15:49. Context term (multi-turn off). Validation passes 16 to 19.
  The thinking-mode gain was small. G6.
- 15:50 to 15:55. Pass 20 (answer positions). The kernel refused one cell
  with `exec` (protocol guard, not counted). Pass 21 had wrong positions:
  the chat template has an example `<tool_call>` in the system prompt. Pass
  22 repeated it with the search after the prompt. G7. Chose δ = 0.5,
  β = 2.33, K = 0.35.
- 15:57. Validation passes 23 to 26. G8.
- 15:58 to 16:01. Veto neuron tested and rejected (pass 27). Damage passes
  28 and 29. G6, G9.
- 16:02. Exported `edits/run1-n1488`. The protocol run ends: 29 of 50 passes,
  43 of 300 code runs. G10.
- 16:03. Built `models/qwen35-wsp2` from the exported edit with openwsp
  `apply_edit` (hash checks pass). `eval/build_checkpoint.py`.
- 16:04. BFCL-V4 started: `qwen35-wsp2` (seed 0, port 8301) and
  `qwen35-wsp2-s2` (seed 1, port 8302), both modes, `single_turn`,
  `multi_turn`, `memory`. Harness: the v1 scripts (`eval/`), v1 BFCL venv
  read only.
- 17:03. The seed-0 non-thinking BFCL run ended. The other three still run.
- 17:25 to 17:30. Scored the seed-0 non-thinking run. `eval/score.sh` needed
  `OPENAI_API_KEY=EMPTY` (as in v1). Added. Non-live single-call accuracy
  fell. Cause found: label and source are the same in the single-turn
  practice sets. G11, G12.
- 17:34, 17:56, 18:02. The other three BFCL generation runs ended (log times).
- 18:03 to 18:05. Scored them. Stopped the BFCL servers on ports 8301 and
  8302. G11 now has the two-seed means of both modes.
- 20:15. Run 2 started, not blind (V2-D7). Built the clean sets `cA` and
  `cB` (`cells/run2/make_sets.py`).
- 20:16. Pass 30: thinking generation on `cA`.
- 21:03. Installed Codex CLI 0.167.0 (npm, user prefix). Smoke test against
  the original vLLM server (Responses API): the pipeline works.
- 21:05. Passes 31 and 32: probes on `cA`. The first `c36` failed (a lookup
  error in the cell, code run 44).
- 21:06. Cells `c37`, `c38`. The model's own `p_call` beats every linear read.
  The decision-neuron route stops. G13. Budget: 32 of 50 passes, 47 of 300
  code runs.
- 21:08 to 21:30. Codex test setup (V2-D8). The first suite run was invalid:
  the Codex sandbox cannot start on this host. Moved Codex into Docker.
- 21:30 to 21:55. Codex suite, thinking mode, 3 models × 16 tasks × 3.
- 22:00 to 22:37. Codex suite, non-thinking mode. Stopped at the user's
  request. G14. The user: the Codex path does not work; focus on BFCL.
- 22:37 to 22:46. Retry diagnostic on BFCL single-turn (base, seed 0, both
  modes). G15.
- 22:47 to 22:54. Multi-turn diagnostics: pass 33 (teacher-forced gold
  steps), pass 34 (multi-turn decision probe). Cells `c39` to `c45`. A ridge
  sweep shows that G13 used too little ridge: the reads beat the model.
  G13 corrected, G16, G17. Budget: 34 of 50 passes, 57 of 300 code runs.
- To 22:57. Passes 35, 36 (held-out probes `cB`, `mt_B`). Cells `c46` to
  `c48`. G18. Budget: 36 of 50 passes, 60 of 300 code runs.
- 23:10. Downloaded glaive (V2-D9). Converted, built `gT`.
  Pass 37. G19.
- 23:12 to 23:23. Edit v3: design, calibration (pass 38), validation
  (passes 39 to 43). G20. Budget: 43 of 50 passes, 71 of 300 code runs.
- 23:23 to 23:30. Edit v4 (thinking-mode "off" term): passes 44 to 47.
  G21. Exported `edits/run2-v4`, built `models/qwen35-wsp4`. Budget: 47 of
  50 passes, 78 of 300 code runs.
- 23:32. BFCL-V4 started: `qwen35-wsp4` (seed 0, port 8301) and
  `qwen35-wsp4-s2` (seed 1, port 8302), both modes, `single_turn`,
  `multi_turn`, `memory`.

## 2026-11-08

- 00:03, 00:55, 00:57, 01:01. The four BFCL v4 generation jobs ended (log
  times). One missing `memory_vector` entry generated again.
- To 01:04. Scored all four. G22. Stopped the BFCL servers.
- 01:04 to 01:13. Run 3 started (V2-D10). Sets, v4 loaded, passes 1 to 5,
  neuron T. G23.
- 01:13 to 01:22. Edit v5 validation (passes 6 to 9): failed. Pass 10 found
  the cause. G23.
- 01:22 to 01:30. Gate direction G_T, neuron T6, edit v6, validation
  (passes 11 to 14). G24. Budget run 3: 14 of 50 passes, 14 of 300 code runs.
- 01:31. Exported `edits/run3-v6`, built `models/qwen35-wsp6`.
- 01:32. BFCL-V4 started: `qwen35-wsp6` (seed 0, port 8301), `qwen35-wsp6-s2`
  (seed 1, port 8302), both modes, `single_turn`, `multi_turn`, `memory`.
- 02:42. The four v6 BFCL generation jobs ended (`__EXIT__`). Memory files
  have 155 entries each. No context errors.
- 02:43 to 02:45. Scored v6. G25.
- 02:45 to 02:50. Cell `c15` (v7), passes 15 to 18, cell `c16`. G26. Budget
  run 3: 18 of 50 passes, 16 of 300 code runs.
- 02:51. Script `tmuxlogs/bfcl7.sh` started: export `edits/run3-v7`, build
  `models/qwen35-wsp7`, BFCL-V4 for `qwen35-wsp7` and `qwen35-wsp7-s2`.
- 02:51 to 03:01. The v7 servers failed to start: the openwsp kernel kept its
  two `vllm serve` processes after the passes (20 GB). Export and build were
  correct (`edit_sha256` 6b456447…). Stopped the kernel servers.
- 03:02. Restarted `tmuxlogs/bfcl7.sh` without the export and build steps.
  BFCL generation started 03:04.
- 03:04 to 03:07. Nothink error analysis (G25 follow-up): most nothink
  failures are text answers with a matching tool. Decision V2-D11. Cell
  `c17` (nothink probe requests). The probe passes failed with CUDA OOM
  while the BFCL servers ran (not counted). Cells `c18`, `c19` written.
- 04:11. The four v7 BFCL jobs ended. 04:12 to 04:15: scored v7 (G27).
- 04:14 to 04:23. Passes 19, 20 (nothink probes). Cells `c18`, `c20`,
  `c21` (neuron N), passes 21 to 26, cells `c19`, `c22`. G28. Budget run 3:
  26 of 50 passes, 23 of 300 code runs. Stopped the kernel servers again.
- 04:23. `tmuxlogs/bfcl8.sh` started: export `edits/run3-v8`, build
  `models/qwen35-wsp8`, BFCL-V4 nothink only (v8 = v7 in thinking mode).
- 04:24. Cell `c23` (edit v9: neuron T8 replaces T7, V2-D11 targets). In
  the store only; not validated yet.
- 04:30 to 10:17. The seed-1 v8 BFCL job hung at 3,925 of 5,017 entries
  (client waited, server idle). Seed 0 ended 04:38. I did not see the hang
  until 10:17. About 5.7 hours lost.
- 10:18. Stopped the hung job. Started it again (`tmuxlogs/bfcl8b.sh`). It
  resumes with the 1,090 missing entries.
- 10:26. The seed-1 v8 rerun ended (5,017 entries). 10:27: scored v8 (G29).
  Started the v9 validation passes 27 to 29.
- 10:28 to 10:31. Passes 27 to 29, cell `c24` (v9 validation). G30. Budget
  run 3: 29 of 50 passes, 25 of 300 code runs.
- 10:32. `tmuxlogs/bfcl9.sh` started: export `edits/run3-v9`, build
  `models/qwen35-wsp9`, BFCL-V4 thinking only (v9 = v8 in nothink).
- 11:23. The v9 BFCL jobs ended (5,017 entries each, no stall). 11:24:
  scored v9 (G31). Keep T7.
- 11:24 to 11:27. Multi-turn failure analysis (G32). The V2-D11 premise was
  wrong; decision V2-D12 withdraws it. The practice data has no rows for the
  self-answer failure, so no new neuron.
- 11:27. Started a third thinking seed (seed 2) for the base and v7
  (`tmuxlogs/bfcl-s3.sh`).
- 12:26. The seed-2 jobs ended (5,017 entries each). 12:27 to 12:28: scored
  (G33). Run 3 summary written. Stopped the servers.
- 12:54. Pass limit for run 3 raised to 100 (V2-D13).
- 12:55 to 13:00. Sets `wA`, `wB` (when2call "ask", `make_sets_w.py`).
  Cells `c25` to `c27`, passes 30 (wA reasoning), 31 (wA probe). Edit v10 =
  v4 + T11. Started validation passes 32 to 35. The user is away for about
  8 hours and gave permission to run the experiments.
- 13:05. Validation done (G34). BFCL-V4 thinking for v10 started (`tmuxlogs/bfcl10.sh`).
- 14:07. BFCL-V4 thinking for v10 ended (4,441 result lines per seed, same as
  v9; no context errors). Stopped the servers. Scored (`tmuxlogs/score10.sh`;
  the first try failed: base-s3 and wsp7-s3 missing from the extra models).
- 14:09. G35: v10 21.99 against v7 22.54. v10 not kept. v7 stays final.
- 14:10. Run 3 concluded. No more edits: each decision-neuron edit trades irrelevance against AST (G35). A choice by BFCL score would break the protocol. Final edit: v7.
- 22:02. Wrote `docs/REPORT.md` (final report of v1 and v2) at the user's request.
