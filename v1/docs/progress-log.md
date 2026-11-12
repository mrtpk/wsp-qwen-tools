# Progress log

All times are UTC+4.

## 2026-11-06 14:13. Decode `modelq`

- Command: `base64 -d models-text/modelq.base64 > v1/modelq/model.safetensors`.
- Result: a `safetensors` file of 1,746,942,600 bytes, with 488 tensors.
- The SHA-256 matches the official `Qwen/Qwen3.5-0.8B` file. See F1.
- A statistics scan found outliers only in the vision encoder (`norm1`,
  `norm2`, `mlp.linear_fc2.bias`, blocks 0 and 8 to 11).

## 2026-11-06 14:20. Research on weight-space programming

- Source: "Sparks of Weight-Space Programming", Takehi, Teoh, Zhang, Khattab,
  MIT CSAIL, 2026-11-02. <https://rikiyat.github.io/blog/weight-space-programming/>
- Experiment 8 removes one loop direction from 64 output columns of
  `down_proj` in layers 15 and 19 (131,072 weights). MATH-500 went from 7.9%
  to 48.1%.
- Replication: <https://github.com/openwsp/openwsp>.

## 2026-11-06 14:25. Baseline facts for BFCL-V4

- The model card reports BFCL-V4 25.3 (thinking) for Qwen3.5-0.8B.
- Third-party leaderboards (not confirmed): Qwen3.5-2B 43.6, Qwen3.5-4B 50.3,
  Qwen3.5-9B 66.1.
- The BFCL-V4 overall weights are non-live 10, live 10, irrelevance 10,
  multi-turn 30, agentic 40. Source: `bfcl_eval/eval_checker/eval_runner_helper.py`
  lines 509 to 518, in the installed package.

## 2026-11-06 14:30. Harness setup

- Created `v1/bfcl/.venv` with `bfcl-eval==2026.3.23` and `soundfile`.
- Wrote `v1/bfcl/run_bfcl.py` (D2, D3).
- Downloaded base and wsp with `snapshot_download`.
- A diff of all tensors confirmed the wsp edit. See F3.

## 2026-11-06 14:36. vLLM startup failure and fix

- Problem: `RuntimeError: FlashInfer requires GPUs with sm75 or higher`. The
  log also showed "SM 12.x requires CUDA >= 12.9".
- Cause: FlashInfer used the system `nvcc`.
- Fix: D5. The root `README.md` documents this fix.
- Servers: base on port 8101 and wsp on port 8102. Each uses
  `--gpu-memory-utilization 0.44` and `--max-model-len 65536`.
- Smoke test: one `get_weather` call. Both thinking modes gave a correct tool
  call through `qwen3_coder`.

## 2026-11-06 14:42. BFCL generation

- Command: `v1/bfcl/gen.sh <model> <port> single_turn,multi_turn,memory`.
- The script uses 48 threads. There are 5,017 entries for each run.
- Four runs: base and wsp, each with thinking on and thinking off.
- Logs: `v1/bfcl/logs/gen-*.log`.

## 2026-11-06 15:25. First scores

- base, non-thinking: complete. Overall 17.56. See F4.
- Problem: The display name had a comma, and the CSV columns moved by one
  place. Fix: remove the comma from `display_name` in `run_bfcl.py`, then
  score again.
- Found the "text instead of a tool call" defect. See F5.

## 2026-11-06 15:35. wsp, non-thinking

- Scored. Overall 15.66. The defect is worse in wsp. See F4 and F5.
- The thinking runs are at about 85%.

## 2026-11-06 15:40. Start of the weight study

- Plan:
  1. Rebuild the exact non-thinking prompts. Check the token counts against
     `input_token_count` in the BFCL results.
  2. Measure the first-token logits: `<tool_call>` compared with the best
     text token.
  3. Find the layers and components that write this decision (logit lens and
     direct logit attribution).
  4. Compute the mean-difference direction between the "call" prompts and the
     "text" prompts at each layer.
  5. Compare this direction with the wsp edit direction.

## 2026-11-06 15:45. Prompt rebuild

- Script: `v1/wsp/dump_prompts.py`. It runs in the BFCL venv. It uses
  `load_dataset_entry` (with the language hint) and `convert_to_tool` with
  `ModelStyle.OPENAI_COMPLETIONS`, the same as the BFCL generation.
- Output: `prompts-base-nothink.jsonl`, `prompts-wsp-nothink.jsonl` (3,625
  rows each).
- Check: `apply_chat_template(..., enable_thinking=False)` gives the same
  token count as vLLM for all 3,625 rows.

## 2026-11-06 15:50. GPU out of memory, and an error that I made

- Problem: The HF probe ran out of memory. The two vLLM servers used 29 GB.
- Error: I read an empty progress line in `gen-base-think.log` as "the run is
  complete". I stopped the base server while the base thinking run was
  active. The run was at 4,423 of 5,017 entries.
- Recovery: BFCL writes each entry when it finishes. I stopped the run. A
  restart skips the entries that are already in the result files.
- Rule for the next time: check that the `run_bfcl.py generate` process is
  gone before you stop a server.

## 2026-11-06 15:52. A context-length defect in the harness

- 26 entries in the non-thinking runs (17 base, 9 wsp) and 21 entries in the
  thinking runs have "Error during inference: 400 ... maximum context length
  is 65536 tokens". All are in multi-turn long-context or memory.
- Cause: `max_tokens=16384` plus a long history is more than
  `--max-model-len 65536`.
- Effect: BFCL counts these as failures. The model is near 0% in these
  categories, so the effect on the scores is small.
- Plan: generate these entries again with `--max-model-len 262144`.

## 2026-11-06 16:00. First-token probe and layer analysis

- `probe_first_token.py` on base: 3,625 prompts in 230 s on the GPU.
- `analyze_layers.py` and `readout_ceiling.py`: see F6 and F7.
- `probe_first_token.py` on wsp, then `wsp_geometry.py`: see F8.

## 2026-11-06 16:15. Tool-swap test

- `make_swap.py` made 858 pairs (1,716 prompts). The probe took 70 s.
- `analyze_swap.py`: see F9.
- An inline analysis of the logits: see F10. The result is in `swap-prior.txt`.
- Decision D8: the next target is the query prior, not the relevance signal.

## 2026-11-06 16:00. Query-prior direction

- `prior_direction.py`: see F11. The output is in `prior-base.txt`.

## 2026-11-06 16:05. Base server restart and resume

- Restarted the base server with `serve.sh ... 262144` (D9). Log:
  `logs/vllm-base-262k.log`.
- Resumed the base thinking run. Log: `logs/gen-base-think-resume.log`.
- The wsp thinking run completed with 5,017 entries.

## 2026-11-06 16:20. All four runs scored

- Scored base thinking and wsp thinking. See F12 for the comparison table.
- The 10 to 17 context-length errors in each run remain. Generate them again
  later with the 262,144 context.

## 2026-11-06 16:30. Thinking-mode experiments start (D11)

- Stopped the idle wsp server to free GPU memory. No BFCL run was active.
- `dump_prompts.py` now also writes `output_token_count`,
  `reasoning_content`, and `result`. Made `prompts-base-think.jsonl` and
  `prompts-wsp-think.jsonl`.
- Checked the rebuild of the thinking sequence. See F13.
- Sampled 36 `value_error:string` cases. See F16.
- `probe_post_think.py` on base: 3,625 rows in 200 s. See F13.
- Over-call origin and the reasoning-action table: see F14.
- The presence-penalty test: see F15. I checked the vLLM penalty code and
  the sampler order in the pinned source before I trusted the script.
- The explicit-conclusion test: 925 rows × 3 variants in 150 s. See F17.
- 16:50. The user said: do the experiments without asking for permission,
  and record them.
- 16:55. `attribute_post_think.py` (1,924 rows, 5 min). See F18.
- 17:00. Removed the context-error entries from the base results (10 for
  thinking, 17 for non-thinking). Backups are in `v1/bfcl/backup/`.
  Generated the base thinking entries again on the 262,144 server. The base
  non-thinking entries are not done yet.
- Limit: 2 of the removed entries were memory "prereq" entries. Their
  dependent entries ran before the repair, so their memory state can
  disagree.
- 17:05. `heads_and_patch.py` failed at row 500 (no `<tools>` block). Fixed
  and ran again: 900 rows in 10 min. See F19.
- 17:20. Separability among the called rows. See F20.
- 17:25. `collect_sites.py` (1,924 rows) and `analyze_sites.py`. See F21.
- 17:35. The base non-thinking repair ran (17 entries) on the 262,144
  server.
- 17:40. E1 sweep, first try: the plain `<tool_call>` readout as u, k = 20 to
  120. At k = 20, the background KL was 1.2 nats. Stopped. Changed u to the
  least-squares direction and the grid to k = 2 to 20. See D12.
- 18:00. E1 sweep complete. See F22. Saved E1a and E1b. Stopped the base
  server. Started `qwen35-e1a` (port 8103) and `qwen35-e1b` (port 8104), each
  with `--gpu-memory-utilization 0.44` and `--max-model-len 262144`.
- Note: `v1/models/` existed before this study. The new folders are
  `qwen35-e1a` and `qwen35-e1b`. Nothing else in that folder changed.
- 18:10. The E1b server failed (KV cache 2.97 GiB < 3.04 GiB needed for
  262,144 tokens), because it started while the E1a server took memory.
  Started it again with `--gpu-memory-utilization 0.50`.
- 18:15 to 19:20. BFCL thinking runs for E1a and E1b (5,017 entries each,
  0 errors).
- 18:55. Early single-turn scoring. See F23. Prepared E2 (D13).
- 19:20. The train/held-out split of the irrelevance gain. See F24.
- 19:25. Full scoring. See F25. Base thinking was scored again after the
  repair.
- 19:30. Stopped the E1a server. Started the E2 sweep (`EDIT_VARIANT=e2`).
  Started `swap_gen.py` on E1b: the 1,716 tool-swap prompts in thinking mode
  (out-of-distribution test). The swapped-tool half is new to the fit.
- 19:45. Base swap baseline. See F27. Stopped the base and E1b servers.
- 19:50. E3 sweep, first attempt. At k = 5, the KL was 13 times larger than
  for E1b. Stopped. See F28.
- 20:05. Wrote the draft report `REPORT.md` with an E3 placeholder (user
  request).
- 20:10. E3, second attempt: `collect_sites.py` with `BG_MODE=all`. The
  background now covers the whole sequence (system prompt, user text,
  reasoning) and 4 positions inside the model's own first tool call. Output:
  `sites-base-allbg.pt`.
- 20:12. E3 read vectors with the full background. See F29.
- 20:15 to 20:25. E3 sweep (γ = 30), the E3 k = 5 checkpoint
  (`v1/models/qwen35-e3k5`), and `segment_kl.py`. See F30. E3 is not
  better than E1b. Updated `REPORT.md`.
- 20:27. Started the E1b server again (port 8104) and the non-thinking BFCL
  run for E1b (`wait_and_gen.sh` waits for the server). This is the
  side-effect check of D11.
- 20:26. The user asked if a disconnect stops the jobs. The `nohup` jobs
  were in a session of the agent shell, so a cleanup of that shell could
  kill them. Stopped the E1b server and the E1b non-thinking run at about
  2,872 of 5,017 entries. Started both again in `tmux` session `wsp`
  (window `server`, window `bfcl`). Logs: `logs/vllm-e1b-3.log`,
  `logs/gen-e1b-nothink-2.log`. BFCL resumed with 2,145 entries left.
- Rule: start long jobs in `tmux`, not with `nohup` in the agent shell.
## 2026-11-07

- 01:57. The user returned and asked for all commands in tmux (D15). Wrote
  `v1/tmux_run.sh`. The first version broke on single quotes. Fixed.
- 01:58. The E1b non-thinking run was complete (0 errors). Scored it and
  base non-thinking again. E1b fails the side-effect check. See F31.
- 02:00. `dump_prompts.py` to `prompts-base-nothink-v2.jsonl`.
  `collect_sites.py` with `MODE=nothink BG_MODE=all` (3,625 rows).
- 02:04. `edit_e4.py read`. See F32.
- 02:05 to 02:15. `edit_e4.py calibrate` (k = 2, 5, 8). See F33.
- 02:17. Saved E4b. Started in tmux: `e4b-server` (port 8105), `e4b-think`,
  and `e4b-nothink`. Both BFCL runs use the same server.
- 03:47. Scored E4b thinking. See F35. Updated `REPORT.md`. Stopped the
  E4b server.
- 08:50. The user said: continue without asking for permission.
- 08:55. Second BFCL sample in thinking mode (D17). `serve.sh` has a new
  optional seed argument. The first sample used the vLLM default seed 0.
  The second sample uses seed 1. Started in tmux: `start-s2`
  (`start_servers.sh` brings up `qwen35-base-s2` on 8111, `qwen35-e1b-s2` on
  8112, and `qwen35-e4b-s2` on 8113, one after the other, 0.30 of the GPU
  each), `s2-base-think`, `s2-e1b-think`, `s2-e4b-think`, and `monitor-s2`
  (`v1/monitor.sh`).
- 09:00. `e4b-live-failure-types.txt`. See F36.
- 09:54. `s2-base-think` complete. The E4b-s2 server had failed at 08:54
  with CUDA out of memory, and `s2-e4b-think` was still waiting. Stopped the
  base-s2 server, started the E4b-s2 server again. Scored base s2. See F37.
- Rule: with several vLLM servers, keep the sum of the fractions at or
  below 0.85, because each server also uses about 1 GB outside its fraction.
- 09:58. `s2-e1b-think` complete. Scored. Stopped its server. See F38.
- 10:52. `s2-e4b-think` complete. Scored. The first table script had an
  f-string quoting error. Moved it to `bfcl/two_sample_table.py`. See F39.
  Updated `REPORT.md`. Stopped the E4b-s2 server.
- 10:58. Tool-swap tests: `swap_gen.py` has a new mode argument. Started in
  tmux: `start-swap` (base on 8101, E4b on 8105, 0.40 each), `swap-e4b`
  (thinking, then non-thinking), `swap-base-nothink`, and `monitor-swap`.
- 10:57 to 11:00. Swap tests complete. See F41. Stopped both servers.
  Analysis of the E4b memory gap. See F40. Corrected F39 (the memory gap is
  not confirmed).
- 11:02. Second non-thinking sample (seed 1) for base and E4b. Started in
  tmux: `start-s2nt` (base-s2 on 8111, E4b-s2 on 8113, 0.40 each),
  `s2-base-nothink`, `s2-e4b-nothink`, and `monitor-s2nt`.
- 11:10. `multiturn-compare.txt`. See F42.
- 11:23. `s2-base-nothink` complete, 0 errors. Scored: overall 17.74 (s1
  17.63), non-live AST 35.31 (32.65), live AST 43.60 (44.19), irrelevance
  77.33 (78.11), "text instead of a call" 699 (685). Stopped its server.
  `s2-e4b-nothink` is still running.
- 12:09. `s2-e4b-nothink` complete (10 context errors). Scored. See F43.
  Updated `REPORT.md`. Stopped the E4b-s2 server.
- 12:15 to 12:30. H9 test, part 1: `dump_mt_decisions.py` rebuilds the
  step-level decision points of the base thinking `multi_turn_base` traces
  (3,132 rows). Three problems and fixes:
  1. `inference_log` alternates turn records and state lists. Keep only the
     dicts with `begin_of_turn_query`.
  2. The Qwen3.5 template needs the tool-call arguments as a mapping. Parse
     the JSON string, as vLLM does.
  3. The rebuilt prompts were 650 to 1,800 tokens too long. vLLM keeps only
     `name`, `description`, and `parameters` of each tool and drops
     `response`. After the fix, 2,757 of 3,132 rows match the vLLM token
     count exactly. The other rows are a few tokens shorter.
- 12:32. H9 test, part 2: `mt_reads.py` on 1,000 random exact-match rows, in
  tmux (`mt-reads`, `monitor-mt`).
- 12:17. `mt-reads` (1,000 rows) complete. See F44.
- 12:20 to 13:10. `mt-reads2` on all 2,757 rows, with the activations
  saved. Wrote `edit_e5.py`.
- 13:12. E5 read vectors, first design (one mid proxy): wrong. Second
  design (two mid proxies). See F45.
- 13:15. E5 calibration at k = 5 (base, E4, and three E5 targets) in tmux
  (`e5-cal`, `monitor-e5`).
- 13:19. E5 calibration complete. See F46. Saved E5c. Started in tmux:
  `srv-e5c` (port 8106, 0.50), `e5c-think`, `e5c-nothink`, `monitor-e5c`.
- 13:55. `e5c-nothink` complete (4 context errors). Scored. The inline
  table code had the f-string quoting error again. Wrote
  `bfcl/compare_table.py` for all later tables. See F47.
- 14:14. `e5c-think` complete. The evaluation failed: one category had 199
  of 200 entries. `gen.sh` again generated the one missing entry. Scored.
  See F48. Stopped the E5c server. Updated `REPORT.md`.
- 14:20. `mt-e1b-gain.txt`. See F49.
- 14:25. The user chose to finalize. Wrote the final `REPORT.md` and the
  script index in `README.md`. Removed `bfcl/two_sample_table.py`
  (`compare_table.py` replaces it) and 26 trivial `tmuxlogs/check*` files.
  No checkpoint or result was deleted. No job is running.
- 14:50. Copied the agent transcript (`agent-transcripts/0c393f55-…/0c393f55-….jsonl`)
  to `v1/logs/conversation/conversation-model-edit-2026-11-06.jsonl`
  (user request). The copy has 483 lines (781,191 bytes) and ends with the
  question about the use of `models-text`. A scan for credential patterns
  (HF, OpenAI, GitHub, Slack, AWS keys, private keys, `*_API_KEY=` values)
  found none.
- HF Transformers uses the slow PyTorch fallback for `causal_conv1d` and
  `chunk_gated_delta_rule`, because `causal_conv1d` and
  `flash-linear-attention` are not installed. The results are correct. The
  speed is enough for now.
