# Weight-space study of Qwen3.5-0.8B tool calling

This folder holds the notes of the study. The study examines how
Qwen3.5-0.8B decides to call a tool, and how a rank-1 weight edit can
improve that decision. Start with `REPORT.md`.

## Documents

- `REPORT.md`: the final report. The recommended edit is E4b.
- `findings.md`: the measured facts (F1 to F49) and the hypotheses (H1 to
  H9). A hypothesis that is not verified has the mark "not confirmed".
- `decisions.md`: the decisions (D1 to D18), with the reason for each one.
- `progress-log.md`: the time log, with the commands, the problems, and the
  fixes.
- `../logs/conversation/conversation-model-edit-2026-11-06.jsonl`: the agent
  conversation transcript, from the start to the question about the use of
  `models-text`.

## Models

| Name | Location | Note |
|---|---|---|
| base | `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17` (local HF cache) | `modelq` is byte-identical |
| wsp | `RikiyaT/Qwen3.5-0.8B-weight-space-programming` at `d1671a178705bcab7644806d23c535517f35723b` | negative reference |
| E1a, E1b | `v1/models/qwen35-e1a`, `qwen35-e1b` | thinking-only fit |
| E3 k = 5 | `v1/models/qwen35-e3k5` | not tested on BFCL |
| **E4b** | `v1/models/qwen35-e4b` | **recommended** |
| E5c | `v1/models/qwen35-e5c` | not better than E4b |

`v1/models/` also has folders from before this study
(`k2-horizon-0.9b`, `qwen3.5-0.8b`, `README.md`). This study did not change
them.

## Scripts

### Run control (`v1/`)

- `tmux_run.sh`: run a command in tmux session `wsp`, with the output in
  `tmuxlogs/<name>.log` (D15).
- `monitor.sh`: print one line when each named tmux job ends.
- `monitor_e4b.sh`: the first monitor (E4b runs only). `monitor.sh`
  replaces it.

### BFCL harness (`v1/bfcl/`)

- `run_bfcl.py`: the bfcl CLI with the Qwen3.5 entries that call vLLM (D2,
  D3). Set `QWEN35_EXTRA_MODELS` for the edited models.
- `serve.sh`: start `vllm serve` with the parsers, `CUDA_HOME`, and an
  optional seed (D5, D17).
- `start_servers.sh`: start several servers in tmux, one after the other.
- `gen.sh`, `wait_and_gen.sh`: BFCL generation (after the server answers).
- `defect.py`: count "text instead of a call".
- `compare_table.py`: comparison tables from `score/data_overall.csv`.
- Results: `result/`, `score/`. Logs: `logs/`. Backups of repaired result
  files: `backup/`.

### Mechanism and edits (`v1/wsp/`)

| Script | Purpose | Findings |
|---|---|---|
| `dump_prompts.py` | BFCL single-turn prompts and outcomes to JSONL (BFCL venv) | F6, F13 |
| `probe_first_token.py` | non-thinking decision token: `p_call` and residual stream | F6 |
| `analyze_layers.py`, `readout_ceiling.py` | layer probes, readout gap | F7 |
| `wsp_geometry.py` | wsp edit directions against the decision | F8 |
| `make_swap.py`, `analyze_swap.py`, `prior_direction.py` | tool-swap pairs, query prior | F9 to F11 |
| `probe_post_think.py`, `analyze_post_think.py` | thinking decision token | F13 |
| `conclusion_test.py` | explicit conclusion in the reasoning | F17 |
| `probe_sampling_bias.py` | presence-penalty effect | F15 |
| `attribute_post_think.py`, `analyze_attribution.py` | direct logit attribution; row selection for later steps | F18 |
| `heads_and_patch.py`, `analyze_heads.py` | head attribution, attention by segment, mean patching | F19 |
| `collect_sites.py`, `analyze_sites.py` | activations at the candidate sites | F21, F29, F32 |
| `edit_e1.py` | E1 and E2 (sweep, save) | F22, F26 |
| `e3_read.py`, `edit_e3.py` | E3 | F26, F28 to F30 |
| `segment_kl.py` | KL by segment | F30 |
| `edit_e4.py` | E4 (read, calibrate, save) | F32, F33 |
| `swap_gen.py` | tool-swap test through vLLM | F27, F41 |
| `dump_mt_decisions.py`, `mt_reads.py` | multi-turn decision points | F44, F45 |
| `edit_e5.py` | E5 (read, calibrate, save) | F45, F46 |

## Disk use

`v1/wsp` 1.7 GB (activation files), `v1/bfcl` 7.4 GB (venv,
results), `v1/models` 12 GB (five edited checkpoints of 1.75 GB, and
the older folders), `v1/modelq` 1.7 GB (the decoded `modelq`, the same
bytes as base).
