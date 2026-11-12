# wsp-qwen-tools: weight edits for tool use in Qwen3.5-0.8B

This repository holds two weight-space programming studies on `Qwen/Qwen3.5-0.8B` and the BFCL-V4 tool-use benchmark. Weight-space programming means that you change a trained network by reading and writing its weights directly, based on what each part computes.

- **v1** (agent: Cursor): a rank-1 edit of `layers.23.mlp.down_proj`, fitted on BFCL prompts and activations. Recommended edit: E4b.
- **v2** (agent: Claude, with the OpenWSP protocol): "neuron programming" in the layer-23 MLP, fitted on practice data only. Recommended edit: v7, two neurons, 6,144 weights.

The v7 model is on Hugging Face: [`mrtpk/Qwen3.5-0.8B-wsp-tool-use`](https://huggingface.co/mrtpk/Qwen3.5-0.8B-wsp-tool-use) (branch `v7`).

Start with the report: [`v2/docs/REPORT.md`](v2/docs/REPORT.md) (both studies). Write-up: [Weight-Space Programming - Teaching Qwen3.5-0.8B when to call a tool](https://github.com/mrtpk/kaizen/tree/master/learnings/20261112_wsp_qwen_tool_use) (kaizen blog).

## Results (BFCL-V4 overall, web search not run and counted as 0)

| Model | thinking | non-thinking | Changed weights |
|---|---|---|---|
| base | 20.84 (3 seeds) | 17.68 (2 seeds) | 0 |
| v1 E4b | 21.49 | 20.29 | rank 1 on a 1,024 x 3,584 tensor |
| v2 v7 | 22.54 (3 seeds) | 19.55 | 6,144 |

Neither edit is better in both modes. Both results are optimistic: v1 was fitted on BFCL, and BFCL scores chose v7 from four candidates. The seed spread is 0.5 to 1.5 points. See the report, sections 1 and 9.

## Contents

| Path | Content |
|---|---|
| `v2/docs/REPORT.md` | The final report of both studies. Start here. |
| `v1/docs/` | v1 documents: report, findings F1 to F49, decisions D1 to D18, time log. |
| `v2/docs/` | v2 documents: findings G1 to G35, decisions V2-D1 to V2-D13, time log, `TASK.md` (the protocol). |
| `v1/wsp/` | v1 scripts (`edit_e4.py`, `collect_sites.py`, ...) and their small outputs. |
| `v1/bfcl/` | The BFCL-V4 harness of both studies: `run_bfcl.py`, `serve.sh`, `gen.sh`. |
| `v1/edits/` | The v1 edits as changed tensors (E1a, E1b, E3k5, E4b, E5c) and the RikiyaT edit. |
| `v2/backend/`, `v2/wsp` | The v2 editor command and the local forward backend. |
| `v2/cells/run1..3/` | Every v2 kernel cell, in order. |
| `v2/work/`, `v2/work3/` | Pass ledgers (`passes.jsonl`), kernel logs (`kernel-log.jsonl`: every cell and its output), designs. |
| `v2/edits/` | All v2 exported edits (OpenWSP format v1). Final edit: `run3-v7`. |
| `v2/eval/` | Checkpoint builder, retry harness, leak filter list. |
| `v2/tmuxlogs/` | The exact BFCL run and score scripts of v2. |
| `v2/codex/` | The Codex CLI test: Dockerfile, tasks, runner, summaries. |
| `results/v1-score-csv/`, `results/v2-score-csv/` | The score tables of every report run. |
| `analysis/report_tables.py` | Prints the report's BFCL tables from the score tables. |
| `analysis/error_counts.py` | Error-type counts. Needs a rescore (level 2). |
| `tools/` | `relocate.sh`, `get_openwsp.sh`, `get_practice.sh`, `apply_edit.py`, `extract_edit.py`, `score_all.sh`, hash lists. |
| `env/` | Exact package lists of the two Python environments. |
| `skills/` | Claude Code skills from this work. Copy a skill folder to `~/.claude/skills/` to use it. |
| `release/ASSETS.md` | Large files that are GitHub release assets (the raw BFCL generations). |
| `hf/MODEL_CARD.md`, `tools/upload_hf.py` | The Hugging Face model card of v7, and the script that checks the built v7 against the base and uploads it. |

Not in this repository:

- Model checkpoints. `tools/apply_edit.py` and `v2/eval/build_checkpoint.py` rebuild them from the base model and the edits. The v7 checkpoint is also on Hugging Face (see above).
- OpenWSP. `tools/get_openwsp.sh` fetches it at the commit that v2 used.
- The practice data. `tools/get_practice.sh` downloads it at pinned revisions and checks the hashes.
- The v2 practice subsets, request files and pass outputs (the files that hold practice-data text). They are withheld until the dataset licences are checked. Without them, v2 levels 5 and the run-1/2 passes need new passes on the GPU.
- v1 activation dumps and BFCL prompt dumps. The v1 scripts rebuild them from the BFCL generations.
- Raw Codex event logs. They hold environment dumps of the original machine.

## Setup

### Base model

`Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`:

```
huggingface-cli download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17
```

The RikiyaT edit is on the Hub too: `RikiyaT/Qwen3.5-0.8B-weight-space-programming`, revision `d1671a178705bcab7644806d23c535517f35723b`. `v1/edits/wsp-rikiyat.*` holds the same changed tensors.

### Paths

The scripts and the ledgers hold two placeholders: `@ROOT@` (this repository) and `@BASE@` (the base model snapshot). Run this once after you clone:

```
tools/relocate.sh /path/to/Qwen3.5-0.8B-snapshot
tools/get_openwsp.sh
```

### Python environments

Both use Python 3.12. The original machine had one RTX 5090 (CUDA 13 wheels).

1. Main environment at `.venv` (torch 2.13.0, transformers 5.17.0, vllm 0.30.0):
   `uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r env/requirements-main.txt`
2. BFCL environment at `v1/bfcl/.venv` (`bfcl-eval==2026.3.23`):
   `uv venv v1/bfcl/.venv --python 3.12 && uv pip install --python v1/bfcl/.venv/bin/python -r env/requirements-bfcl.txt`
3. vLLM on an RTX 5090 needs `CUDA_HOME` and `FLASHINFER_EXTRA_LDFLAGS`. The serve scripts set both. They expect a link `.cudart/libcudart.so` to the CUDA 13 runtime:
   `mkdir -p .cudart && ln -s "$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13" .cudart/libcudart.so`

## Reproduction levels

### Level 1: the report tables (seconds, no GPU)

```
python3 analysis/report_tables.py
```

This prints every BFCL-V4 number of the report (two-seed means, the seeds, the three-seed means) from the saved score tables.

### Level 2: score the saved generations again (minutes, no GPU)

Download the release assets (`release/ASSETS.md`), then:

```
tools/score_all.sh
python3 analysis/report_tables.py v1/bfcl/score v2/eval/score
python3 analysis/error_counts.py
```

Tested on the original machine: all 42 runs give the level-1 numbers, with one exception. In v4 thinking seed 0, entry `multi_turn_base_11` passed in the original scoring and fails in the rescore (overall 20.93 to 20.89). Expect differences of this size.

### Level 3: rebuild the edited models (minutes, no GPU)

- v7 without a build: download `mrtpk/Qwen3.5-0.8B-wsp-tool-use`, revision `v7`. It is the output of the command below; `tools/upload_hf.py` checked that only the 3 edit tensors differ from the base and that their SHA-256 match `v2/edits/run3-v7/manifest.json`.
- v2 edits (exact, with hash checks):
  `PYTHONPATH=v2/openwsp/src .venv/bin/python v2/eval/build_checkpoint.py BASE_DIR v2/edits/run3-v7 models/qwen35-wsp7`
- v1 edits (exact; each changed tensor is checked against its SHA-256):
  `.venv/bin/python tools/apply_edit.py BASE_DIR v1/edits/e4b.safetensors models/qwen35-e4b`
- Edit to model names: `run1-n1488` is `qwen35-wsp2`, `run2-v4` is `qwen35-wsp4`, `run3-v6` to `run3-v10` are `qwen35-wsp6` to `qwen35-wsp10`.
- `v1/edits/*.json` gives the changed-value count, the relative norm and the first two singular values of each change.

### Level 4: run BFCL-V4 again (about 1 hour per model and mode, GPU)

```
v2/eval/serve.sh models/qwen35-wsp7 qwen35-wsp7 8301 0.40 262144 0
QWEN35_EXTRA_MODELS=qwen35-wsp7 v2/eval/gen.sh qwen35-wsp7-think-FC 8301 single_turn,multi_turn,memory
```

- Use seed 1 (`-s2` names) and seed 2 (`-s3`) for the other seeds. The exact v2 scripts are `v2/tmuxlogs/bfclN.sh`.
- Sampling: temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, 16,384 output tokens. Web search is not run.
- vLLM sampling is not bit-exact across machines and batch sizes. Expect the seed spread (about 0.5 to 1.5 overall), not the same numbers.
- Keep the sum of vLLM memory fractions on one GPU at or below 0.85.
- A generation job can hang with an idle server. Check that the log changes. Restart the job: it resumes with the missing entries.

### Level 5: the v2 edit cells

- The cells need the withheld practice subsets and pass outputs (see above), or new passes: `tools/get_practice.sh`, then run the cells of `v2/cells/run1..3/` in order with `v2/wsp run <cell>`. Forward passes (`v2/wsp forward REQUEST.json`) need the GPU.
- `v2/work3/kernel-log.jsonl` holds the original output of each cell for comparison.
- The protocol rules are in `v2/docs/TASK.md`.

### Level 6: v1 mechanism scripts

The v1 scripts in `v1/wsp/` read prompt dumps and activations made from the BFCL generations. The order of the commands is in `v1/docs/REPORT.md`, section 9.

## Licences of third-party material

- `Qwen/Qwen3.5-0.8B`: see its model card. The edits in `v1/edits/` and `v2/edits/` are changes of its weights.
- OpenWSP: MIT (fetched, not included).
- The RikiyaT edit: see `RikiyaT/Qwen3.5-0.8B-weight-space-programming` on the Hub.
- BFCL-V4 prompts in the release archives and in `v2/eval/bfcl-user-messages.json`: see the BFCL repository.
- Practice data (When2Call, ToolACE, APIGen-MT-5k, glaive-function-calling-v2): downloaded by `tools/get_practice.sh`; see each dataset card.

## Reference

- [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming), Takehi, Teoh, Zhang and Khattab, MIT CSAIL.
- [OpenWSP](https://github.com/openwsp/openwsp).
