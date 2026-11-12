---
name: bfcl-local-eval
description: Run and score BFCL-V4 (bfcl-eval 2026.3.23) on a local model served by vLLM, over several seeds and in thinking and non-thinking mode. Use when the user asks to evaluate tool calling of a local or edited model with BFCL, to rescore BFCL results, or to compare BFCL scores of models.
---

# BFCL-V4 on a local vLLM model

Reference scripts: `v1/bfcl/` (v1 harness) and `v2/eval/`
(`serve.sh`, `gen.sh`, `run_bfcl.py`, `run_bfcl_retry.py`). Exact run and score scripts:
`v2/tmuxlogs/bfclN.sh` and `scoreN.sh`.

## Environments

- Main venv (vLLM 0.30.0, torch 2.13.0): `env/requirements-main.txt`.
- BFCL venv (`bfcl-eval==2026.3.23`): `env/requirements-bfcl.txt`. Keep it apart from vLLM.
- vLLM on an RTX 5090 (CUDA 13 wheels) needs `CUDA_HOME` set to the `nvidia/cu13` folder of
  the venv, and `FLASHINFER_EXTRA_LDFLAGS=-L<dir with libcudart.so>`. `serve.sh` sets both.

## Run

1. Run every job in a tmux session. Write the logs to files.
2. Serve one model per seed on its own port:
   `serve.sh MODEL_DIR NAME PORT 0.40 262144 SEED`. Seed 0 → `NAME`, seed 1 → `NAME-s2`, seed 2 → `NAME-s3`.
3. Keep the sum of `--gpu-memory-utilization` on one GPU at or below 0.85.
4. Wait for the server: `until curl -sf localhost:PORT/v1/models; do sleep 10; done`.
5. Register each served name in `QWEN35_EXTRA_MODELS` (comma list). A missing name gives a
   `KeyError` like `qwen35-base-s3-think-FC`, also at score time.
6. Generate both modes per server:
   `gen.sh NAME-think-FC PORT single_turn,multi_turn,memory` and the same with `-nothink-FC`.
7. Sampling: temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, 16,384 output tokens.
8. A job can hang with an idle server. Check that the log grows. Kill and restart the job:
   it resumes and fills the missing entries only.

## Score

```
export BFCL_PROJECT_ROOT=$PWD OPENAI_API_KEY=EMPTY RETRY_LOG=$PWD/retry-log.jsonl
export QWEN35_EXTRA_MODELS=<every served name in result/>
$BFCL_PY run_bfcl_retry.py evaluate --model A-think-FC,A-s2-think-FC --test-category single_turn,multi_turn,memory
```

- `run_bfcl_retry.py` needs `RETRY_LOG`, or it fails with `KeyError 'RETRY_LOG'`.
- The score table is `score/data_overall.csv`.
- Overall weights: non-live 10, live 10, irrelevance 10, multi-turn 30, agentic 40.
  Web search is not run and counts as 0. Say so with every overall score.

## Read the results

- Report the mean over at least two seeds, and each seed.
- The seed spread is about 0.5 to 1.5 overall points. A difference smaller than that is noise.
  Use a third seed before you decide between two close models.
- Look at the error types in `score/*/` (`ast_decoder` and wrong-function counts) per
  category, not only at the overall score.
- A rescore of the same files can change a multi-turn entry (evaluator state, not confirmed).
- vLLM sampling is not bit-exact across machines or batch sizes. Do not expect the same numbers.
