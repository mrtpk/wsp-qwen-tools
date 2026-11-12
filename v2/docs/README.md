# Weight-space programming for tool use in Qwen3.5-0.8B (v2)

v2 runs the openwsp weight-space programming protocol on a tool-use goal.
The editor is Claude. v1 (`v1/`) is the reference.

## Documents

- `REPORT.md`: the final report of v1 and v2 (verdict, results, lessons, limits).
- `TASK.md`: the editor brief (goal, allowed inputs, budgets, deliverable).
- `findings.md`: the measured facts (run 1: G1 to G12; run 2: G13 to G22;
  run 3: G23 to G35; Codex: G14; retry diagnostic: G15).
- `decisions.md`: the decisions (V2-D1 to V2-D13).
- `progress-log.md`: the time log.

## Layout

| Path | Content |
|---|---|
| `wsp` | The editor command: `run`, `forward`, `diff`, `export`, `status`. |
| `backend/` | `tooluse_backend.py` (local GPU forward backend), `wsp.py` (openwsp SDK entry). |
| `openwsp/` | openwsp at commit `8d09276`, not changed. |
| `practice/` | Practice data and `convert.py`. `raw/` holds the downloaded files. |
| `cells/run1/`, `cells/run2/`, `cells/run3/` | Every kernel cell and request builder of each run, in order. |
| `work/` | openwsp work directory of runs 1 and 2: `passes.jsonl` (pass ledger), `kernel-log.jsonl` (every cell and its output), `forward/` (pass outputs), `edited/` (submitted tensors). |
| `work3/` | openwsp work directory of run 3 (own budget). `imports/` holds copied run-2 probe files. |
| `edits/` | Exported edits: `run1-n1488` (run 1, failed), `run2-v4` (neuron 1488), `run3-v6` (neurons 1488 and 1152). |
| `models/` | Full checkpoints built from the edits: `qwen35-wsp2` (run 1), `qwen35-wsp4` (v4), `qwen35-wsp6` (v6). |
| `eval/` | BFCL harness (copied from v1), leak filter, checkpoint builder, retry diagnostic (`run_bfcl_retry.py`), results. |
| `codex/` | Codex test (G14): Docker image, task suite, runner, results. |

## Commands

- Editor: `./wsp status`, `./wsp run cells/run1/<cell>.py`,
  `./wsp forward <request.json>`, `./wsp export <dir>`. Run 3: set
  `OPENWSP_WORK=$PWD/work3` first.
- Checkpoint: `PYTHONPATH=openwsp/src ../.venv/bin/python
  eval/build_checkpoint.py <base-dir> edits/run1-n1488 models/qwen35-wsp2`.
- BFCL: `eval/serve.sh <model-dir> <name> <port> 0.40 262144 <seed>`, then
  `QWEN35_EXTRA_MODELS=<name> eval/gen.sh <name>-think-FC <port>
  single_turn,multi_turn,memory`, then `eval/score.sh`.
