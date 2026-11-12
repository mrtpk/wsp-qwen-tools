# Editor task: weight-space programming for tool use (v2, run 1)

This file is the task brief for the editor. The editor is Claude. The protocol
follows openwsp (`openwsp/docs/TASK.md`, Experiment 8) with a tool-use target.

## Goal

Edit the weights of `Qwen/Qwen3.5-0.8B` (revision
`2fc06364715b967f1860aea9cf38778875588b17`) so that the model uses tools
better, in thinking mode and in non-thinking mode:

1. Call a tool when a supplied tool fits the request.
2. Answer in text when no supplied tool fits, or when the request does not
   give a required value.
3. In a multi-turn agent conversation, call tools until the task is done.
   Then stop and answer.
4. Do not damage other behavior: the reasoning text, the tool-call format,
   and the argument values.

## What the editor can use

- The openwsp kernel through `./wsp run CELL.py`. The kernel has `NAMES`,
  `load`, `load_original`, `save_edit`, `drop_edit`, `edited`, `tok`, `np`,
  and `torch`. The kernel does not run model layers.
- The forward tool through `./wsp forward REQUEST.json`. Request types:
  `generate` (tool-use chats through vLLM) and `teacher_forced` (log
  probabilities, hidden states, MLP inputs, attention). See
  `backend/tooluse_backend.py`.
- The practice data in `practice/`: `when2call.jsonl`, `toolace.jsonl`,
  `apigen_mt.jsonl`. See `practice/convert.py`.
- Knowledge from the v1 documents (`v1/docs/`).

## What the editor cannot use

- BFCL data, BFCL prompts, BFCL results, or any file made from them. This
  includes the v1 activation files and prompt dumps in `v1/wsp/`.
- Gradients, optimizers, or training of any kind.
- Edits of the final norm, the embedding (tied to the output layer), or the
  output layer. Rescaling a whole tensor. The kernel refuses these.

## Budgets

| Item | Budget |
|---|---|
| Forward passes | 50 (one accepted request is one pass; an identical request on the same weights is free; a failed pass is free) |
| Samples in one pass | 512 (`both` counts the two arms) |
| Code runs | 300 |

`./wsp status` shows the use.

## Deliverable

One edit, exported with `./wsp export DIR` (openwsp format v1:
`manifest.json` and `edit.npz`).

## Evaluation (after the deliverable, by the harness)

- BFCL-V4 `single_turn`, `multi_turn`, and `memory`, in both modes, with
  vLLM seeds 0 and 1. The harness and the sampling are the same as in v1, so
  the v1 base and E4b numbers are the reference.
- The changed-weight count, and the KL at generated positions.
- The editor sees the BFCL results only after the export. A later run that
  uses these results is marked "BFCL-informed".
