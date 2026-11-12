---
name: repro-package
description: Build a small, tested folder that lets another person or agent reproduce a study report (scripts, edits, results, pinned environments). Use when the user asks for a reproduction package, a "reproducible folder", an archive of an experiment, or asks to save space before deleting experiment folders.
---

# Build a reproduction package

The goal: another machine reproduces every number of the report from a
small folder. Rebuild or download large files. Do not copy them.

## Rules

- Do not change or delete the original folders. Copy only.
- Never write tokens or credentials into the package.
- Test every reproduction level that you claim. Mark the rest "not tested".
- Tell the user what the package leaves out and what that loses.

## Steps

1. Read the report. List each number and each claim, and the file it comes from.
2. Measure the folders: `du -sh */ | sort -h`. Find the large items.
3. Give each large item one class:
   - **keep**: small, or not rebuildable (ledgers, kernel logs, score tables, edits).
   - **rebuild**: a script makes it exactly from kept data (checkpoints, score files).
   - **download**: public data at a pinned revision (base model, datasets).
   - **drop**: not needed for the report. Record the loss in the README.
4. Copy with `rsync -a` and explicit `--exclude` lines. Keep the exclude list in the README.
5. Store model edits as changed tensors only. Use `extract_edit.py BASE EDITED OUT.safetensors`.
   It writes a JSON with the shape, the count of changed values, and a SHA-256 for each tensor.
   `apply_edit.py BASE EDIT.safetensors OUT_DIR` rebuilds the model and checks each hash.
6. Compress raw generations with `tar | zstd -19`. JSON results compress to about 10%.
7. Pin everything:
   - Hub models and datasets at full commit hashes. Write a download script that checks SHA-256.
   - Python environments: `uv pip freeze` into `env/requirements-*.txt`. Remove `-e file:` lines.
   - Record the Python version, the GPU, the driver, and the CUDA wheels.
8. Fix the absolute paths. Write `tools/relocate.sh` that rewrites the old root and the old
   model path. Do not run it on `results/` or on itself.
9. Fix scripts that need the original layout. Example: `uv run` needs a `pyproject.toml`;
   call `.venv/bin/<tool>` instead.
10. Scan for credentials in the copy and inside every archive:
    `grep -rIl -E 'hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|ghp_|glpat-|TOKEN=|API_KEY=' .`
    and `zstd -dc X.tar.zst | tar -xO | grep -c ...`. Agent event logs often hold environment
    dumps. Drop them and keep summaries made from the originals.
11. Write the README: contents table, exclusions with the reason and the loss, setup,
    reproduction levels from cheap to expensive, checks.
12. Test in a scratch copy, not in the package. Link to the existing venvs only if you say so in
    the reply; a clean install is then "not tested".
    - Level 1: print the report tables from the saved score files. Compare with the report.
    - Level 2: rescore the saved generations. Report every difference.
    - Level 3: rebuild each edited model. Compare the tensors bit for bit.
    - Level 5: rerun the CPU edit cells. Compare the exported edit hash.
13. Write `MANIFEST.sha256` last: `find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256`.
    Run `sha256sum -c` once.
14. If asked, make the archive in tmux: `zip -r -q -y NAME.zip NAME`, then `unzip -tq NAME.zip`.

## Defects found the first time (check each one)

- Scoring needed an env var (`RETRY_LOG`) that only the original shell had.
- A CPU rerun needed the edit store (`work*/edited`). The first exclude list dropped it.
- A summary script needed `stderr.log`, which a `*.log` exclude dropped. Run summaries on the originals.
- A small run log (`retry-log.jsonl`) was dropped by mistake. Check each exclude against the report.
- A rescore changed one multi-turn entry with no file change. The evaluator probably keeps state.
  Report such differences. Do not hide them.
- `rm -rf` on the scratch test copy can be blocked. Tell the user the path and the size.

## Final reply

Give: path and size, the tested levels with results, each difference, each excluded item with
its loss, and what is not tested (clean install, downloads, GPU levels).
