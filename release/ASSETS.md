# Release assets

These files are too large for git. Download them from the GitHub release of this repository into `release/`, then check them with `(cd release && sha256sum -c ASSETS.sha256)`.

| File | Content | Used by |
| --- | --- | --- |
| `v1-bfcl-result.tar.zst` | All BFCL-V4 generations of the v1 runs (raw model outputs) | `tools/score_all.sh` (level 2) |
| `v2-bfcl-result.tar.zst` | All BFCL-V4 generations of the v2 runs | `tools/score_all.sh` (level 2) |

Absolute paths of the original machine in these archives are replaced by `@ROOT@`. The BFCL prompts in them are from BFCL-V4 (Apache-2.0).
