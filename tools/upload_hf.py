"""Check the built v7 checkpoint against the base model, stage it with the model card, and upload it
to a Hugging Face model repository (branch `main`, then branch `v7` from the same commit).

Build the checkpoint first (README, level 3):
  PYTHONPATH=v2/openwsp/src python v2/eval/build_checkpoint.py BASE v2/edits/run3-v7 models/qwen35-wsp7

Usage (the token comes from the HF_TOKEN environment variable or the huggingface-cli login):
  python tools/upload_hf.py BASE_DIR models/qwen35-wsp7 --repo USER/NAME --stage-dir DIR [--dry-run]

The check fails when a tensor other than the 3 edit tensors differs from the base (bit compare),
or when an edited tensor does not match its SHA-256 in the edit manifest. After the upload, the
script compares the SHA-256 of every uploaded file with the staged file.
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
BASE_REPO = "Qwen/Qwen3.5-0.8B"
BASE_REV = "2fc06364715b967f1860aea9cf38778875588b17"
MANIFEST = ROOT / "v2/edits/run3-v7/manifest.json"
CARD = ROOT / "hf/MODEL_CARD.md"
BRANCH = "v7"


def bits(t: torch.Tensor) -> torch.Tensor:
    """Integer view of a tensor, for an exact bit compare."""
    view = {2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
    return t.contiguous().view(view)


def tensor_sha256(t: torch.Tensor) -> str:
    """The same convention as OpenWSP edited_sha256 and tools/apply_edit.py."""
    raw = t.view(torch.int16) if t.dtype == torch.bfloat16 else t
    return hashlib.sha256(raw.contiguous().numpy().tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def load_all(folder: Path) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for f in sorted(folder.glob("*.safetensors")):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                out[k] = h.get_tensor(k)
    return out


def check(base_dir: Path, built: Path) -> dict[str, Any]:
    """Compare every tensor with the base. Return the per-tensor record for wsp_edit.json."""
    m = json.loads(MANIFEST.read_text())
    assert m["base_revision"] == BASE_REV, "the edit is for another base revision"
    expected = {t["name"]: t["edited_sha256"] for t in m["tensors"]}
    base, new = load_all(base_dir), load_all(built)
    assert set(new) == set(base), "tensor names differ from the base"
    for k in base:
        assert base[k].dtype == new[k].dtype and base[k].shape == new[k].shape, f"dtype or shape differs: {k}"
    changed = {k for k in base if not torch.equal(bits(base[k]), bits(new[k]))}
    assert changed == set(expected), f"changed {sorted(changed)}, expected {sorted(expected)}"
    record: dict[str, dict[str, Any]] = {}
    for k in sorted(changed):
        h = tensor_sha256(new[k])
        assert h == expected[k], f"hash mismatch in {k}"
        record[k] = {"shape": list(new[k].shape), "dtype": str(new[k].dtype).replace("torch.", ""),
                     "base_sha256": tensor_sha256(base[k]), "edited_sha256": h,
                     "changed_values": int((bits(base[k]) != bits(new[k])).sum())}
    total = sum(r["changed_values"] for r in record.values())
    assert total == m["changed_entries_total"], f"{total} changed values, manifest says {m['changed_entries_total']}"
    print(f"check OK: {len(changed)} tensors, {total:,} values changed, all other tensors bit-identical")
    return {"edit": "v7", "edit_sha256": m["edit_sha256"], "base_model": BASE_REPO, "base_revision": BASE_REV,
            "source": "https://github.com/mrtpk/wsp-qwen-tools/tree/main/v2/edits/run3-v7",
            "changed_values": total, "changed_tensors": record}


def stage(base_dir: Path, built: Path, meta: dict[str, Any], repo: str, out: Path) -> None:
    """Copy the checkpoint, replace the base README with the model card, add wsp_edit.json."""
    card = (CARD.read_text().replace("@REPO@", repo).replace("@BASE_REV@", BASE_REV)
            .replace("@V7_CHANGED@", f"{meta['changed_values']:,}"))
    assert not re.search(r"@[A-Z0-9_]+@", card), "placeholder left in the model card"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(built, out, ignore=shutil.ignore_patterns(".*", "README.md"))
    (out / "README.md").write_text(card)
    (out / "wsp_edit.json").write_text(json.dumps(meta, indent=1) + "\n")
    want = {f.name for f in base_dir.iterdir() if not f.name.startswith(".")} | {"wsp_edit.json"}
    have = {f.name for f in out.iterdir()}
    assert have == want, f"staged files differ from the base files: {sorted(have ^ want)}"
    print("staged", out, sorted(have))


def upload(repo: str, staged: Path) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    refs = api.list_repo_refs(repo)
    assert BRANCH not in {b.name for b in refs.branches}, f"branch {BRANCH} exists; delete it or use a new name"
    info = api.upload_folder(repo_id=repo, folder_path=str(staged), revision="main",
                             commit_message="v7: neuron programming edit (2 neurons, layer 23 MLP)")
    api.create_branch(repo, branch=BRANCH, revision=info.oid)
    local = {f.name: file_sha256(f) for f in staged.iterdir()}
    for rev in ("main", BRANCH):
        siblings = api.model_info(repo, revision=rev, files_metadata=True).siblings or []
        remote = {s.rfilename: s for s in siblings if s.rfilename != ".gitattributes"}
        assert set(remote) == set(local), f"{rev}: files differ: {sorted(set(remote) ^ set(local))}"
        for name, s in remote.items():
            if s.lfs is not None:
                assert s.lfs.sha256 == local[name], f"{rev}: LFS hash differs for {name}"
            else:
                assert s.size == (staged / name).stat().st_size, f"{rev}: size differs for {name}"
        print(f"{rev}: {len(remote)} files match the staged files")
    print("uploaded https://huggingface.co/" + repo, "commit", info.oid)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path)
    p.add_argument("built", type=Path)
    p.add_argument("--repo", required=True)
    p.add_argument("--stage-dir", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true", help="check and stage only, upload nothing")
    a = p.parse_args()
    meta = check(a.base, a.built)
    stage(a.base, a.built, meta, a.repo, a.stage_dir)
    if not a.dry_run:
        upload(a.repo, a.stage_dir)


if __name__ == "__main__":
    main()
