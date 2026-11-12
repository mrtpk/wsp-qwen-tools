"""Harness: apply an exported openwsp edit to the pinned base and write a full checkpoint.

Usage: build_checkpoint.py <base-dir> <edit-dir> <out-dir>
"""
import shutil, sys
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from openwsp.core import WeightStore

base, edit, out = map(Path, sys.argv[1:4])
store = WeightStore(base, edit_dir=out.with_name(out.name + "-store"))
print(store.apply_edit(edit))
out.mkdir(parents=True, exist_ok=False)
names = set(store.edited())
for f in base.iterdir():
    if f.suffix == ".safetensors":
        with safe_open(f, framework="pt") as h:
            t = {n: h.get_tensor(n) for n in h.keys()}
        for n in names & t.keys():
            t[n] = store.load(n).to(t[n].dtype).contiguous()
        save_file(t, out / f.name, metadata={"format": "pt"})
    elif f.is_file():
        shutil.copyfile(f, out / f.name)
shutil.rmtree(out.with_name(out.name + "-store"))
print("written", out, "edited", sorted(names))
