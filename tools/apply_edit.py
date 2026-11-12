"""Build a full checkpoint: copy the base directory, then replace the tensors in EDIT.safetensors.
Checks each written tensor against the sha256 in EDIT.json.
Usage: apply_edit.py BASE_DIR EDIT.safetensors OUT_DIR"""
import hashlib, json, shutil, sys
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

base, edit, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
fn = "model.safetensors-00001-of-00001.safetensors"
info = json.load(open(edit.replace(".safetensors", ".json")))
new = load_file(edit)
assert set(new) == set(info), "edit file and json differ"
shutil.copytree(base, out, ignore=shutil.ignore_patterns(fn, ".*"))
with safe_open(base / fn, "pt") as f:
    meta = f.metadata()
    tensors = {k: f.get_tensor(k) for k in f.keys()}
for k, y in new.items():
    assert k in tensors and tensors[k].shape == y.shape and tensors[k].dtype == y.dtype, k
    h = hashlib.sha256(y.view(torch.int16).numpy().tobytes() if y.dtype == torch.bfloat16 else y.numpy().tobytes()).hexdigest()
    assert h == info[k]["sha256"], f"hash mismatch {k}"
    tensors[k] = y
save_file(tensors, out / fn, metadata=meta)
print("wrote", out, "replaced", len(new), "tensors")
