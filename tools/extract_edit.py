"""Write the tensors that differ between an edited checkpoint and the base.
Usage: extract_edit.py BASE_DIR EDITED_DIR OUT.safetensors"""
import hashlib, json, sys
from safetensors import safe_open
from safetensors.torch import save_file
import torch

base, edited, out = sys.argv[1:4]
fn = "model.safetensors-00001-of-00001.safetensors"
changed, info = {}, {}
with safe_open(f"{base}/{fn}", "pt") as b, safe_open(f"{edited}/{fn}", "pt") as e:
    assert set(b.keys()) == set(e.keys()), "tensor names differ"
    for k in sorted(e.keys()):
        x, y = b.get_tensor(k), e.get_tensor(k)
        if not torch.equal(x, y):
            changed[k] = y.contiguous()
            d = (y.float() - x.float())
            s = torch.linalg.svdvals(d) if d.dim() == 2 else None
            info[k] = {"shape": list(y.shape), "dtype": str(y.dtype), "changed_values": int((x != y).sum()),
                       "sha256": hashlib.sha256(y.view(torch.uint8).numpy().tobytes() if y.dtype != torch.bfloat16 else y.view(torch.int16).numpy().tobytes()).hexdigest(),
                       "delta_sigma_1_2": [float(v) for v in s[:2]] if s is not None else None,
                       "relative_norm": float(d.norm() / x.float().norm())}
assert changed, "no tensor differs"
save_file(changed, out, metadata={"base": base.rstrip("/").split("/")[-1]})
json.dump(info, open(out.replace(".safetensors", ".json"), "w"), indent=1)
print(out, {k: v["changed_values"] for k, v in info.items()})
