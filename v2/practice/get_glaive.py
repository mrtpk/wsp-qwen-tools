"""Download glaiveai/glaive-function-calling-v2 (pinned revision) into practice/raw/."""
from huggingface_hub import HfApi, hf_hub_download
import shutil
repo = "glaiveai/glaive-function-calling-v2"
info = HfApi().dataset_info(repo)
print("revision", info.sha, "license", (info.card_data or {}).get("license"))
print([s.rfilename for s in info.siblings])
for f in [s.rfilename for s in info.siblings if s.rfilename.endswith((".json", ".md"))]:
    p = hf_hub_download(repo, f, repo_type="dataset", revision=info.sha)
    shutil.copy(p, f"practice/raw/glaive__{f.replace('/', '_')}")
    print("saved", f)
