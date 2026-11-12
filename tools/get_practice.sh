#!/usr/bin/env bash
# Download the raw practice data at the pinned revisions, check the hashes, and convert it.
# Run from the repository root (the folder that holds v2/). Needs: python with huggingface_hub.
set -euo pipefail
P=v2/practice
mkdir -p "$P/raw"
python3 - <<'PY'
import shutil
from huggingface_hub import hf_hub_download
files = [  # (repo, revision, file in repo, name in practice/raw)
 ("nvidia/When2Call", "0582f7749df63a96fdc3070932e83e72396ace53", "train/when2call_train_sft.jsonl", "When2Call__train_when2call_train_sft.jsonl"),
 ("nvidia/When2Call", "0582f7749df63a96fdc3070932e83e72396ace53", "README.md", "When2Call__README.md"),
 ("Team-ACE/ToolACE", "6bda777c88d21e5a204703c1ee45597a8fa4f734", "data.json", "ToolACE__data.json"),
 ("Team-ACE/ToolACE", "6bda777c88d21e5a204703c1ee45597a8fa4f734", "README.md", "ToolACE__README.md"),
 ("Salesforce/APIGen-MT-5k", "abc4a517d67c541f85f6470cbd8fd3186b36830e", "apigen-mt_5k.json", "APIGen-MT-5k__apigen-mt_5k.json"),
 ("Salesforce/APIGen-MT-5k", "abc4a517d67c541f85f6470cbd8fd3186b36830e", "README.md", "APIGen-MT-5k__README.md"),
 ("glaiveai/glaive-function-calling-v2", "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac", "glaive-function-calling-v2.json", "glaive__glaive-function-calling-v2.json"),
 ("glaiveai/glaive-function-calling-v2", "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac", "README.md", "glaive__README.md"),
]
for repo, rev, f, name in files:
    shutil.copy(hf_hub_download(repo, f, repo_type="dataset", revision=rev), f"v2/practice/raw/{name}")
    print("saved", name)
PY
(cd "$P/raw" && sha256sum -c ../../../tools/practice-raw.sha256)
python3 -I "$P/convert.py" "$P/raw" "$P" v2/eval/bfcl-user-messages.json
python3 -I "$P/convert_glaive.py" "$P/raw/glaive__glaive-function-calling-v2.json" "$P/glaive.jsonl" v2/eval/bfcl-user-messages.json
(cd "$P" && sha256sum -c ../../tools/practice-converted.sha256)
