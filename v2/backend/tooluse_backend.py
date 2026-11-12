"""Local GPU forward backend for openwsp: Qwen3.5-0.8B with tool-use prompts.

Factory: ``tooluse_backend:create`` (no arguments). Configuration comes from the
environment (see ``create``). Two request types, as in Experiment 8:

generate
    {"type": "generate", "model": "original" | "edited" | "both",
     "items": [{"messages": [...], "tools": [...]}, ...],
     "enable_thinking": bool, "n": 1, "max_tokens": int, "temperature": float,
     "top_p": float, "top_k": int, "presence_penalty": float, "seed": int}
    Runs through ``vllm serve`` with the qwen3_coder tool parser and the qwen3
    reasoning parser (the BFCL path). At most 512 samples in total.

teacher_forced
    The Experiment 8 schema (``texts`` or ``chats``, ``return`` options),
    executed by ``replications.exp8.internals.capture`` with HF Transformers.

Budget: one accepted request is one forward pass. An identical request on the
same weights returns the stored result and does not count. A failed pass does
not count. The ledger refuses passes above OPENWSP_PASS_BUDGET.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

ROOT = Path("@ROOT@")
SAMPLE_LIMIT = 512
GEN_KEYS = {"type", "model", "items", "enable_thinking", "n", "max_tokens", "temperature", "top_p",
            "top_k", "presence_penalty", "seed"}


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def weights_fingerprint(store) -> str:
    """sha256 over the bf16 values of every submitted tensor; 'base' when nothing is edited."""
    names = sorted(store.edited())
    if not names:
        return "base"
    digest = hashlib.sha256()
    for name in names:
        tensor = store.load(name).to(torch.bfloat16).contiguous()
        digest.update(_canonical([name, list(tensor.shape)]).encode())
        digest.update(tensor.view(torch.uint16).numpy().tobytes())
    return digest.hexdigest()


class Ledger:
    """Append-only pass ledger in the work directory."""

    def __init__(self, path: Path, budget: int):
        self.path, self.budget = path, budget
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def rows(self):
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def counted(self) -> int:
        return sum(1 for row in self.rows() if row["status"] == "done")

    def find(self, key):
        return next((row for row in self.rows() if row["key"] == key and row["status"] == "done"), None)

    def append(self, row):
        with self.path.open("a") as handle:
            handle.write(_canonical(row) + "\n")


class VLLMServer:
    """One `vllm serve` process for one checkpoint; state in <work>/servers/<arm>.json."""

    def __init__(self, arm: str, port: int, fraction: float, work: Path, max_len: int):
        self.arm, self.port, self.fraction, self.max_len = arm, port, fraction, max_len
        self.state = work / "servers" / f"{arm}.json"
        self.log = work / "logs" / f"vllm-{arm}.log"
        self.state.parent.mkdir(parents=True, exist_ok=True)
        self.log.parent.mkdir(parents=True, exist_ok=True)

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as response:
                return response.status == 200
        except OSError:
            return False

    def _stop(self):
        if self.state.exists():
            pid = json.loads(self.state.read_text())["pid"]
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            for _ in range(120):
                if not self._healthy():
                    break
                time.sleep(1)
            time.sleep(5)
            self.state.unlink()

    def ensure(self, checkpoint: str, fingerprint: str):
        if self.state.exists():
            info = json.loads(self.state.read_text())
            if info["fingerprint"] == fingerprint and self._healthy():
                return
            self._stop()
        env = dict(os.environ, CUDA_HOME=str(ROOT / ".venv/lib/python3.12/site-packages/nvidia/cu13"),
                   FLASHINFER_EXTRA_LDFLAGS=f"-L{ROOT / '.cudart'}",
                   PATH=f"{ROOT / '.venv/bin'}:{os.environ['PATH']}")
        command = [str(ROOT / ".venv/bin/vllm"), "serve", checkpoint, "--served-model-name", self.arm,
                   "--port", str(self.port), "--gpu-memory-utilization", str(self.fraction),
                   "--max-model-len", str(self.max_len), "--language-model-only",
                   "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
                   "--reasoning-parser", "qwen3", "--enable-prefix-caching"]
        with self.log.open("a") as log:
            log.write(f"\n=== start {time.ctime()} fingerprint {fingerprint}\n")
            log.flush()
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        self.state.write_text(_canonical({"pid": process.pid, "fingerprint": fingerprint,
                                          "checkpoint": checkpoint}))
        for _ in range(600):
            if process.poll() is not None:
                raise RuntimeError(f"vLLM server {self.arm} exited; see {self.log}")
            if self._healthy():
                return
            time.sleep(1)
        raise TimeoutError(f"vLLM server {self.arm} did not start; see {self.log}")

    def chat(self, item, sampling, thinking):
        from openai import OpenAI

        client = OpenAI(base_url=f"http://127.0.0.1:{self.port}/v1", api_key="EMPTY", timeout=3600)
        kwargs = dict(model=self.arm, messages=item["messages"], max_tokens=sampling["max_tokens"],
                      temperature=sampling["temperature"], top_p=sampling["top_p"],
                      presence_penalty=sampling["presence_penalty"], n=sampling["n"], seed=sampling["seed"],
                      extra_body={"top_k": sampling["top_k"],
                                  "chat_template_kwargs": {"enable_thinking": thinking}})
        if item.get("tools"):
            kwargs["tools"] = item["tools"]
        response = client.chat.completions.create(**kwargs)
        samples = []
        for choice in response.choices:
            message = choice.message
            samples.append({
                "reasoning": getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None),
                "content": message.content,
                "tool_calls": [{"name": call.function.name, "arguments": call.function.arguments}
                               for call in (message.tool_calls or [])],
                "finish_reason": choice.finish_reason,
            })
        return {"samples": samples, "completion_tokens": response.usage.completion_tokens}


class ToolUseBackend:
    def __init__(self, base: Path, work: Path, budget: int):
        self.base, self.work = base, work
        self.ledger = Ledger(work / "passes.jsonl", budget)
        self.servers = {
            "original": VLLMServer("original", 8201, 0.28, work, 32768),
            "edited": VLLMServer("edited", 8202, 0.28, work, 32768),
        }
        self._hf = None

    # ---- checkpoints -------------------------------------------------------------
    def _checkpoint(self, store, fingerprint) -> str:
        if fingerprint == "base":
            return str(self.base)
        target = self.work / "checkpoints" / fingerprint
        marker = target / "openwsp-v2-complete.json"
        if marker.exists():
            return str(target)
        from safetensors import safe_open
        from safetensors.torch import save_file

        for old in (self.work / "checkpoints").glob("*"):
            shutil.rmtree(old)
        temporary = target.with_name(target.name + ".tmp")
        temporary.mkdir(parents=True)
        names = set(store.edited())
        for source in self.base.iterdir():
            if source.suffix == ".safetensors":
                with safe_open(source, framework="pt", device="cpu") as handle:
                    tensors = {name: handle.get_tensor(name) for name in handle.keys()}  # noqa: SIM118
                for name in names & tensors.keys():
                    tensors[name] = store.load(name).to(tensors[name].dtype).contiguous()
                save_file(tensors, temporary / source.name, metadata={"format": "pt"})
                names -= tensors.keys()
            elif source.is_file():
                shutil.copyfile(source, temporary / source.name)
        assert not names, f"edited tensors not in the checkpoint: {sorted(names)}"
        (temporary / marker.name).write_text(_canonical({"fingerprint": fingerprint}))
        temporary.rename(target)
        return str(target)

    # ---- request types -----------------------------------------------------------
    def _generate(self, request, store, fingerprint, destination):
        assert set(request) <= GEN_KEYS, f"unknown generate keys: {sorted(set(request) - GEN_KEYS)}"
        model, items = request["model"], request["items"]
        assert model in ("original", "edited", "both"), model
        assert isinstance(items, list) and items, "items must be a nonempty list"
        for item in items:
            assert isinstance(item.get("messages"), list) and item["messages"], "each item needs messages"
        sampling = {"n": request.get("n", 1), "max_tokens": request.get("max_tokens", 16384),
                    "temperature": request.get("temperature", 1.0), "top_p": request.get("top_p", 0.95),
                    "top_k": request.get("top_k", 20), "presence_penalty": request.get("presence_penalty", 1.5),
                    "seed": request.get("seed", 0)}
        arms = ("original", "edited") if model == "both" else (model,)
        assert len(items) * sampling["n"] * len(arms) <= SAMPLE_LIMIT, "at most 512 samples per pass"
        thinking = request.get("enable_thinking", True)
        assert isinstance(thinking, bool)
        result = {}
        for arm in arms:
            arm_fp = "base" if arm == "original" else fingerprint
            self.servers[arm].ensure(self._checkpoint(store, arm_fp), arm_fp)
        for arm in arms:
            with ThreadPoolExecutor(64) as pool:
                outputs = list(pool.map(lambda item, arm=arm: self.servers[arm].chat(item, sampling, thinking),
                                        items))
            result[arm] = outputs
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "outputs.json"
        path.write_text(json.dumps(result))
        summary = {}
        for arm, outputs in result.items():
            samples = [s for o in outputs for s in o["samples"]]
            summary[arm] = {"samples": len(samples),
                            "with_tool_call": sum(bool(s["tool_calls"]) for s in samples),
                            "finish_length": sum(s["finish_reason"] == "length" for s in samples)}
        return {"ok": True, "type": "generate", "model": model, "file": str(path), "summary": summary}

    def _teacher_forced(self, request, store, destination, warm_only=False):
        from replications.exp8.internals import capture
        from transformers import AutoModelForCausalLM, AutoTokenizer

        arm = request.get("model")
        assert arm in ("original", "edited"), "teacher_forced model must be original or edited"
        if self._hf is None:
            tokenizer = AutoTokenizer.from_pretrained(self.base)
            model = AutoModelForCausalLM.from_pretrained(self.base, dtype=torch.bfloat16, device_map="cuda")
            self._hf = (model, tokenizer)
        if warm_only:
            return None
        model, tokenizer = self._hf
        params = dict(model.named_parameters())

        def param(name):
            for candidate in (name, name.replace("model.language_model.", "model."), "model." + name):
                if candidate in params:
                    return params[candidate]
            raise KeyError(f"no HF parameter for {name}")

        names = store.edited() if arm == "edited" else []
        saved = {name: param(name).detach().clone() for name in names}
        try:
            for name in names:
                param(name).data.copy_(store.load(name).to(param(name).dtype))
            request = {key: value for key, value in request.items() if key not in ("type", "model")}
            meta = capture(model, tokenizer, request, destination)
        finally:
            for name, value in saved.items():
                param(name).data.copy_(value)
        return {**meta, "model": arm, "directory": str(destination)}

    def _probe(self, request, store, destination):
        """Compact reads at chosen positions.

        {"type": "probe", "model": "original" | "edited", "texts": [str, ...],
         "positions": "last" | [[int, ...], ...],
         "mlp_in": [layers], "mlp_act": [layers], "hidden": [layers], "token_ids": [ids]}
        mlp_in_L: the input of layer L's MLP (after its norm, 1024-d). mlp_act_L: the
        input of down_proj (3584-d). hidden_L: the residual stream before layer L.
        logp: log probabilities of token_ids for the NEXT token at each position.
        """
        allowed = {"type", "model", "texts", "positions", "mlp_in", "mlp_act", "hidden", "token_ids"}
        assert set(request) <= allowed, f"unknown probe keys: {sorted(set(request) - allowed)}"
        arm = request["model"]
        assert arm in ("original", "edited")
        texts = request["texts"]
        assert isinstance(texts, list) and 1 <= len(texts) <= 1024
        positions = request.get("positions", "last")
        assert positions == "last" or (isinstance(positions, list) and len(positions) == len(texts))
        token_ids = request.get("token_ids", [])
        self._teacher_forced({"type": "x", "model": "original", "texts": ["warm"]}, store, destination / "_warm",
                             warm_only=True)
        model, tokenizer = self._hf
        text_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        layers = text_model.layers
        params = dict(model.named_parameters())

        def param(name):
            for candidate in (name, name.replace("model.language_model.", "model."), "model." + name):
                if candidate in params:
                    return params[candidate]
            raise KeyError(name)

        names = store.edited() if arm == "edited" else []
        saved = {name: param(name).detach().clone() for name in names}
        store_out = {f"mlp_in_{layer}": [] for layer in request.get("mlp_in", [])}
        store_out.update({f"mlp_act_{layer}": [] for layer in request.get("mlp_act", [])})
        store_out.update({f"hidden_{layer}": [] for layer in request.get("hidden", [])})
        rows, logps, current = [], [], {}
        hooks = []
        for layer in request.get("mlp_in", []):
            hooks.append(layers[layer].mlp.register_forward_pre_hook(
                lambda m, a, key=f"mlp_in_{layer}": current.__setitem__(key, a[0][0])))
        for layer in request.get("mlp_act", []):
            hooks.append(layers[layer].mlp.down_proj.register_forward_pre_hook(
                lambda m, a, key=f"mlp_act_{layer}": current.__setitem__(key, a[0][0])))
        for layer in request.get("hidden", []):
            hooks.append(layers[layer].register_forward_pre_hook(
                lambda m, a, k, key=f"hidden_{layer}": current.__setitem__(
                    key, (a[0] if a else k["hidden_states"])[0]), with_kwargs=True))
        try:
            for name in names:
                param(name).data.copy_(store.load(name).to(param(name).dtype))
            with torch.inference_mode():
                for i, text in enumerate(texts):
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    assert 0 < len(ids) <= 32768
                    pos = [len(ids) - 1] if positions == "last" else positions[i]
                    assert all(0 <= p < len(ids) for p in pos)
                    tokens = torch.tensor([ids], device=text_model.embed_tokens.weight.device)
                    out = text_model(input_ids=tokens, use_cache=False)
                    index = torch.tensor(pos, device=tokens.device)
                    for key in store_out:
                        store_out[key].append(current[key][index].float().cpu())
                    if token_ids:
                        head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
                        logp = head(out.last_hidden_state[0, index]).float().log_softmax(-1)[:, token_ids]
                        logps.append(logp.cpu())
                    rows += [(i, p) for p in pos]
        finally:
            for hook in hooks:
                hook.remove()
            for name, value in saved.items():
                param(name).data.copy_(value)
        import numpy as np

        destination.mkdir(parents=True, exist_ok=True)
        arrays = {key: torch.cat(value).numpy().astype(np.float32) for key, value in store_out.items()}
        arrays["rows"] = np.array(rows, dtype=np.int64)
        if token_ids:
            arrays["logp"] = torch.cat(logps).numpy()
        path = destination / "probe.npz"
        np.savez(path, **arrays)
        return {"ok": True, "type": "probe", "model": arm, "file": str(path),
                "shapes": {key: list(value.shape) for key, value in arrays.items()}}

    # ---- entry point -------------------------------------------------------------
    def forward(self, request, store):
        fingerprint = weights_fingerprint(store)
        kind = request.get("type")
        assert kind in ("generate", "teacher_forced", "probe"), "type must be generate, teacher_forced or probe"
        uses_edit = request.get("model") in ("edited", "both")
        key = hashlib.sha256(_canonical({"request": request,
                                         "weights": fingerprint if uses_edit else "base"}).encode()).hexdigest()
        cached = self.ledger.find(key)
        if cached is not None:
            return {**cached["result"], "cached": True, "pass": cached["pass"]}
        used = self.ledger.counted()
        if used >= self.ledger.budget:
            raise RuntimeError(f"Forward-pass budget exhausted ({used}/{self.ledger.budget})")
        destination = self.work / "forward" / key[:16]
        started = time.time()
        try:
            if kind == "generate":
                result = self._generate(request, store, fingerprint, destination)
            elif kind == "probe":
                result = self._probe(request, store, destination)
            else:
                result = self._teacher_forced(request, store, destination)
        except Exception as exc:
            self.ledger.append({"key": key, "status": "failed", "error": f"{type(exc).__name__}: {exc}"[:500],
                                "time": time.time()})
            raise
        result.update(weights=fingerprint if uses_edit else "base", seconds=round(time.time() - started, 1))
        self.ledger.append({"key": key, "status": "done", "pass": used + 1, "type": kind,
                            "model": request.get("model"), "weights": result["weights"], "result": result,
                            "time": time.time()})
        return {**result, "cached": False, "pass": used + 1}


def create():
    base = Path(os.environ["OPENWSP_BASE"])
    work = Path(os.environ["OPENWSP_WORK"])
    budget = int(os.environ.get("OPENWSP_PASS_BUDGET", "50"))
    return ToolUseBackend(base, work, budget)
