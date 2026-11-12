"""BFCL with one harness retry (diagnostic, not a model change).

Same as run_bfcl.py, plus registry entries <served>-retry-<mode>-FC. After the first answer, the
handler checks each tool call with harness knowledge only (no answer key):
  - a tool-call markup in the text that the parser could not read,
  - an unknown function name,
  - arguments that are not a JSON object, a missing required parameter, a wrong JSON type, or a
    value outside the enum.
If a check fails, it sends the errors back (as tool messages, the way an agent harness does) and
queries once more. The second answer is the result. A text answer with no tool call gets no retry.
Each retry is logged to $RETRY_LOG (jsonl).
Usage: run_bfcl_retry.py generate --model qwen35-base-retry-think-FC --test-category single_turn
"""

import json
import os
import time

import run_bfcl
from bfcl_eval.__main__ import cli
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig

LOG = os.environ["RETRY_LOG"]
TYPES = {"string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,),
         "array": (list,), "object": (dict,)}


def type_ok(value, schema):
    t = schema.get("type")
    if t not in TYPES:
        return True
    if t in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, TYPES[t])


def check(message, tools):
    """Return a list of (tool_call_id or None, error text). Empty list: no retry."""
    schemas = {t["function"]["name"]: t["function"].get("parameters", {}) for t in tools}
    calls = message.tool_calls or []
    if not calls:
        text = message.content or ""
        if "<tool_call>" in text or "<function=" in text:
            return [(None, "Error: the tool call could not be parsed. Use the tool-call format.")]
        return []
    errors = []
    for c in calls:
        name = c.function.name
        if name not in schemas:
            errors.append((c.id, f"Error: unknown tool '{name}'. Available tools: {', '.join(schemas)}."))
            continue
        try:
            args = json.loads(c.function.arguments or "{}")
        except json.JSONDecodeError:
            errors.append((c.id, f"Error: the arguments of '{name}' are not valid JSON."))
            continue
        if not isinstance(args, dict):
            errors.append((c.id, f"Error: the arguments of '{name}' must be a JSON object."))
            continue
        props = schemas[name].get("properties", {})
        problems = [f"missing required parameter '{p}'" for p in schemas[name].get("required", []) if p not in args]
        for k, v in args.items():
            if k not in props:
                problems.append(f"unknown parameter '{k}'")
            elif not type_ok(v, props[k]):
                problems.append(f"parameter '{k}' must be of type {props[k]['type']}")
            elif "enum" in props[k] and v not in props[k]["enum"]:
                problems.append(f"parameter '{k}' must be one of {props[k]['enum']}")
        if problems:
            errors.append((c.id, f"Error in '{name}': " + "; ".join(problems) + "."))
    return errors


class RetryHandler(run_bfcl.Qwen35VLLMHandler):
    def _query_FC(self, inference_data: dict):
        first, latency = super()._query_FC(inference_data)
        msg = first.choices[0].message
        errors = check(msg, inference_data.get("tools") or [])
        if not errors:
            return first, latency
        history = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            history["tool_calls"] = [{"id": c.id, "type": "function",
                                      "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                     for c in msg.tool_calls]
        extra = [history]
        ids = {e[0] for e in errors}
        for c in msg.tool_calls or []:
            text = next((e for i, e in errors if i == c.id), "OK (not run, because another call has an error).")
            extra.append({"role": "tool", "tool_call_id": c.id, "content": text})
        if None in ids:
            extra.append({"role": "user", "content": errors[0][1]})
        retry = dict(inference_data, message=inference_data["message"] + extra)
        second, latency2 = super()._query_FC(retry)
        m2 = second.choices[0].message
        with open(LOG, "a") as f:
            f.write(json.dumps({"model": self.model_name, "thinking": self.enable_thinking,
                                "question": inference_data["message"][-1].get("content"),
                                "errors": [e for _, e in errors],
                                "second_calls": [[c.function.name, c.function.arguments] for c in m2.tool_calls or []],
                                "second_errors": [e for _, e in check(m2, inference_data.get("tools") or [])],
                                "time": time.time()}) + "\n")
        return second, latency + latency2


for served in list(run_bfcl.SERVED):
    for thinking in (True, False):
        mode = "think" if thinking else "nothink"
        MODEL_CONFIG_MAPPING[f"{served}-retry-{mode}-FC"] = ModelConfig(
            model_name=served,
            display_name=f"{run_bfcl.SERVED[served]} retry ({mode} FC)",
            url="https://huggingface.co/Qwen/Qwen3.5-0.8B", org="Qwen", license="apache-2.0",
            model_handler=type(f"Retry_{mode}", (RetryHandler,), {"enable_thinking": thinking}),
            input_price=None, output_price=None, is_fc_model=True, underscore_to_dot=True)

if __name__ == "__main__":
    cli()
