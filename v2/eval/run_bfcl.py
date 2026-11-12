"""Run the bfcl CLI with Qwen3.5 entries that call a local vLLM chat server.

The vLLM server must run with `--tool-call-parser qwen3_coder` and
`--reasoning-parser qwen3`. Set OPENAI_BASE_URL to the server.
Usage: run_bfcl.py generate --model qwen35-base-think-FC --test-category all_scoring
"""

import os

from bfcl_eval.__main__ import cli
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)

MAX_TOKENS = int(os.getenv("QWEN35_MAX_TOKENS", "16384"))

# Sampling of the Qwen3.5-0.8B README language table, which reports BFCL-V4.
SAMPLING = {
    True: dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5),
    False: dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5),
}


class Qwen35VLLMHandler(OpenAICompletionsHandler):
    enable_thinking: bool

    def _query_FC(self, inference_data: dict):
        message = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}
        s = SAMPLING[self.enable_thinking]
        kwargs = dict(
            messages=message,
            model=self.model_name,
            temperature=s["temperature"],
            top_p=s["top_p"],
            presence_penalty=s["presence_penalty"],
            max_tokens=MAX_TOKENS,
            extra_body={
                "top_k": s["top_k"],
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            },
        )
        if tools:
            kwargs["tools"] = tools
        return self.generate_with_backoff(**kwargs)

    def _parse_query_response_FC(self, api_response):
        data = super()._parse_query_response_FC(api_response)
        msg = api_response.choices[0].message
        history = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            history["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        data["model_responses_message_for_chat_history"] = history
        data["reasoning_content"] = getattr(msg, "reasoning", None) or getattr(
            msg, "reasoning_content", None
        ) or ""
        data["finish_reason"] = api_response.choices[0].finish_reason
        return data


def _handler(thinking: bool):
    return type(
        f"Qwen35VLLMHandler_{'think' if thinking else 'nothink'}",
        (Qwen35VLLMHandler,),
        {"enable_thinking": thinking},
    )


# Served model name in vLLM -> registry prefix.
SERVED = {"qwen35-base": "Qwen3.5-0.8B base", "qwen35-wsp": "Qwen3.5-0.8B RikiyaT WSP"}
for extra in filter(None, os.getenv("QWEN35_EXTRA_MODELS", "").split(",")):
    SERVED[extra] = extra

for served, label in SERVED.items():
    for thinking in (True, False):
        key = f"{served}-{'think' if thinking else 'nothink'}-FC"
        MODEL_CONFIG_MAPPING[key] = ModelConfig(
            model_name=served,
            display_name=f"{label} ({'think' if thinking else 'nothink'} FC)",
            url="https://huggingface.co/Qwen/Qwen3.5-0.8B",
            org="Qwen",
            license="apache-2.0",
            model_handler=_handler(thinking),
            input_price=None,
            output_price=None,
            is_fc_model=True,
            underscore_to_dot=True,
        )

if __name__ == "__main__":
    cli()
