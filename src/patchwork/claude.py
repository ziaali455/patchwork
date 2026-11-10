"""Native Anthropic editor support."""
import os

from anthropic import Anthropic

from . import editor


def client(key_env="ANTHROPIC_API_KEY"):
    key = os.environ.get(key_env)
    if not key:
        raise ValueError(f"Set {key_env} with a key from the Claude Console.")
    return Anthropic(api_key=key, timeout=300, max_retries=2)


class AnthropicEditor:
    def __init__(self, api):
        self.api = api

    def turn(self, model, messages):
        response = self.api.messages.create(model=model, max_tokens=8000, system=editor.SYSTEM,
                                            tools=editor.TOOLS, messages=messages)
        return editor.Turn(
            history=[{"role": "assistant", "content": response.content}],
            log=[block.model_dump() for block in response.content],
            calls=[(block.id, block.name, block.input) for block in response.content if block.type == "tool_use"],
            text=[block.text for block in response.content if block.type == "text"],
            input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens,
            refused=response.stop_reason == "refusal",
        )

    def tool_results(self, results):
        return [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call_id, "content": content, "is_error": error}
            for call_id, content, error in results
        ]}]


def session(api, model, goal, revision, weights, probe, **kwargs):
    return editor.session(AnthropicEditor(api), model, goal, revision, weights, probe, **kwargs)
