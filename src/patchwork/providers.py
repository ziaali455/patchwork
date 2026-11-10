"""Connect editor tool calls to a provider's wire format."""
import json
import os
from contextlib import contextmanager

import httpx

from . import claude, editor

PROVIDERS = ("anthropic", "openai", "openai-compatible")


def _function_tools():
    return [{"name": tool["name"], "description": tool["description"],
             "parameters": tool["input_schema"]} for tool in editor.TOOLS]


class OpenAIEditor:
    def __init__(self, client):
        self.client = client
        self.tools = [{"type": "function", "strict": False, **tool} for tool in _function_tools()]

    def turn(self, model, messages):
        data = _post(self.client, "responses", {
            "model": model, "instructions": editor.SYSTEM, "input": messages,
            "tools": self.tools, "max_output_tokens": 8000, "store": False,
            "include": ["reasoning.encrypted_content"],
        })
        if data.get("status") == "failed":
            raise ValueError(f"Editor response failed: {data.get('error')}")
        output = data["output"]
        usage = data.get("usage") or {}
        content = [part for item in output if item["type"] == "message" for part in item["content"]]
        return editor.Turn(
            history=output, log=output,
            calls=[(item["call_id"], item["name"], item["arguments"])
                   for item in output if item["type"] == "function_call"],
            text=[part["text"] for part in content if part["type"] == "output_text"],
            input_tokens=usage.get("input_tokens") or 0, output_tokens=usage.get("output_tokens") or 0,
            refused=any(part["type"] == "refusal" for part in content),
        )

    def tool_results(self, results):
        return [{"type": "function_call_output", "call_id": call_id, "output": _output(content, error)}
                for call_id, content, error in results]


class ChatEditor:
    def __init__(self, client):
        self.client = client
        self.tools = [{"type": "function", "function": tool} for tool in _function_tools()]

    def turn(self, model, messages):
        data = _post(self.client, "chat/completions", {
            "model": model, "messages": [{"role": "system", "content": editor.SYSTEM}, *messages],
            "tools": self.tools, "max_tokens": 8000,
        })
        choice = data["choices"][0]
        message = choice["message"]
        usage = data.get("usage") or {}
        return editor.Turn(
            history=[message], log=[message],
            calls=[(call["id"], call["function"]["name"], call["function"]["arguments"])
                   for call in (message.get("tool_calls") or []) if call["type"] == "function"],
            text=[message["content"]] if message.get("content") else [],
            input_tokens=usage.get("prompt_tokens") or 0, output_tokens=usage.get("completion_tokens") or 0,
            refused=bool(message.get("refusal") or choice.get("finish_reason") == "content_filter"),
        )

    def tool_results(self, results):
        return [{"role": "tool", "tool_call_id": call_id, "content": _output(content, error)}
                for call_id, content, error in results]


def _output(content, error):
    return json.dumps({"error": content}) if error else content


def _post(client, endpoint, payload):
    response = client.post(endpoint, json=payload)
    if response.is_error:
        raise ValueError(f"Editor API returned HTTP {response.status_code}: {response.text[:1000]}")
    return response.json()


@contextmanager
def connect(provider, model, *, base_url=None, key_env=None):
    if provider != "openai-compatible" and base_url is not None:
        raise ValueError("--editor-base-url requires --editor-provider openai-compatible.")
    if provider == "anthropic":
        with claude.client(key_env or "ANTHROPIC_API_KEY") as api:
            api.models.retrieve(model)
            yield claude.AnthropicEditor(api)
        return

    if provider == "openai":
        base_url = "https://api.openai.com/v1/"
        key_env = key_env or "OPENAI_API_KEY"
        adapter = OpenAIEditor
    else:
        if not base_url:
            raise ValueError("Provide --editor-base-url for the OpenAI-compatible endpoint.")
        url = httpx.URL(base_url)
        if url.scheme not in ("http", "https") or not url.host or url.userinfo or url.query or url.fragment:
            raise ValueError("--editor-base-url must be an HTTP(S) API root without credentials, query or fragment.")
        adapter = ChatEditor

    key = os.environ.get(key_env) if key_env else None
    if key_env and not key:
        raise ValueError(f"Set {key_env} to use this editor endpoint.")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    with httpx.Client(base_url=base_url.rstrip("/") + "/", headers=headers, timeout=300) as client:
        yield adapter(client)
