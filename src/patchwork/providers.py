"""Connect editor tool calls to a provider's wire format."""
import json
import os
import time
from contextlib import contextmanager

import anthropic
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

    def turn(self, model, messages, *, system=editor.SYSTEM, use_tools=True, max_output_tokens=8000):
        data = _post(self.client, "responses", {
            "model": model, "instructions": system, "input": messages,
            "tools": self.tools if use_tools else [], "max_output_tokens": max_output_tokens, "store": False,
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

    def text_turn(self, model, messages, *, max_output_tokens=16384, reasoning_effort="high"):
        """Stream a tool-free response so long weight prompts make progress."""
        payload = dict(model=model, input=messages, tools=[], store=False, stream=True,
                       instructions="Follow the supplied policy specification. Return numerical edits. Use no tools.",
                       reasoning={"effort": reasoning_effort}, max_output_tokens=max_output_tokens)
        data = None
        with self.client.stream("POST", "responses", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                kind = event.get("type")
                if kind == "response.completed":
                    data = event["response"]
                elif kind in ("error", "response.failed", "response.incomplete"):
                    raise ValueError(f"Editor stream ended with {kind}.")
        if data is None:
            raise ValueError("Editor stream ended without a complete response.")
        output = data["output"]
        content = [part for item in output if item["type"] == "message" for part in item["content"]]
        usage = data.get("usage") or {}
        return editor.Turn(history=output, log=output, calls=[],
                           text=[part["text"] for part in content if part["type"] == "output_text"],
                           input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
                           refused=any(part["type"] == "refusal" for part in content))

    def tool_results(self, results):
        return [{"type": "function_call_output", "call_id": call_id, "output": _output(content, error)}
                for call_id, content, error in results]


class ChatEditor:
    def __init__(self, client):
        self.client = client
        self.tools = [{"type": "function", "function": tool} for tool in _function_tools()]

    def turn(self, model, messages, *, system=editor.SYSTEM, use_tools=True, max_output_tokens=8000):
        data = _post(self.client, "chat/completions", {
            "model": model, "messages": [{"role": "system", "content": system}, *messages],
            "tools": self.tools if use_tools else [], "max_tokens": max_output_tokens,
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
    for attempt in range(3):
        try:
            response = client.post(endpoint, json=payload)
            break
        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    if response.is_error:
        raise ValueError(f"Editor API returned HTTP {response.status_code}: {response.text[:1000]}")
    return response.json()


def _check_provider(provider, base_url):
    if provider not in PROVIDERS:
        raise ValueError("Choose a supported editor provider.")
    if provider != "openai-compatible" and base_url is not None:
        raise ValueError("--editor-base-url requires --editor-provider openai-compatible.")


def _openai_client(provider, base_url, key_env, timeout):
    if provider == "openai":
        base_url = "https://api.openai.com/v1/"
        key_env = key_env or "OPENAI_API_KEY"
    else:
        if not base_url:
            raise ValueError("Provide --editor-base-url for the OpenAI-compatible endpoint.")
        url = httpx.URL(base_url)
        if url.scheme not in ("http", "https") or not url.host or url.userinfo or url.query or url.fragment:
            raise ValueError("--editor-base-url must be an HTTP(S) API root without credentials, query or fragment.")

    key = os.environ.get(key_env) if key_env else None
    if key_env and not key:
        raise ValueError(f"Set {key_env} to use this editor endpoint.")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return httpx.Client(base_url=base_url.rstrip("/") + "/", headers=headers, timeout=timeout)


def list_models(provider, *, base_url=None, key_env=None):
    """Read the provider catalog without sending a generation request."""
    _check_provider(provider, base_url)
    try:
        if provider == "anthropic":
            with claude.client(key_env or "ANTHROPIC_API_KEY", timeout=10) as api:
                page = api.with_options(max_retries=0).models.list()
                return [{"id": model.id, "label": model.display_name} for model in page]
        with _openai_client(provider, base_url, key_env, timeout=10) as client:
            response = client.get("models")
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict) or not isinstance(data.get("data"), list):
                raise ValueError("The endpoint returned an invalid model list.")
            ids = {item["id"] for item in data["data"] if isinstance(item, dict)
                   and isinstance(item.get("id"), str) and item["id"].strip()}
            return [{"id": model_id, "label": model_id} for model_id in sorted(ids, key=str.casefold)]
    except (anthropic.APIStatusError, httpx.HTTPStatusError) as exc:
        status = exc.response.status_code
        if status in (401, 403):
            raise ValueError(f"The provider rejected access (HTTP {status}). Check the API key and its permissions.") from None
        if status in (404, 405, 501) and provider == "openai-compatible":
            raise ValueError(f"Model listing is unavailable at this endpoint (HTTP {status}). Check the base URL or enter a model ID manually.") from None
        raise ValueError(f"The provider could not list models (HTTP {status}).") from None
    except (anthropic.APIConnectionError, httpx.RequestError):
        raise ValueError("Could not reach the provider to list models. Check the endpoint and try again.") from None
    except json.JSONDecodeError:
        raise ValueError("The endpoint returned an invalid model list.") from None


@contextmanager
def connect(provider, model, *, base_url=None, key_env=None, timeout=300):
    _check_provider(provider, base_url)
    if provider == "anthropic":
        with claude.client(key_env or "ANTHROPIC_API_KEY", timeout=timeout) as api:
            api.models.retrieve(model)
            yield claude.AnthropicEditor(api)
        return

    adapter = OpenAIEditor if provider == "openai" else ChatEditor
    with _openai_client(provider, base_url, key_env, timeout) as client:
        yield adapter(client)
