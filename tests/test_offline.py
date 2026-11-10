"""Offline end-to-end tests with tiny random models and scripted editors.

Covers scoring, greedy generation, every editor tool, both HTTP protocols,
exact revert, and patch replay and export for multiple target architectures.
No downloads, no API key.  Run:  uv run --extra local python tests/test_offline.py
"""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from patchwork import claude, patchfile, providers  # noqa: E402
from patchwork.benchmark import Generator, evaluate, is_correct  # noqa: E402
from patchwork.cli import main  # noqa: E402
from patchwork.model import load  # noqa: E402
from patchwork.probe import Probe  # noqa: E402
from patchwork.weights import Weights  # noqa: E402

DOWN = "model.layers.1.mlp.down_proj.weight"


def tiny_model(seed=0):
    chars = list("abcdefghijklmnopqrstuvwxyz0123456789 .,?$\\{}+-*/=\n") + ["<|endoftext|>", "<pad>"]
    vocab = {c: i for i, c in enumerate(chars)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<pad>"))
    tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>", pad_token="<pad>")
    config = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, eos_token_id=vocab["<|endoftext|>"],
                         pad_token_id=vocab["<pad>"])
    torch.manual_seed(seed)
    model = Qwen2ForCausalLM(config).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


class FakeClaude:
    """Plays a short, plausible session, including two invalid requests."""

    SCRIPT = [
        ("inspect_columns", {"tensor": DOWN, "start": 0, "count": 4}),
        ("run_practice", {"n_random": 4}),
        ("find_direction", {"name": "d", "layer": 1, "group_a": ["what is 1+1?"], "group_b": ["abc def"]}),
        ("align_columns", {"tensor": DOWN, "direction": "d", "top_k": 8}),
        ("apply_edit", {"tensor": DOWN, "columns": [3, 7, 11, 15], "operation": "project_out",
                        "direction": "d", "strength": 1.0, "rationale": "remove d"}),
        ("apply_edit", {"tensor": DOWN, "start": 20, "count": 4, "operation": "scale", "strength": 0.5, "rationale": "damp"}),
        ("apply_edit", {"tensor": DOWN, "columns": [3], "operation": "scale", "strength": 0.5, "rationale": "overlap"}),  # error
        ("apply_edit", {"tensor": "nope", "start": 0, "count": 1, "operation": "scale", "strength": 1, "rationale": "x"}),  # error
        ("revert_edit", {"which": "last"}),
        ("run_practice", {"ids": [0, 1]}),
        ("finish", {"summary": "kept the projection", "helped": True}),
    ]

    def __init__(self):
        self.step, self.messages = 0, self
        self.tool_results = []

    def create(self, messages, **_):
        if messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list):
            self.tool_results.extend(messages[-1]["content"])
        name, args = self.SCRIPT[self.step]
        self.step += 1
        block = SimpleNamespace(type="tool_use", id=f"call{self.step}", name=name, input=args,
                                model_dump=lambda: {"name": name, "input": args})
        return SimpleNamespace(content=[block], stop_reason="tool_use",
                               usage=SimpleNamespace(input_tokens=100, output_tokens=10))


def test_editor_cli():
    import os

    _, tokenizer = tiny_model()
    target = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(tokenizer), n_embd=32, n_inner=64, n_head=4, n_layer=1,
        bos_token_id=None, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    ))
    tensor_name = "transformer.h.0.mlp.c_proj.weight"
    edit = {"tensor": tensor_name, "columns": [1], "operation": "scale",
            "strength": 0.5, "rationale": "verify the transposed layout"}
    finish = {"summary": "edited one column", "helped": True}
    client = httpx.Client

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        checkpoint = tmp / "gpt2"
        target.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        practice, evaluation = tmp / "practice.json", tmp / "eval.json"
        practice.write_text('[{"prompt":"practice case","answer":"7"}]')
        evaluation.write_text('[{"prompt":"held out case","answer":"9"}]')
        for provider, endpoint, flags in (
            ("openai", "https://api.openai.com/v1/responses", []),
            ("openai-compatible", "https://editor.example/v1/chat/completions",
             ["--editor-base-url", "https://editor.example/v1", "--editor-key-env", "EDITOR_TEST_KEY"]),
        ):
            requests = []
            reasoning = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "encrypted"}

            def respond(request):
                payload = json.loads(request.content)
                assert str(request.url) == endpoint
                assert request.headers["authorization"] == "Bearer test-key"
                assert payload["model"] == "test-model"
                assert "held out case" not in request.content.decode()
                requests.append(payload)
                first = len(requests) == 1
                calls = [("bad", "apply_edit", "not json"), ("edit", "apply_edit", json.dumps(edit))] if first else [
                    ("done", "finish", json.dumps(finish))]
                if provider == "openai":
                    assert payload["store"] is False and payload["include"] == ["reasoning.encrypted_content"]
                    assert payload["tools"][0]["type"] == "function"
                    if not first:
                        assert reasoning in payload["input"]
                        error, result = payload["input"][-2:]
                        assert error["call_id"] == "bad" and "error" in json.loads(error["output"])
                        assert result["call_id"] == "edit" and json.loads(result["output"])["parameters_changed"] == 32
                    items = [{"type": "function_call", "call_id": id_, "name": name, "arguments": args}
                             for id_, name, args in calls]
                    return httpx.Response(200, json={"output": [reasoning, *items],
                                                     "usage": {"input_tokens": 11, "output_tokens": 4}})
                assert payload["tools"][0]["function"]["name"] == "inspect_columns"
                if not first:
                    error, result = payload["messages"][-2:]
                    assert error["tool_call_id"] == "bad" and "error" in json.loads(error["content"])
                    assert result["tool_call_id"] == "edit" and json.loads(result["content"])["parameters_changed"] == 32
                message = {"role": "assistant", "content": None, "tool_calls": [
                    {"id": id_, "type": "function", "function": {"name": name, "arguments": args}}
                    for id_, name, args in calls
                ]}
                return httpx.Response(200, json={"choices": [{"message": message}],
                                                 "usage": {"prompt_tokens": 11, "completion_tokens": 4}})

            out = tmp / provider
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "EDITOR_TEST_KEY": "test-key"}), patch.object(
                providers.httpx, "Client", side_effect=lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs),
            ):
                assert main(["patch", "--target-model", str(checkpoint), "--editor-provider", provider,
                             "--editor-model", "test-model", "--output", str(out), "--prompt", "test edits",
                             "--max-new-tokens", "2", "--eval", str(evaluation), "--practice", str(practice), *flags]) == 0
            assert len(requests) == 2
            report = json.loads((out / "report.json").read_text())
            assert report["editor_provider"] == provider and report["editor_summary"] == finish
            assert report["usage"] == {"input_tokens": 22, "output_tokens": 8, "turns": 2}
            assert report["parameters_changed"] == 32

        assert main(["eval", "--patch", str(out), "--eval", str(evaluation), "--max-new-tokens", "2"]) == 0
        exported = tmp / "exported"
        assert main(["export", "--patch", str(out), "--output", str(exported)]) == 0
        model, _ = load(exported, "cpu")
        expected = dict(target.named_parameters())[tensor_name].detach().clone()
        expected[1] *= 0.5
        assert torch.equal(dict(model.named_parameters())[tensor_name], expected)
    print("editor CLI ok: two API formats, GPT-2 patch, manifest-driven eval and export")


def main_test():
    # Scoring
    assert is_correct("so the answer is \\boxed{1,234}", "1234")
    assert is_correct("It is 42.", "42") and not is_correct("no idea", "42")
    assert is_correct("  tokyo ", "Tokyo") and not is_correct("Kyoto", "Tokyo")

    model, tokenizer = tiny_model()
    generator = Generator(model, tokenizer, max_new_tokens=12, thinking=False)
    cases = [{"prompt": f"what is {i}+1?", "answer": str(i + 1)} for i in range(6)]
    a, b = evaluate(generator, cases), evaluate(generator, cases)
    assert [c["raw_output"] for c in a["cases"]] == [c["raw_output"] for c in b["cases"]], "greedy should be deterministic"
    print("eval ok:", {k: a[k] for k in ("accuracy", "correct", "total", "reached_token_limit")})

    original = {n: p.clone() for n, p in model.named_parameters()}
    weights = Weights(model, budget_fraction=0.05)
    probe = Probe(model, tokenizer, generator, weights, cases, budget=5)
    fake = FakeClaude()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        finished, usage = claude.session(fake, "fake", "goal", "rev", weights, probe,
                                         log_path=tmp / "t.jsonl", say=lambda s: None)
        assert finished["helped"] and probe.used == 3, (finished, probe.used)
        errors = [r for r in fake.tool_results if r["is_error"]]
        assert len(errors) == 2, [r["content"] for r in errors]
        assert len(weights.history) == 1 and weights.history[0]["edit"]["operation"] == "project_out"
        print(f"session ok: {usage['turns']} turns, {probe.used} passes, 2 invalid edits rejected, revert worked")

        # The kept edit removed the direction from exactly those columns
        d = weights.directions["d"]
        cols = torch.tensor([3, 7, 11, 15])
        assert (d @ weights.tensors[DOWN][:, cols]).abs().max() < 1e-5
        untouched = [n for n in original if n != DOWN]
        assert all(torch.equal(original[n], dict(model.named_parameters())[n]) for n in untouched)

        # Patch file round trip
        tensors, changed = patchfile.build(weights)
        patchfile.write(tmp / "patch", tensors, {"goal": "g", "base": {"fingerprint": "fp"}})
        fresh, _ = tiny_model()
        assert patchfile.apply(fresh, tmp / "patch", "fp") == changed
        for (n, x), (_, y) in zip(model.named_parameters(), fresh.named_parameters()):
            assert torch.allclose(x, y, atol=1e-6), n
        patchfile.apply(fresh, tmp / "patch", "fp", sign=-1)
        for n, y in fresh.named_parameters():
            assert torch.allclose(original[n], y, atol=1e-6), n
        try:
            patchfile.apply(fresh, tmp / "patch", "other")
            raise AssertionError("should refuse a different base")
        except ValueError:
            pass
        manifest = json.loads((tmp / "patch" / "manifest.json").read_text())
        print(f"patch ok: {changed} weights, apply / remove / wrong-base refusal all work ({manifest['format']})")

        # revert all restores everything exactly
        weights.revert("all")
        assert all(torch.equal(original[n], p) for n, p in model.named_parameters())
        print("revert-all ok")

    # CLI: clear error without a key
    import os
    os.environ.pop("ANTHROPIC_API_KEY", None)
    assert main(["login"]) == 1
    print("cli ok")
    test_editor_cli()
    print("\nALL OFFLINE TESTS PASSED")


if __name__ == "__main__":
    main_test()
