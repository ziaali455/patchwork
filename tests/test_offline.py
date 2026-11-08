"""Offline end-to-end test: a tiny random Qwen-style model and a scripted fake Claude.

Covers scoring, greedy generation, every Claude tool (including error paths),
exact revert, and the sparse patch format (write, apply, remove, wrong base).
No downloads, no API key.  Run:  uv run --extra local python tests/test_offline.py
"""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from patchwork import claude, patchfile  # noqa: E402
from patchwork.benchmark import Generator, evaluate, is_correct  # noqa: E402
from patchwork.cli import main  # noqa: E402
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
    print("\nALL OFFLINE TESTS PASSED")


if __name__ == "__main__":
    main_test()
