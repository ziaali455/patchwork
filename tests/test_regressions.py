"""Regression coverage for exact patches, validation state and direct assignments."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from test_offline import tiny_model, DOWN
from patchwork import patchfile
from patchwork.weights import Weights
from patchwork.probe import Probe
from patchwork.benchmark import Generator


def run():
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        model, _ = tiny_model()
        model.to(dtype)
        weights = Weights(model, budget_fraction=1)
        with torch.no_grad():
            weights.tensors[DOWN][:, 0] = 1
        original = weights.tensors[DOWN].clone()
        weights.apply(dict(tensor=DOWN, columns=[0], operation='set', values=[[0.001]] * 32))
        expected = weights.tensors[DOWN].clone()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            tensors, _ = patchfile.build(weights)
            patchfile.write(directory, tensors, {'base': {'fingerprint': 'test'}})
            weights.revert('all')
            patchfile.apply(model, directory, 'test')
            assert torch.equal(weights.tensors[DOWN], expected)
            patchfile.apply(model, directory, 'test', -1)
            assert torch.equal(weights.tensors[DOWN], original)
            # A malformed second entry must leave the first entry unapplied.
            tensors['unknown::idx'] = torch.tensor([0])
            tensors['unknown::before'] = torch.tensor([0.], dtype=dtype)
            tensors['unknown::after'] = torch.tensor([1.], dtype=dtype)
            save_file(tensors, directory / 'patch.safetensors')
            try:
                patchfile.apply(model, directory, 'test')
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid patch accepted')
            assert torch.equal(weights.tensors[DOWN], original)

    # Version 1 remains loadable.
    model, tokenizer = tiny_model()
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        (directory / 'manifest.json').write_text(json.dumps({'format': patchfile.LEGACY_FORMAT, 'base': {'fingerprint': 'legacy'}}))
        save_file({DOWN + '::idx': torch.tensor([0]), DOWN + '::delta': torch.tensor([0.5])}, directory / 'patch.safetensors')
        prior = dict(model.named_parameters())[DOWN].flatten()[0].item()
        patchfile.apply(model, directory, 'legacy')
        assert abs(dict(model.named_parameters())[DOWN].flatten()[0].item() - prior - 0.5) < 1e-7

    from patchwork.cli import main
    for flags, expected_limit in (([], 8192), (["--max-new-tokens", "32768"], 32768)):
        with patch("patchwork.cli.evaluate_cmd") as evaluate_cli:
            assert main(["eval", *flags]) == 0
            assert evaluate_cli.call_args.args[0].max_new_tokens == expected_limit

    from transformers import GPT2Config, GPT2LMHeadModel
    biased = GPT2LMHeadModel(GPT2Config(n_embd=8, n_layer=1, n_head=1, n_inner=16))
    bias_weights = Weights(biased, budget_fraction=1)
    bias_name = "transformer.h.0.mlp.c_proj.bias"
    before_bias = bias_weights.tensors[bias_name].clone()
    bias_weights.apply(dict(tensor=bias_name, columns=[0, 2], operation="set", values=[[2, 3]]))
    assert bias_weights.tensors[bias_name][0, 2] == 3
    bias_weights.revert()
    assert torch.equal(before_bias, bias_weights.tensors[bias_name])

    weights = Weights(model, budget_fraction=1)
    probe = Probe(model, tokenizer, lambda prompt: dict(answer='1', reached_token_limit=False,
                  generated_tokens=1, raw_output='1'), weights,
                  [{'prompt': 'a', 'answer': '1'}, {'prompt': 'b', 'answer': '1'}], 10)
    probe.run_practice({'ids': [0]})
    edit = dict(tensor=DOWN, columns=[0], operation='set', values=[[0.0]] * 32)
    weights.apply(edit)
    assert probe.validation()['status'] == 'untested'
    probe.run_practice({'ids': [1]})
    assert probe.validation()['status'] == 'tested_without_matching_baseline'
    probe.run_practice({'ids': [0]})
    assert probe.validation()['status'] == 'paired'
    weights.apply({**edit, 'columns': [1]})
    assert probe.validation()['status'] == 'untested'
    weights.revert()
    assert probe.validation()['status'] == 'paired'
    weights.revert()
    weights.apply(edit)
    assert probe.validation()['status'] == 'untested'

    generator = Generator(model, tokenizer, max_new_tokens=5, thinking=False,
                          temperature=0.6, presence_penalty=1.5, seed=42)
    assert generator('abc')['raw_output'] == generator('abc')['raw_output']
    assert generator.config.do_sample
    # Penalize each generated token once and exclude prompt-only tokens.
    def check_penalty(input_ids, generation_config, logits_processor, **kwargs):
        scores = torch.zeros((1, model.config.vocab_size))
        assert torch.equal(logits_processor(input_ids, scores.clone()), scores)
        extended = torch.cat([input_ids, torch.tensor([[3]])], dim=1)
        result = logits_processor(extended, scores.clone())
        assert result[0, 3] == -1.5 and (result != 0).sum() == 1
        extended = torch.cat([extended, torch.tensor([[3]])], dim=1)
        assert torch.equal(logits_processor(extended, scores.clone()), result)
        extended = torch.cat([extended, torch.tensor([[4]])], dim=1)
        result = logits_processor(extended, scores.clone())
        assert result[0, 4] == -1.5 and (result != 0).sum() == 2
        return extended

    with patch.object(model, "generate", side_effect=check_penalty):
        generator('abc')
    with tempfile.TemporaryDirectory() as directory:
        from patchwork.model import load
        directory = Path(directory)
        model.save_pretrained(directory)
        tokenizer.save_pretrained(directory)
        half_model, _ = load(directory, 'cpu', 'float16')
        assert next(half_model.parameters()).dtype == torch.float16
    for limit in (0, -1):
        try:
            Generator(model, tokenizer, max_new_tokens=limit, thinking=False)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid token cap accepted')
    print('Exact replay/reversal, atomic validation, legacy patches, state tracking and sampling passed.')


if __name__ == '__main__':
    run()
