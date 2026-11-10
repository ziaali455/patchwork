# Patchwork

Patchwork is an experimental CLI that lets an editor model change the weights of a smaller local model from a plain-language goal, without training or gradients. The editor investigates the target, tries small weight edits, and uses practice results to decide what to keep. Patchwork saves the remaining changes as a sparse **patch** and reports the before-and-after benchmark. The defaults are Claude Sonnet and Qwen3.5-0.8B.

It follows [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (MIT CSAIL, Nov 2026), where an LLM fixed Qwen3.5-0.8B's reasoning loops by editing 0.016% of its weights.

## How it works

The editor can call the following local tools. Patchwork validates and executes each call.

| Tool | What it does | Cost |
|---|---|---|
| `inspect_columns`, `align_columns` | Read MLP down-projection weights; find columns aligned with a direction | free |
| `run_practice` | The target answers practice problems and the editor sees accuracy, token-limit hits, and output endings | 1 forward pass |
| `find_direction` | Mean activation difference between two groups of texts (e.g. looping vs finished) at a layer | 1 forward pass |
| `apply_edit` | `scale` columns, or `project_out` a direction (measured, or an existing column) | applied immediately |
| `revert_edit` | Undo the last edit or all edits, exactly | free |

Forward passes are the scarce resource. In the paper, 10–25 barely helped and 50 is where the fix landed, so the default budget is 50. Every edit is validated (finite values, no overlaps, a parameter budget of 0.1% of weights by default).

Practice uses the GSM8K **train** split (or your `--practice` file). Evaluation uses the **test** split (or your `--eval` file). The editor receives the goal, tensor catalog, and tool results, including practice output excerpts. Evaluation data stays local.

## Install

With Python 3.11–3.14 and [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/ziaali455/patchwork
cd patchwork
uv tool install '.[local]'
patchwork --help
```

The default device is CPU. Pass `--device mps` on Apple Silicon or `--device cuda` with a CUDA build of PyTorch.

## Run your first patch

1. Set `ANTHROPIC_API_KEY` to a key from the [Claude Console](https://platform.claude.com/). Requests use [API billing](https://platform.claude.com/docs/en/about-claude/pricing). Anthropic doesn't allow third-party tools to use Claude.ai subscription logins ([policy](https://code.claude.com/docs/en/legal-and-compliance)).

   ```sh
   patchwork login
   ```

2. Download the model and measure the baseline. This costs nothing and tells you whether Qwen loops on your machine.

   ```sh
   patchwork download
   patchwork eval --thinking --eval gsm8k:10
   ```

3. Patch. Start small, then raise `--passes` toward 50.

   ```sh
   patchwork patch \
     --prompt "Stop getting stuck in reasoning loops so math answers finish." \
     --claude-model sonnet \
     --thinking \
     --passes 10 \
     --output runs/loops
   ```

   `--claude-model` takes `opus`, `sonnet`, `haiku`, `fable`, or any ID from `patchwork models`. Fable has extra safeguards around LLM R&D and may decline.

4. Use the patch.

   ```sh
   patchwork eval --patch runs/loops --thinking           # re-benchmark
   patchwork export --patch runs/loops --output qwen-fixed  # full checkpoint for other tools
   ```

## Choose a different editor and target

Use `--target-model` with a Hugging Face repository ID or a local Transformers checkpoint directory. The download command fills the local cache for Hugging Face targets. Use the same ID when creating the patch.

```sh
patchwork download --target-model HuggingFaceTB/SmolLM2-135M-Instruct

# Set OPENAI_API_KEY in your shell before using the OpenAI editor.
patchwork patch \
  --target-model HuggingFaceTB/SmolLM2-135M-Instruct \
  --editor-provider openai --editor-model gpt-6-astra \
  --prompt "Improve accuracy on short arithmetic problems." \
  --eval gsm8k:5 --passes 3 --output runs/smollm

patchwork eval --patch runs/smollm --eval gsm8k:5
```

| `--editor-provider` | Model and credentials |
|---|---|
| `anthropic` (default) | Uses the Anthropic Messages API and `ANTHROPIC_API_KEY`. Accepts an Anthropic model ID or the existing shortcuts with `--editor-model` or `--claude-model`. |
| `openai` | Uses the OpenAI Responses API and `OPENAI_API_KEY`. Pass a model ID that supports function calling with `--editor-model`. |
| `openai-compatible` | Uses a Chat Completions API with function calling. Pass `--editor-model` and `--editor-base-url` (for example, `http://localhost:8000/v1`). For authenticated endpoints, set a key in your shell and pass its variable name with `--editor-key-env`. |

The target must be a text-generation checkpoint that Transformers can load with built-in model code and editable floating-point weights. Patchwork detects common feed-forward output projections, including `down_proj`, GPT-2's `c_proj`, GPT-NeoX's `dense_4h_to_h`, and OPT's `fc2`. For a different layout, pass `--edit-pattern` with a full tensor-name regex containing a zero-based `(?P<layer>...)` group. For example, `'model\.layers\.(?P<layer>\d+)\.ffn\.output\.weight'`. GGUF, quantized-only checkpoints, and models that require remote Python code are outside the supported scope.

Evaluation and export infer the base target and revision from the patch. For a local checkpoint moved to another directory, use `--target-model /path/to/checkpoint` to override the recorded path. The fingerprint still has to match.

## What a run produces

```
runs/loops/
  patch.safetensors   only the changed weights (flat indices + float32 deltas)
  manifest.json       goal, base revision + sha256 fingerprint, edits, stats
  report.json         settings, editor summary, token usage, per-case before/after
  transcript.jsonl    editor messages and tool calls, for auditing
```

A patch applies only to the exact checkpoint it was made from (checked by fingerprint), and `apply` with the deltas negated removes it. Patches are KB to a few MB instead of a 3 GB checkpoint, which makes them easy to share.

## Custom goals

Bring your own data: `--practice` and `--eval` take JSON arrays of `{"prompt": ..., "answer": ...}`. Numeric answers are compared by value (preferring the last `\boxed{}`); text answers by case-insensitive exact match. Keep the two files disjoint. See `examples/math-smoke.json`.

## Notes

- Decoding is greedy, so before and after differ only because of the edit. Compare runs with identical device, token limit, thinking mode, and eval data.
- CPU generation is slow. With `--thinking`, answers are long; the paper allowed 32k tokens, the default here is 1,024, which counts long-but-fine answers as hitting the limit.
- Weight editing removes obstacles more readily than it adds abilities. In the paper, it fixed Qwen's loops but couldn't improve a model that didn't loop.

## Development

```sh
uv run --extra local python tests/test_offline.py   # tiny random models + fake editors, no model downloads
```
# patchwork
