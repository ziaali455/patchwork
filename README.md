# Patchwork

Patchwork is an experimental CLI where Claude edits the weights of Qwen3.5-0.8B directly from a plain-language goal, with no training and no gradients. Claude investigates the model, makes small targeted edits, tests them, and keeps only what helps. The result is a tiny, shareable **patch** plus a before-and-after benchmark.

It follows [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (MIT CSAIL, Nov 2026), where an LLM fixed Qwen3.5-0.8B's reasoning loops by editing 0.016% of its weights.

## How it works

Claude never runs code on your machine. It can only call these tools:

| Tool | What it does | Cost |
|---|---|---|
| `inspect_columns`, `align_columns` | Read MLP down-projection weights; find columns aligned with a direction | free |
| `run_practice` | Qwen answers practice problems; Claude sees accuracy, which answers loop, and how they end | 1 forward pass |
| `find_direction` | Mean activation difference between two groups of texts (e.g. looping vs finished) at a layer | 1 forward pass |
| `apply_edit` | `scale` columns, or `project_out` a direction (measured, or an existing column) | applied immediately |
| `revert_edit` | Undo the last edit or all edits, exactly | free |

Forward passes are the scarce resource. In the paper, 10–25 barely helped and 50 is where the fix landed, so the default budget is 50. Every edit is validated (finite values, no overlaps, a parameter budget of 0.1% of weights by default).

Practice uses the GSM8K **train** split (or your `--practice` file), and evaluation uses the **test** split (or your `--eval` file), so Claude never sees the benchmark. Practice prompts and Qwen's practice outputs are sent to Claude; evaluation data stays local.

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

## What a run produces

```
runs/loops/
  patch.safetensors   only the changed weights (flat indices + float32 deltas)
  manifest.json       goal, base revision + sha256 fingerprint, edits, stats
  report.json         settings, Claude's summary, token usage, per-case before/after
  transcript.jsonl    every Claude message and tool call, for auditing
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
uv run --extra local python tests/test_offline.py   # tiny random model + scripted fake Claude, no downloads
```
# patchwork
