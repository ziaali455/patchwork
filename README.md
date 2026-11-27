# Patchwork

Patchwork is an experimental CLI that lets an editor model change the weights of a smaller local model from a plain-language goal, without training or gradients. The editor investigates the target, tries small weight edits, and uses practice results to decide what to keep. Patchwork saves the remaining changes as a sparse **patch** and reports the before-and-after benchmark. The defaults are Claude Sonnet and Qwen3.5-0.8B.

It follows [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (MIT CSAIL, Nov 2026), where an LLM fixed Qwen3.5-0.8B's reasoning loops by editing 0.016% of its weights.

## How it works

The editor can call the following local tools. Patchwork validates and executes each call.

| Tool | What it does | Cost |
|---|---|---|
| `inspect_columns`, `align_columns` | Read MLP output weights and biases; find columns aligned with a direction | free |
| `run_practice` | The target answers practice problems and the editor sees accuracy, token-limit hits, and output endings | 1 forward pass |
| `find_direction` | Mean activation difference between two groups of texts (e.g. looping vs finished) at a layer | 1 forward pass |
| `apply_edit` | `set` explicit column values, `scale` columns, or `project_out` a direction (measured, or an existing column) | applied immediately |
| `revert_edit` | Undo the last edit or all edits, exactly | free |

Forward passes are the scarce resource. In the paper, 10–25 barely helped and 50 is where the fix landed, so the default budget is 50. An initial practice baseline uses one pass. Subsequent default practice calls use the same problem IDs for paired comparisons. Every edit is validated (finite values, no overlaps, a parameter budget of 0.1% of weights by default).

Practice uses the GSM8K **train** split (or your `--practice` file). Evaluation uses the **test** split (or your `--eval` file). The editor receives the goal, tensor catalog, and tool results, including practice output excerpts. Evaluation data stays local.

## Install

With Python 3.11–3.14 and [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/ziaali455/patchwork
cd patchwork
uv tool install '.[local]'
patchwork --help
```

The default device is CPU. Pass `--device mps --dtype float16` on memory-constrained Apple Silicon machines or `--device cuda` with a CUDA build of PyTorch.

## Desktop app

From the repository root, with Node.js 22.12 or newer:

```sh
uv sync --extra local
npm --prefix desktop ci
npm --prefix desktop run dev
```

**Run configuration → Target model** lists compatible-looking snapshots from the Hugging Face cache at startup. Select one to verify and register it locally. Open **Models** to download a new checkpoint or import a standalone Transformers directory. Use **Playground** to prompt a target and compare it with any committed version of that checkpoint. In **Editor settings**, choose Anthropic or OpenAI. The app reads `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` from the environment or the repository's `.env`. You can also enter a key in the app, which uses the OS credential encryption available to Electron.

In **Editor settings**, click **Load models** to read the provider's catalog with your configured or entered key. Choose a text model that supports function calling, then save the connection. You can also enter the exact model ID manually.

Start an edit from **Edit**, then open its run in **History** to review the benchmark and write a commit message. Commits and run artifacts live in the app's local user-data folder. Each committed patch is based on a pinned checkpoint and can be compared in the playground or exported as a full Transformers checkpoint from History. The desktop workflow creates edits from base checkpoints.

Run `npm --prefix desktop run build` to type-check and build the app, then `npm --prefix desktop start` to launch that build. Both modes use the repository's `.venv`. Set `PATCHWORK_PYTHON` to an absolute interpreter path to use a different Patchwork environment.

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
  patch.safetensors   changed indices with exact original and replacement values
  manifest.json       goal, base revision + sha256 fingerprint, edits, stats
  report.json         settings, editor summary, token usage, per-case before/after
  transcript.jsonl    editor messages and tool calls, for auditing
```

New patches store exact original and replacement values, so applying and removing them preserves the evaluated weights exactly. Application checks the checkpoint fingerprint and current source values, and validates all entries before changing any tensor. Older version 1 delta patches remain readable, with their original rounding limitations. Patches are KB to a few MB instead of a 3 GB checkpoint, which makes them easy to share.

## Custom goals

Bring your own data: `--practice` and `--eval` take JSON arrays of `{"prompt": ..., "answer": ...}`. Numeric answers are compared by value (preferring the last `\boxed{}`); text answers by case-insensitive exact match. Keep the two files disjoint. See `examples/math-smoke.json`.

## Notes

- Decoding defaults to greedy. Set `--temperature` above zero to enable sampling; `--top-p`, `--top-k`, `--presence-penalty` and `--seed` control sampling. Seeds are derived per prompt for paired comparisons. Reports record these settings. Compare runs on the same device and software versions.
- CPU generation is slow. With `--thinking`, answers are long; the paper allowed 32k tokens, the default here is 8,192; `--max-new-tokens` overrides it for both `patch` and `eval`. A shorter cap can count valid long reasoning as hitting the limit.
- Weight editing removes obstacles more readily than it adds abilities. In the paper, it fixed Qwen's loops but couldn't improve a model that didn't loop.

## Development

```sh
uv run --extra local python tests/test_offline.py   # tiny random models + fake editors, no model downloads
```
# patchwork

## Output token limits and sampling

`--max-new-tokens` limits each target answer, including thinking tokens. It is separate from `--token-budget`, which limits editor usage. The default is 8,192 tokens.

```sh
patchwork eval --thinking --max-new-tokens 32768 --eval gsm8k:10
patchwork patch --prompt "Reduce reasoning loops" --thinking \
  --max-new-tokens 32768 --temperature 0.6 --top-p 0.95 --top-k 20 \
  --presence-penalty 1.5 --seed 0 --output runs/sampled
```

`report.json` records `practice_validation` for the retained weight state. Its status is `unmodified`, `untested`, `tested_without_matching_baseline`, or `paired`. Paired results include before/after accuracy and token-limit counts on identical practice IDs. This evidence is separate from the editor's `helped` claim and the final held-out benchmark. A paired comparison can show regression or no improvement. Edits retained after a turn or token budget ends are explicitly marked untested when appropriate.

Direct assignments use `apply_edit` with `operation: "set"`, selected columns and a two-dimensional `values` array shaped `[rows, selected_columns]`. Biases are exposed as one-row matrices. Direct assignments obey the same touched-parameter budget and exact revert behavior as other edits.

## Breakout networks from prompts

Install the optional game dependencies with `uv pip install -e '.[local,breakout]'` in your environment, or `uv tool install --force '.[local,breakout]'` for the CLI installation.

```sh
patchwork breakout prompt > breakout-prompt.txt
```

Give that prompt to an editor in a fresh chat with tools disabled. Save the response as `attempt.md`, including its fenced `edits` block. The CLI imports the numbers into a zero-initialized 600 → 32 → 32 → 6 network and runs it locally.

```sh
patchwork breakout apply attempt.md attempt.npz
patchwork breakout evaluate --weights zero attempt.npz --episodes 100 --output result.json
```

To compare the published checkpoint, supply its location explicitly:

```sh
patchwork breakout evaluate --weights zero reference attempt.npz \
  --reference ../wsp-local/lm_understand-logs/exp5_write_from_scratch/networks/breakout/hand.npz \
  --episodes 100 --output comparison.json
```

For a revision, save the editor's incremental response and use `breakout apply revision.md revision.npz --base attempt.npz`. Omit `--base` for a full replacement from zeros. Use fresh evaluation seeds after selecting a candidate, for example `--episodes 1000 --seed 10000`.

Breakout uses sampled actions, 0.1 sticky actions and a 10,000-step cap. Reports include episode seeds, scores, standard errors and checkpoint hashes. Local RNG choices differ from the publication. Invalid edits reject the entire import. Breakout saves NumPy network checkpoints and accepts either provider API responses or manually supplied responses through `breakout run --editor-mode`. The language-model editing loop saves sparse patch files.

The bundled prompt comes from Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab's [Experiment 5](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp5_write_from_scratch), licensed CC BY 4.0. The MinAtar source embedded in it is by Kenny Young and Tian Tian under GPL-3.0; its license is bundled alongside the prompt.

Additional regression checks: `uv run --extra local python tests/test_regressions.py`.

## Choose API or manual Breakout editing

Use `breakout run --editor-mode api` to request weights from a provider and evaluate them automatically:

```sh
patchwork breakout run --editor-mode api \
  --editor-provider openai --editor-model gpt-6-astra \
  --episodes 100 --output runs/breakout-api
```

Provider credentials and endpoint flags match `patchwork patch`. Anthropic defaults to Sonnet when `--editor-model` is omitted. OpenAI and OpenAI-compatible endpoints require a model ID. `--editor-max-tokens` controls the editor response length (default 16,384).

The default `--rounds 1` requests a single response with no tools or gameplay feedback. Set `--rounds 3` for three attempts, with aggregate practice results supplied after each of the first two. Each response specifies a complete network from zeros. Patchwork keeps the candidate with the highest mean practice reward, including the zero-weight baseline as a candidate.

For a response obtained in a chat, choose manual mode:

```sh
patchwork breakout run --editor-mode manual --reply attempt.md \
  --episodes 100 --output runs/breakout-manual
```

Manual mode makes no provider calls. Repeat `--reply` to compare several complete candidate responses. For incremental edits, use the existing `breakout apply --base` command. `--notes` records provenance, such as exposure to the published solution.

Both modes save the prompt, responses, candidate `.npz` files, `selected.npz`, and `report.json`. The report includes practice scores, selection, provider token usage, and final before/after scores on a disjoint seed range. `--seed` defaults to 0 for practice and `--eval-seed` to 10,000 for final evaluation; `--episodes` controls both sample sizes. Final evaluation results are kept out of editor feedback. Existing output directories are rejected. Invalid or refused API responses leave an explicitly failed report and the received response for inspection.

Provider integration tests use simulated responses: `uv run --extra local --extra breakout python tests/test_breakout_editors.py`.

`--dtype` accepts `float32` (default), `float16`, or `bfloat16`. Patches record the selected dtype, and `eval --patch` and `export --patch` infer it. Use a dtype supported by your device. Float16 reduces weight memory use on an M1 Mac.

## Repair the Kinetix car policy

The integrated car workflow uses the tested Kinetix commit and the public rounded actor from Experiment 4. Install its optional dependencies in a Python 3.12 environment:

```sh
uv venv --python 3.12
uv pip install -e '.[car]'
```

Request a repair and one feedback-driven revision:

```sh
uv run --python 3.12 --extra car patchwork car run \
  --editor-mode api --editor-provider openai --editor-model gpt-6-astra \
  --rounds 2 --practice-episodes 64 --eval-episodes 1024 \
  --output runs/car-repair
```

OpenAI uses streamed Responses requests with high reasoning effort by default, configurable with `--reasoning-effort`. All providers use a 20-minute request timeout for this large prompt. `--editor-max-tokens` defaults to 16,384. Other supported editor providers use the same credential and endpoint flags as Breakout. Export your API key into the process environment before running.

The first attempt receives the original task and trained weights. Each additional round receives practice scores and a recorded trajectory from its preceding candidate. Every response supplies a complete set of edits against the original rounded actor. The highest practice success rate determines the selected actor, with ties retaining the earlier candidate. Fresh evaluation seeds are used only after selection. Defaults are practice seeds starting at 30,000 and evaluation seeds starting at 40,000.

Import responses obtained from a chat:

```sh
patchwork car prompt > car-prompt.txt
patchwork car run --editor-mode manual --reply first.md --reply revision.md \
  --practice-episodes 64 --eval-episodes 1024 --output runs/car-manual
```

Manual mode makes no API requests. Each reply in `car run` describes edits relative to the original trained actor. For incremental edits to a saved checkpoint, use `car apply --base`:

```sh
patchwork car apply revision.md revised.npz --base previous.npz
patchwork car evaluate --weights revised.npz --episodes 1024 --seed 50000 --output car-eval.json
```

Use `car run --base previous.npz` to start from an existing repair. In API mode, Patchwork describes its changed weights, measures its practice behavior, and supplies a recorded trajectory before requesting a revision. The supplied checkpoint remains a candidate during selection.

A run saves `base.npz`, `initial.npz`, each candidate response and actor, feedback files, `selected.npz`, and `report.json`. Reports record simulator versions, checkpoint hashes, token usage, seed ranges, practice selection and held-out success. Evaluation progress is saved after every batch, and interrupted runs are marked failed. Choose a new output directory for a retry. The actor starts with trained weights; unused edit entries retain their original values. Numerical responses are parsed as data and invalid edits reject the import.

The bundled prompt and actor are licensed CC BY 4.0, with attribution in `patchwork/data/car-attribution.txt`. The simulator is installed from its pinned upstream Git repository. CPU simulation uses float32 and sampled actions with a 256-step episode limit. Local RNG choices differ from the paper.

Car regression checks can be run with `python tests/test_car.py`. They cover manual and API orchestration with simulated providers. The simulator is verified separately against the released policies.
