import argparse
import json
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

from . import claude
from .model import QWEN

SHORTCUTS = {
    "opus": "claude-opus-5-5",
    "sonnet": "claude-sonnet-5-5",
    "haiku": "claude-haiku-5-5",
    "fable": "claude-fable-5-1",
}


def _progress(label):
    def show(i, n):
        print(f"\r  {label} {i}/{n}", end="\n" if i == n else "", flush=True)
    return show


def _load(args):
    from .benchmark import Generator
    from .model import checkpoint_for, fingerprint, load

    checkpoint = checkpoint_for(args.revision)
    print(f"Loading {QWEN}@{checkpoint.name[:10]} on {args.device}…", flush=True)
    model, tokenizer = load(checkpoint, args.device)
    generator = Generator(model, tokenizer, max_new_tokens=args.max_new_tokens, thinking=args.thinking)
    return checkpoint, fingerprint(checkpoint), model, tokenizer, generator


def _settings(args):
    return {"eval": str(args.eval), "max_new_tokens": args.max_new_tokens, "thinking": args.thinking,
            "device": args.device, "decoding": "greedy", "dtype": "float32",
            "torch": version("torch"), "transformers": version("transformers")}


def _print_compare(before, after):
    print("\n               before    after")
    print(f"  accuracy     {before['accuracy']:6.1%}   {after['accuracy']:6.1%}")
    print(f"  hit limit    {before['reached_token_limit']:4}/{before['total']:<3} {after['reached_token_limit']:4}/{after['total']}")
    print(f"  sec/case     {before['mean_generation_seconds']:6.2f}   {after['mean_generation_seconds']:6.2f}\n")


def patch(args):
    from .benchmark import evaluate, read_cases, read_practice
    from .patchfile import build, write
    from .probe import Probe
    from .weights import Weights

    if not args.prompt.strip() or args.max_new_tokens < 1 or args.passes < 1:
        raise ValueError("Provide a goal, a positive token limit and a positive pass budget.")
    if args.output.exists() or args.output.is_symlink():
        raise ValueError(f"Output already exists: {args.output}")
    claude_model = SHORTCUTS.get(args.claude_model, args.claude_model)
    cases = read_cases(args.eval)
    practice = read_practice(args.practice)

    with claude.client() as api:
        api.models.retrieve(claude_model)  # fail fast on a bad model ID or key
        checkpoint, fp, model, tokenizer, generator = _load(args)
        weights = Weights(model, budget_fraction=args.param_budget)
        print(f"Benchmarking baseline on {len(cases)} held-out cases…", flush=True)
        before = evaluate(generator, cases, progress=_progress("baseline"))
        print(f"Baseline: {before['accuracy']:.1%} correct, {before['reached_token_limit']}/{before['total']} hit the token limit")

        args.output.mkdir(parents=True)
        probe = Probe(model, tokenizer, generator, weights, practice, budget=args.passes)
        print(f"\nEditing session with {claude_model} ({args.passes} forward passes)…", flush=True)
        finished, usage = claude.session(
            api, claude_model, args.prompt, f"{QWEN}@{checkpoint.name}", weights, probe,
            log_path=args.output / "transcript.jsonl", max_turns=args.max_turns, token_budget=args.token_budget,
        )
    print(f"\nSession used {usage['input_tokens']:,} input + {usage['output_tokens']:,} output tokens, "
          f"{usage['turns']} turns, {probe.used}/{args.passes} forward passes.")

    report = {"base_model": QWEN, "base_revision": checkpoint.name, "claude_model": claude_model,
              "goal": args.prompt, "claude_summary": finished, "usage": usage,
              "forward_passes": probe.used, "edits": weights.summary(), "evaluation": _settings(args),
              "before": before}
    if not weights.history:
        report["after"] = None
        (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("No edits were kept, so there is no patch. Details in report.json.")
        return

    tensors, changed = build(weights)
    total = sum(p.numel() for p in model.parameters())
    write(args.output, tensors, {
        "goal": args.prompt, "base": {"repo": QWEN, "revision": checkpoint.name, "fingerprint": fp},
        "claude_model": claude_model, "edits": weights.summary(),
        "parameters_changed": changed, "fraction_changed": changed / total,
    })
    print(f"Benchmarking the patched model ({changed:,} weights, {changed / total:.4%})…", flush=True)
    after = evaluate(generator, cases, progress=_progress("patched"))
    report.update(after=after, parameters_changed=changed)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print_compare(before, after)
    print(f"Saved patch to {args.output} (apply with: patchwork eval --patch {args.output})")


def evaluate_cmd(args):
    from .benchmark import evaluate, read_cases
    from .patchfile import apply

    cases = read_cases(args.eval)
    _, fp, model, _, generator = _load(args)
    if args.patch:
        print(f"Applied {apply(model, args.patch, fp):,} weight changes from {args.patch}")
    result = evaluate(generator, cases, progress=_progress("eval"))
    print(f"{result['accuracy']:.1%} correct ({result['correct']}/{result['total']}), "
          f"{result['reached_token_limit']} hit the token limit, {result['mean_generation_seconds']:.2f}s per case")
    if args.json:
        args.json.write_text(json.dumps({"settings": _settings(args), **result}, indent=2) + "\n", encoding="utf-8")


def export(args):
    from .model import checkpoint_for, fingerprint, load
    from .patchfile import apply

    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}")
    checkpoint = checkpoint_for(args.revision)
    model, tokenizer = load(checkpoint, "cpu")
    apply(model, args.patch, fingerprint(checkpoint))
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    for extra in checkpoint.glob("*.json"):
        if not (args.output / extra.name).exists():
            shutil.copy(extra, args.output / extra.name)
    print(f"Wrote a full patched checkpoint to {args.output}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="patchwork", description="Claude edits Qwen3.5-0.8B's weights directly, no training.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("login", help="Verify your ANTHROPIC_API_KEY.")
    commands.add_parser("models", help="List Claude model IDs available to your key.")
    download = commands.add_parser("download", help="Download the Qwen checkpoint.")
    download.add_argument("--revision", default="main")

    def model_options(p):
        p.add_argument("--revision", default="main", help="Downloaded Hugging Face branch or commit.")
        p.add_argument("--eval", default="gsm8k", help="Eval JSON file, or gsm8k / gsm8k:N (default: 30 GSM8K test problems).")
        p.add_argument("--max-new-tokens", type=int, default=1024)
        p.add_argument("--thinking", action="store_true", help="Enable Qwen's reasoning mode (where the loops happen).")
        p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")

    run = commands.add_parser("patch", help="Let Claude edit the weights, then benchmark before and after.")
    run.add_argument("--prompt", required=True, help="The behavior change you want, in plain language.")
    run.add_argument("--claude-model", default="sonnet", help="opus | sonnet | haiku | fable, or a full model ID.")
    run.add_argument("--output", required=True, type=Path, help="New directory for the patch and report.")
    run.add_argument("--practice", default=None, help="Practice JSON file Claude may test on (default: GSM8K train).")
    run.add_argument("--passes", type=int, default=50, help="Forward-pass budget for Claude (default 50).")
    run.add_argument("--param-budget", type=float, default=0.001, help="Max fraction of weights touched (default 0.1%%).")
    run.add_argument("--max-turns", type=int, default=60)
    run.add_argument("--token-budget", type=int, default=3_000_000, help="Stop after this many Claude tokens.")
    model_options(run)

    ev = commands.add_parser("eval", help="Benchmark the base model, or a patched one.")
    ev.add_argument("--patch", type=Path, default=None)
    ev.add_argument("--json", type=Path, default=None, help="Also write full results here.")
    model_options(ev)

    ex = commands.add_parser("export", help="Write a full checkpoint with a patch applied.")
    ex.add_argument("--patch", type=Path, required=True)
    ex.add_argument("--output", type=Path, required=True)
    ex.add_argument("--revision", default="main")

    args = parser.parse_args(argv)
    try:
        if args.command in ("login", "models"):
            with claude.client() as api:
                models = api.models.list()
                if args.command == "login":
                    print("Anthropic API key verified.")
                else:
                    for model in models:
                        print(f"{model.id}\t{model.display_name}")
            return 0
        try:
            from huggingface_hub.errors import LocalEntryNotFoundError
        except ImportError:
            raise ValueError("Install local model support with: uv tool install '.[local]'") from None
        try:
            if args.command == "download":
                from .model import fingerprint, snapshot

                path = snapshot(revision=args.revision, local_only=False)
                print(f"{path}\nfingerprint {fingerprint(path)[:16]}")
            else:
                {"patch": patch, "eval": evaluate_cmd, "export": export}[args.command](args)
        except LocalEntryNotFoundError:
            raise ValueError(f"Download the checkpoint first: patchwork download --revision {args.revision}") from None
    except Exception as exc:  # noqa: BLE001 - one clean error line for the user
        print(f"patchwork: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
