import argparse
import json
import shlex
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

from . import claude, editor, providers
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

    checkpoint = checkpoint_for(args.revision, args.target_model)
    digest = fingerprint(checkpoint)
    if args.expect_fingerprint and digest != args.expect_fingerprint:
        raise ValueError("The checkpoint weights changed. Import the new checkpoint before using it.")
    print(f"Loading {args.target_model} ({checkpoint.name[:10]}) on {args.device}…", flush=True)
    model, tokenizer = load(checkpoint, args.device, args.dtype)
    generator = Generator(model, tokenizer, max_new_tokens=args.max_new_tokens, thinking=args.thinking,
                          temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                          presence_penalty=args.presence_penalty, seed=args.seed)
    return checkpoint, digest, model, tokenizer, generator


def _target(args):
    base = {}
    if args.command in ("eval", "export", "prompt") and args.patch:
        from .patchfile import read_manifest

        base = read_manifest(args.patch)["base"]
    saved_target = base.get("repo") or base.get("path")
    target = args.target_model or saved_target or QWEN
    saved_revision = base.get("revision") if target == saved_target else None
    return target, args.revision or saved_revision or "main"


def _settings(args):
    return {"eval": str(args.eval), "max_new_tokens": args.max_new_tokens, "thinking": args.thinking,
            "device": args.device, "decoding": "sampled" if args.temperature > 0 else "greedy",
            "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
            "presence_penalty": args.presence_penalty, "seed": args.seed, "dtype": args.dtype,
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
    provider = args.editor_provider
    editor_model = args.editor_model if args.editor_model is not None else args.claude_model
    if provider == "anthropic":
        if editor_model is None:
            editor_model = "sonnet"
        editor_model = SHORTCUTS.get(editor_model, editor_model)
    elif args.claude_model or editor_model is None:
        raise ValueError("Use --editor-model with the selected editor provider.")
    if not editor_model.strip():
        raise ValueError("--editor-model must be a nonempty model ID.")

    with providers.connect(provider, editor_model, base_url=args.editor_base_url,
                           key_env=args.editor_key_env) as connection:
        cases = read_cases(args.eval)
        practice = read_practice(args.practice)
        checkpoint, fp, model, tokenizer, generator = _load(args)
        weights = Weights(model, budget_fraction=args.param_budget, edit_pattern=args.edit_pattern)
        if Path(args.target_model).expanduser().is_dir():
            base = {"path": str(checkpoint), "fingerprint": fp}
        else:
            base = {"repo": args.target_model, "revision": checkpoint.name, "fingerprint": fp}
        print(f"Benchmarking baseline on {len(cases)} held-out cases…", flush=True)
        before = evaluate(generator, cases, progress=_progress("baseline"))
        print(f"Baseline: {before['accuracy']:.1%} correct, {before['reached_token_limit']}/{before['total']} hit the token limit")

        args.output.mkdir(parents=True)
        probe = Probe(model, tokenizer, generator, weights, practice, budget=args.passes)
        probe.run_practice({"n_random": 8})
        print(f"\nEditing session with {editor_model} ({args.passes} forward passes)…", flush=True)
        finished, usage = editor.session(
            connection, editor_model, args.prompt,
            {"target": base.get("repo", checkpoint.name), "revision": base.get("revision")},
            weights, probe,
            log_path=args.output / "transcript.jsonl", max_turns=args.max_turns, token_budget=args.token_budget,
        )
    print(f"\nSession used {usage['input_tokens']:,} input + {usage['output_tokens']:,} output tokens, "
          f"{usage['turns']} turns, {probe.used}/{args.passes} forward passes.")

    editor_info = {"editor_provider": provider, "editor_model": editor_model}
    if provider == "anthropic":
        editor_info["claude_model"] = editor_model
    report = {"base_model": base.get("repo") or base["path"], "base_revision": base.get("revision"),
              **editor_info, "goal": args.prompt, "editor_summary": finished, "usage": usage,
              "forward_passes": probe.used, "edits": weights.summary(), "evaluation": _settings(args),
              "before": before, "practice_validation": probe.validation()}
    if provider == "anthropic":
        report["claude_summary"] = finished
    if not weights.history:
        report["after"] = None
        (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("No edits were kept, so there is no patch. Details in report.json.")
        return

    print(f"Practice validation: {probe.validation()['status']}")
    tensors, changed = build(weights)
    total = sum(p.numel() for p in model.parameters())
    write(args.output, tensors, {
        "goal": args.prompt, "base": base, "dtype": args.dtype, **editor_info, "edits": weights.summary(),
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
    checkpoint = checkpoint_for(args.revision, args.target_model)
    model, tokenizer = load(checkpoint, "cpu", args.dtype)
    apply(model, args.patch, fingerprint(checkpoint))
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    weight_indexes = {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
    for extra in checkpoint.glob("*.json"):
        if extra.name not in weight_indexes and not (args.output / extra.name).exists():
            shutil.copy(extra, args.output / extra.name)
    print(f"Wrote a full patched checkpoint to {args.output}")


def prompt_cmd(args):
    from .patchfile import apply

    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    if not prompt.strip():
        raise ValueError("Provide a prompt.")
    _, fingerprint, model, _, generate = _load(args)
    result = {}
    if not args.patch or args.compare:
        print("Generating base response…", flush=True)
        result["base"] = generate(prompt)
    if args.patch:
        apply(model, args.patch, fingerprint)
        print("Generating patched response…", flush=True)
        result["patched"] = generate(prompt)
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.json:
        args.json.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("breakout", "car"):
        try:
            if argv[0] == "car":
                from .car import main as workflow_main
            else:
                from .breakout import main as workflow_main
            workflow_main(argv[1:])
            return 0
        except Exception as exc:
            print(f"patchwork: {exc}", file=sys.stderr)
            return 1
    parser = argparse.ArgumentParser(prog="patchwork", description="Use an editor model to change local model weights directly, without training.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("car", help="Repair and evaluate Kinetix car policies; use car --help.")
    commands.add_parser("breakout", help="Author, import and evaluate small Breakout networks; use breakout --help.")
    commands.add_parser("login", help="Verify your ANTHROPIC_API_KEY.")
    commands.add_parser("models", help="List Claude model IDs available to your key.")

    def checkpoint_options(p):
        p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), help="Target weight dtype (default: float32, or saved patch dtype).")
        p.add_argument("--target-model", help=f"Hugging Face repo ID or local checkpoint directory (default: {QWEN}, or the patch's base).")
        p.add_argument("--revision", help="Hugging Face branch or commit (default: main, or the patch's base).")

    download = commands.add_parser("download", help="Download the target checkpoint from Hugging Face.")
    checkpoint_options(download)
    download.add_argument("--json", type=Path, help="Write the resolved checkpoint identity here.")
    download.add_argument("--cached-only", action="store_true", help="Resolve the checkpoint without contacting Hugging Face.")

    cache = commands.add_parser("cache", help="List cached text-generation checkpoints as JSON.")
    cache.add_argument("--json", type=Path, help="Write the list to a file.")

    def model_options(p, *, benchmark=True, tokens=8192):
        checkpoint_options(p)
        p.add_argument("--expect-fingerprint", help="Require this SHA-256 checkpoint fingerprint.")
        if benchmark:
            p.add_argument("--eval", default="gsm8k", help="Eval JSON file, or gsm8k / gsm8k:N (default: 30 GSM8K test problems).")
        p.add_argument("--max-new-tokens", type=int, default=tokens, help=f"Maximum generated tokens per answer (default: {tokens}).")
        p.add_argument("--temperature", type=float, default=0.0, help="0 for greedy; positive values enable sampling.")
        p.add_argument("--top-p", type=float, default=0.95)
        p.add_argument("--top-k", type=int, default=20)
        p.add_argument("--presence-penalty", type=float, default=0.0)
        p.add_argument("--seed", type=int, default=0, help="Sampling seed; paired by prompt across before/after runs.")
        p.add_argument("--thinking", action="store_true", help="Enable reasoning mode if supported by the target chat template.")
        p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")

    run = commands.add_parser("patch", help="Let an editor change the weights, then benchmark before and after.")
    run.add_argument("--prompt", required=True, help="The behavior change you want, in plain language.")
    run.add_argument("--editor-provider", choices=providers.PROVIDERS, default="anthropic")
    editor_models = run.add_mutually_exclusive_group()
    editor_models.add_argument("--editor-model", help="Model ID from the editor provider (Anthropic default: sonnet).")
    editor_models.add_argument("--claude-model", help="Legacy alias for --editor-model with Anthropic.")
    run.add_argument("--editor-base-url", help="API root for openai-compatible endpoints, including the API version path.")
    run.add_argument("--editor-key-env", help="Environment variable containing the editor API key.")
    run.add_argument("--edit-pattern", help="Override editable tensor discovery with a full-match regex containing a named layer group.")
    run.add_argument("--output", required=True, type=Path, help="New directory for the patch and report.")
    run.add_argument("--practice", default=None, help="Practice JSON file for the editor (default: GSM8K train).")
    run.add_argument("--passes", type=int, default=50, help="Forward-pass budget for the editor (default 50).")
    run.add_argument("--param-budget", type=float, default=0.001, help="Max fraction of weights touched (default 0.1%%).")
    run.add_argument("--max-turns", type=int, default=60)
    run.add_argument("--token-budget", type=int, default=3_000_000, help="Stop after this many editor tokens, if reported by the provider.")
    model_options(run)

    ev = commands.add_parser("eval", help="Benchmark the base model, or a patched one.")
    ev.add_argument("--patch", type=Path, default=None)
    ev.add_argument("--json", type=Path, default=None, help="Also write full results here.")
    model_options(ev)

    prompt = commands.add_parser("prompt", help="Generate an answer from the base model or a patch.")
    prompt.add_argument("--prompt", help="Prompt text. Reads stdin if omitted.")
    prompt.add_argument("--patch", type=Path, default=None)
    prompt.add_argument("--compare", action="store_true", help="Generate both base and patched answers.")
    prompt.add_argument("--json", type=Path, help="Write the responses to this file.")
    model_options(prompt, benchmark=False, tokens=512)

    ex = commands.add_parser("export", help="Write a full checkpoint with a patch applied.")
    ex.add_argument("--patch", type=Path, required=True)
    ex.add_argument("--output", type=Path, required=True)
    checkpoint_options(ex)

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
        if args.command == "cache":
            from .model import cached_checkpoints

            output = json.dumps(cached_checkpoints()) + "\n"
            if args.json:
                args.json.write_text(output, encoding="utf-8")
            else:
                print(output, end="")
            return 0
        args.target_model, args.revision = _target(args)
        if args.dtype is None:
            args.dtype = "float32"
            if args.command in ("eval", "export", "prompt") and args.patch:
                from .patchfile import read_manifest
                args.dtype = read_manifest(args.patch).get("dtype", "float32")
        try:
            if args.command == "download":
                from .model import checkpoint_for, fingerprint

                path = checkpoint_for(args.revision, args.target_model, local_only=args.cached_only)
                digest = fingerprint(path)
                print(f"{path}\nfingerprint {digest[:16]}")
                if args.json:
                    local = Path(args.target_model).expanduser().is_dir()
                    info = {"path": str(path), "fingerprint": digest,
                            "revision": None if local else path.name}
                    args.json.write_text(json.dumps(info) + "\n", encoding="utf-8")
            else:
                {"patch": patch, "eval": evaluate_cmd, "export": export, "prompt": prompt_cmd}[args.command](args)
        except LocalEntryNotFoundError:
            command = shlex.join(["patchwork", "download", "--target-model", args.target_model,
                                  "--revision", args.revision])
            raise ValueError(f"Download the checkpoint first: {command}") from None
    except Exception as exc:  # noqa: BLE001 - one clean error line for the user
        print(f"patchwork: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
