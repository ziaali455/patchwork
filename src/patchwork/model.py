"""Finding, fingerprinting and loading a local Transformers checkpoint."""
import hashlib
import json
from pathlib import Path

QWEN = "Qwen/Qwen3.5-0.8B"
DOWNLOAD_FILES = ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "*.tiktoken"]


def snapshot(*, revision, local_only, repo=QWEN):
    from huggingface_hub import snapshot_download

    options = {"repo_id": repo, "revision": revision, "local_files_only": local_only}
    checkpoint = Path(snapshot_download(**options, allow_patterns=DOWNLOAD_FILES))
    if not local_only and not any(checkpoint.glob("*.safetensors")):
        options["revision"] = checkpoint.name
        checkpoint = Path(snapshot_download(**options, allow_patterns=["pytorch_model*.bin"]))
    return checkpoint


def _weight_files(checkpoint: Path):
    return sorted(checkpoint.glob("*.safetensors")) or sorted(checkpoint.glob("pytorch_model*.bin"))


def cached_checkpoints():
    """List likely text-generation snapshots from cache metadata without loading weights."""
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    try:
        cache = scan_cache_dir()
    except CacheNotFound:
        return []

    candidates = []
    for repo in cache.repos:
        if repo.repo_type != "model":
            continue
        for revision in repo.revisions:
            path = revision.snapshot_path
            try:
                config = json.loads((path / "config.json").read_text(encoding="utf-8"))
                if not isinstance(config, dict) or config.get("quantization_config") or config.get("is_encoder_decoder"):
                    continue
                architectures = config.get("architectures")
                if architectures and not any(
                    isinstance(name, str) and name.endswith(("ForCausalLM", "LMHeadModel", "ForConditionalGeneration"))
                    for name in architectures
                ):
                    continue
                weights = _weight_files(path)
                if not weights or not any(
                    (path / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
                ):
                    continue
                index_name = "model.safetensors.index.json" if weights[0].suffix == ".safetensors" else "pytorch_model.bin.index.json"
                index = path / index_name
                if index.is_file():
                    shards = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
                    if not shards or not all((path / shard).is_file() for shard in set(shards.values())):
                        continue
                candidates.append({"source": repo.repo_id, "revision": revision.commit_hash})
            except (OSError, ValueError, KeyError, AttributeError, TypeError):
                continue
    return sorted(candidates, key=lambda item: (item["source"].casefold(), item["revision"]))


def fingerprint(checkpoint: Path) -> str:
    """Hash weight filenames and bytes, preserving existing Safetensors fingerprints."""
    files = _weight_files(Path(checkpoint))
    if not files:
        raise ValueError(f"No Transformers weight files found in {checkpoint}.")
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode())
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load(checkpoint: Path, device: str, dtype="float32"):
    """Load floating-point weights with gradients disabled."""
    import transformers

    options = {"local_files_only": True, "trust_remote_code": False}
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint, **options)
    use_safetensors = any(checkpoint.glob("*.safetensors"))
    model, load_error = None, None
    for name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(checkpoint, dtype=dtype, use_safetensors=use_safetensors,
                                        **options)
            break
        except (ValueError, KeyError) as exc:
            load_error = exc
    if model is None:
        raise ValueError(f"Could not load {checkpoint} as a Transformers text generation model: {load_error}")
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer


def checkpoint_for(revision: str, target: str = QWEN, *, local_only=True) -> Path:
    path = Path(target).expanduser()
    if path.is_dir():
        checkpoint = path.resolve()
    elif target.startswith((".", "~", "/")):
        raise ValueError(f"Target checkpoint directory does not exist: {path}")
    else:
        checkpoint = snapshot(repo=target, revision=revision, local_only=local_only)
    if not _weight_files(checkpoint):
        raise ValueError(f"Checkpoint has no Transformers weight files: {checkpoint}")
    return checkpoint
