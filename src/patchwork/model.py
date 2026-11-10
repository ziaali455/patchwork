"""Finding, fingerprinting and loading a local Transformers checkpoint."""
import hashlib
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


def load(checkpoint: Path, device: str):
    """Load in float32 with gradients off everywhere."""
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
            model = cls.from_pretrained(checkpoint, dtype="float32", use_safetensors=use_safetensors,
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
