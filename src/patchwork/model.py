"""Finding, fingerprinting and loading the Qwen checkpoint."""
import hashlib
from pathlib import Path

QWEN = "Qwen/Qwen3.5-0.8B"
DOWNLOAD_FILES = ["*.json", "*.safetensors", "*.jinja", "*.txt"]


def snapshot(*, revision, local_only):
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id=QWEN, revision=revision, allow_patterns=DOWNLOAD_FILES, local_files_only=local_only,
    ))


def fingerprint(checkpoint: Path) -> str:
    """sha256 over the weight files. A patch only applies to this exact checkpoint."""
    digest = hashlib.sha256()
    for file in sorted(Path(checkpoint).glob("*.safetensors")):
        digest.update(file.name.encode())
        with open(file, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load(checkpoint: Path, device: str):
    """Load in float32 with gradients off everywhere (no training, by construction)."""
    import transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = None
    for name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(checkpoint, local_files_only=True, dtype="float32")
            break
        except (ValueError, KeyError):
            continue
    if model is None:
        raise ValueError("This transformers version can't load the checkpoint. Upgrade transformers.")
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer


def checkpoint_for(revision: str) -> Path:
    checkpoint = snapshot(revision=revision, local_only=True)
    if not any(file.is_file() for file in checkpoint.glob("*.safetensors")):
        raise ValueError(f"Checkpoint is incomplete. Run patchwork download --revision {revision}.")
    return checkpoint
