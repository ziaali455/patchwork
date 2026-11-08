"""Sparse patch files: only the weights that changed, and by how much.

A patch is a folder with:
  patch.safetensors  "<tensor>::idx" (flat indices) and "<tensor>::delta" (float32)
  manifest.json      goal, base checkpoint revision + fingerprint, edits, stats
It applies only to the exact checkpoint it was made from, and can be removed
again by subtracting the same deltas.
"""
import json
import time
from pathlib import Path

import torch

FORMAT = "patchwork-sparse-delta/1"


@torch.inference_mode()
def build(weights):
    """Deltas for every edited column, from the originals saved in the edit history."""
    tensors, changed = {}, 0
    for entry in weights.history:
        matrix = weights.tensors[entry["tensor"]]
        cols = torch.tensor(entry["columns"], device=matrix.device)
        delta = (matrix[:, cols].float() - entry["original"].float()).cpu()
        rows, which = delta.nonzero(as_tuple=True)
        flat = rows * matrix.shape[1] + torch.tensor(entry["columns"])[which]
        key = entry["tensor"]
        idx = torch.cat([tensors.get(f"{key}::idx", torch.empty(0, dtype=torch.long)), flat])
        val = torch.cat([tensors.get(f"{key}::delta", torch.empty(0)), delta[rows, which]])
        tensors[f"{key}::idx"], tensors[f"{key}::delta"] = idx.contiguous(), val.contiguous()
        changed += flat.numel()
    return tensors, changed


def write(out_dir: Path, tensors, manifest):
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out_dir / "patch.safetensors")
    manifest = {"format": FORMAT, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **manifest}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_manifest(patch_dir: Path):
    manifest = json.loads((Path(patch_dir) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError(f"Not a {FORMAT} patch.")
    return manifest


@torch.inference_mode()
def apply(model, patch_dir: Path, fingerprint: str, sign: float = 1.0) -> int:
    from safetensors import safe_open

    manifest = read_manifest(patch_dir)
    if manifest["base"]["fingerprint"] != fingerprint:
        raise ValueError("This patch was made for a different checkpoint. Refusing to apply it.")
    params, applied = dict(model.named_parameters()), 0
    with safe_open(Path(patch_dir) / "patch.safetensors", framework="pt") as handle:
        for key in handle.keys():
            if not key.endswith("::idx"):
                continue
            name = key[: -len("::idx")]
            if name not in params:
                raise ValueError(f"Patch touches {name}, which this model doesn't have.")
            flat = params[name].data.view(-1)
            idx, delta = handle.get_tensor(key), handle.get_tensor(f"{name}::delta")
            flat.index_add_(0, idx.to(flat.device), (sign * delta).to(flat.dtype).to(flat.device))
            applied += idx.numel()
    return applied
