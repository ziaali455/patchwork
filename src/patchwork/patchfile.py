"""Sparse patches with exact original and replacement values.

Version 2 replays and reverses values exactly, and validates every entry before
mutation. Version 1 delta patches remain readable; their original values cannot
always be recovered exactly from rounded deltas.
"""
import json
import time
from pathlib import Path

import torch

LEGACY_FORMAT = "patchwork-sparse-delta/1"
FORMAT = "patchwork-sparse-values/2"


@torch.inference_mode()
def build(weights):
    """Exact original and edited values for each changed entry."""
    tensors, changed = {}, 0
    for entry in weights.history:
        matrix = weights.tensors[entry["tensor"]]
        cols = torch.tensor(entry["columns"], device=matrix.device)
        changed_mask = (matrix[:, cols] != entry["original"]).cpu()
        rows, which = changed_mask.nonzero(as_tuple=True)
        columns = torch.tensor(entry["columns"])[which]
        key = entry["tensor"]
        if key in weights.transposed:
            flat = columns * matrix.shape[0] + rows
        else:
            flat = rows * matrix.shape[1] + columns
        idx = torch.cat([tensors.get(f"{key}::idx", torch.empty(0, dtype=torch.long)), flat])
        tensors[f"{key}::idx"] = idx.contiguous()
        for suffix, source in (("before", entry["original"]), ("after", matrix[:, cols])):
            values = source[rows.to(source.device), which.to(source.device)].cpu()
            previous = tensors.get(f"{key}::{suffix}", torch.empty(0, dtype=values.dtype))
            tensors[f"{key}::{suffix}"] = torch.cat([previous, values]).contiguous()
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
    if manifest.get("format") not in (FORMAT, LEGACY_FORMAT):
        raise ValueError(f"Not a {FORMAT} patch.")
    return manifest


@torch.inference_mode()
def apply(model, patch_dir: Path, fingerprint: str, sign: float = 1.0) -> int:
    from safetensors import safe_open

    manifest = read_manifest(patch_dir)
    if manifest["base"]["fingerprint"] != fingerprint:
        raise ValueError("This patch was made for a different checkpoint. Refusing to apply it.")
    if sign not in (-1.0, 1.0):
        raise ValueError("sign must be 1 or -1.")
    params, prepared = dict(model.named_parameters()), []
    legacy = manifest["format"] == LEGACY_FORMAT
    with safe_open(Path(patch_dir) / "patch.safetensors", framework="pt") as handle:
        keys = set(handle.keys())
        names = {key.rsplit("::", 1)[0] for key in keys}
        suffixes = ("idx", "delta") if legacy else ("idx", "before", "after")
        if keys != {f"{name}::{suffix}" for name in names for suffix in suffixes}:
            raise ValueError("Incomplete or unexpected patch tensors.")
        for name in sorted(names):
            if name not in params:
                raise ValueError(f"Patch touches unknown tensor {name}.")
            param = params[name]
            idx = handle.get_tensor(f"{name}::idx")
            if idx.dtype != torch.int64 or idx.ndim != 1 or idx.unique().numel() != idx.numel():
                raise ValueError("Patch indices must be unique int64 vectors.")
            if idx.numel() and (idx.min() < 0 or idx.max() >= param.numel()):
                raise ValueError("Patch index outside target tensor.")
            device_idx = idx.to(param.device)
            current = param.detach().reshape(-1)[device_idx]
            if legacy:
                delta = handle.get_tensor(f"{name}::delta")
                if delta.shape != idx.shape or not torch.isfinite(delta).all():
                    raise ValueError("Invalid legacy delta.")
                updated = (current.float() + sign * delta.to(param.device).float()).to(param.dtype)
            else:
                before = handle.get_tensor(f"{name}::before")
                after = handle.get_tensor(f"{name}::after")
                for value in (before, after):
                    if value.shape != idx.shape or value.dtype != param.dtype or not torch.isfinite(value).all():
                        raise ValueError("Patch values must match the target dtype and index shape and be finite.")
                expected, replacement = (before, after) if sign == 1 else (after, before)
                if not torch.equal(current, expected.to(param.device)):
                    raise ValueError("Current weights do not match the patch's expected source values.")
                updated = replacement.to(param.device)
            if not torch.isfinite(updated).all():
                raise ValueError("Patch would produce non-finite weights.")
            prepared.append((param, device_idx, updated))
    # Validate every entry before mutating any target tensor.
    for param, idx, updated in prepared:
        flat = param.detach().reshape(-1).clone()
        flat[idx] = updated
        param.copy_(flat.reshape(param.shape))
    return sum(idx.numel() for _, idx, _ in prepared)
