"""The only things Claude can do to the weights.

The editor calls these validated operations on the MLP
down-projection matrices (the matrices that write each layer's MLP output
back into the model's main "residual" stream):

  scale        multiply some columns by a factor
  project_out  remove a direction from some columns. The direction is either
               an existing column or one measured from activations (probe.py)

Every edit is applied immediately and can be reverted exactly.
"""
import math
import re

import torch

DOWN_PROJECTION = (
    r"(?:.*\.)?(?:layers|h|blocks|block)\.(?P<layer>\d+)\."
    r"(?:mlp\.(?:down_proj|c_proj|dense_4h_to_h|fc2|fc_out|proj_out)|fc2)\.weight"
)
NON_TEXT_PREFIXES = {"vision", "visual", "vision_model", "vision_tower", "vision_encoder",
                     "audio", "audio_model", "audio_encoder"}
MAX_COLUMNS_PER_EDIT = 256


class Weights:
    def __init__(self, model, budget_fraction=0.001, edit_pattern=None):
        from transformers.pytorch_utils import Conv1D

        self.tensors, self.layer_of = {}, {}
        self.transposed = set()
        pattern = re.compile(edit_pattern or DOWN_PROJECTION)
        if "layer" not in pattern.groupindex:
            raise ValueError("--edit-pattern requires a named group (?P<layer>...) for the layer index.")
        config = model.config.get_text_config()
        hidden_size = config.hidden_size
        for name, param in model.named_parameters():
            match = pattern.fullmatch(name)
            if not match or param.ndim != 2 or not param.is_floating_point():
                continue
            if edit_pattern is None and NON_TEXT_PREFIXES.intersection(name.split(".")):
                continue
            module = model.get_submodule(name.rsplit(".", 1)[0])
            transpose = isinstance(module, Conv1D) or (param.shape[0] != hidden_size and param.shape[1] == hidden_size)
            matrix = param.T if transpose else param
            if matrix.shape[0] != hidden_size:
                continue
            layer = int(match.group("layer"))
            if not 0 <= layer < config.num_hidden_layers:
                raise ValueError(f"Invalid layer index {layer} for editable tensor {name}.")
            self.tensors[name], self.layer_of[name] = matrix, layer
            if transpose:
                self.transposed.add(name)
        if not self.tensors:
            raise ValueError("No editable text MLP down-projection matrices found. "
                             "Use --edit-pattern for a different Transformers layout.")
        self.catalog = {name: list(param.shape) for name, param in self.tensors.items()}
        self.budget = int(sum(p.numel() for p in model.parameters()) * budget_fraction)
        self.directions: dict[str, torch.Tensor] = {}
        self.history = []  # applied edits, each with exact original values for revert

    # Selecting columns -----------------------------------------------------
    def _tensor(self, name):
        if not isinstance(name, str) or name not in self.tensors:
            raise ValueError("Select a tensor from the catalog.")
        return self.tensors[name]

    def _columns(self, matrix, edit):
        if "columns" in edit:
            cols = edit["columns"]
            if not isinstance(cols, list) or not cols or any(type(c) is not int for c in cols):
                raise ValueError("columns must be a nonempty list of integers.")
            cols = sorted(set(cols))
        else:
            start, count = edit.get("start"), edit.get("count")
            if type(start) is not int or type(count) is not int or count < 1:
                raise ValueError("Give columns, or start and count.")
            cols = list(range(start, start + count))
        if cols[0] < 0 or cols[-1] >= matrix.shape[1]:
            raise ValueError(f"Columns must be within 0..{matrix.shape[1] - 1}.")
        if len(cols) > MAX_COLUMNS_PER_EDIT:
            raise ValueError(f"At most {MAX_COLUMNS_PER_EDIT} columns per edit.")
        return torch.tensor(cols, dtype=torch.long, device=matrix.device)

    def touched(self):
        return sum(len(entry["columns"]) * self.tensors[entry["tensor"]].shape[0]
                   for entry in self.history)

    # Reading ---------------------------------------------------------------
    @torch.inference_mode()
    def inspect(self, request):
        matrix = self._tensor(request.get("tensor"))
        cols = self._columns(matrix, request)
        if len(cols) > 8:
            raise ValueError("Inspect at most eight columns at a time.")
        values = matrix[:, cols].float()
        return {"columns": cols.tolist(), "l2_norms": values.norm(dim=0).tolist(),
                "first_eight_rows": values[:8].tolist()}

    @torch.inference_mode()
    def align(self, request):
        """Which columns of a tensor point along a direction? (cosine similarity)"""
        matrix = self._tensor(request.get("tensor"))
        direction = self._direction(request.get("direction")).to(matrix.device)
        top_k = min(int(request.get("top_k", 16)), 128)
        values = matrix.float()
        cos = (direction @ values) / values.norm(dim=0).clamp_min(1e-12)
        order = cos.abs().argsort(descending=True)[:top_k]
        return {
            "top_columns": order.tolist(),
            "cosines": [round(cos[i].item(), 4) for i in order],
            "mean_abs_cosine_all_columns": round(cos.abs().mean().item(), 4),
        }

    def _direction(self, name):
        if name not in self.directions:
            raise ValueError(f"Unknown direction {name!r}. Known: {sorted(self.directions)}")
        return self.directions[name]

    # Writing ---------------------------------------------------------------
    @torch.inference_mode()
    def apply(self, edit):
        """Validate and apply one edit. Returns a summary; raises ValueError if invalid."""
        if not isinstance(edit, dict):
            raise ValueError("An edit must be an object.")
        name = edit.get("tensor")
        matrix = self._tensor(name)
        cols = self._columns(matrix, edit)
        if any(e["tensor"] == name and set(e["columns"]) & set(cols.tolist()) for e in self.history):
            raise ValueError("These columns were already edited. Revert first to change them.")
        if self.touched() + len(cols) * matrix.shape[0] > self.budget:
            raise ValueError(f"Edit would exceed the {self.budget:,}-parameter budget.")
        strength = edit.get("strength")
        if type(strength) not in (int, float) or not 0 <= strength <= 2:
            raise ValueError("strength must be between 0 and 2.")

        values = matrix[:, cols].float()
        operation = edit.get("operation")
        if operation == "scale":
            updated = values * strength
        elif operation == "project_out":
            if strength > 1:
                raise ValueError("project_out strength is a fraction between 0 and 1.")
            if "direction" in edit:
                direction = self._direction(edit["direction"]).to(values.device)
            elif "reference" in edit:
                ref = edit["reference"]
                if type(ref) is not int or not 0 <= ref < matrix.shape[1]:
                    raise ValueError("reference must be a column index.")
                direction = matrix[:, ref].float()
            else:
                raise ValueError("project_out needs a direction name or a reference column.")
            norm = direction.norm()
            if norm.item() == 0 or not math.isfinite(norm.item()):
                raise ValueError("Direction must have a finite, nonzero norm.")
            direction = direction / norm
            updated = values - strength * direction[:, None] * (direction @ values)
        else:
            raise ValueError("operation must be scale or project_out.")
        updated = updated.to(matrix.dtype)
        if not torch.isfinite(updated).all():
            raise ValueError("Edit would create non-finite weights.")

        original = matrix[:, cols].clone()
        matrix[:, cols] = updated
        changed = int(torch.count_nonzero(updated != original))
        record = {k: v for k, v in edit.items()}
        self.history.append({"tensor": name, "columns": cols.tolist(), "original": original,
                             "edit": record, "changed": changed})
        return {"edit_id": len(self.history), "parameters_changed": changed,
                "budget_used": self.touched(), "budget": self.budget}

    @torch.inference_mode()
    def revert(self, which="last"):
        """Undo the last edit, or all of them. Exact: restores saved values."""
        count = len(self.history) if which == "all" else min(1, len(self.history))
        for _ in range(count):
            entry = self.history.pop()
            cols = torch.tensor(entry["columns"], device=self.tensors[entry["tensor"]].device)
            self.tensors[entry["tensor"]][:, cols] = entry["original"]
        return {"reverted": count, "edits_remaining": len(self.history)}

    def summary(self):
        return [{"tensor": e["tensor"], "columns": e["columns"], **{k: v for k, v in e["edit"].items()
                 if k not in ("tensor", "columns", "start", "count")}, "parameters_changed": e["changed"]}
                for e in self.history]
