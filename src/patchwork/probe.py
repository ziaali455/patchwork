"""Forward passes: let the editor observe how the target model behaves.

Each call costs one pass from a fixed budget. In the MIT experiment, 10 to 25
passes barely helped and 50 is where the fix landed, so the budget matters.

  run_practice    The target model answers practice problems (never the eval set)
  find_direction  The target model reads two groups of texts; we store the difference of
                  their average activations at one layer as a named direction
"""
import random

import torch


class Probe:
    def __init__(self, model, tokenizer, generator, weights, practice, budget):
        self.model, self.tokenizer, self.generator = model, tokenizer, generator
        self.weights, self.practice = weights, practice
        self.budget, self.used = budget, 0
        self.measurements = {}
        self.last = []  # outputs from the latest run_practice

    def _spend(self):
        if self.used >= self.budget:
            raise ValueError(f"Forward-pass budget exhausted ({self.budget} used).")
        self.used += 1

    def run_practice(self, request):
        ids, n = request.get("ids"), request.get("n_random")
        if ids is not None:
            if not isinstance(ids, list) or not ids:
                raise ValueError("ids must be a nonempty list of practice indices.")
            if any(type(i) is not int or not 0 <= i < len(self.practice) for i in ids):
                raise ValueError(f"ids must be within 0..{len(self.practice) - 1}.")
            chosen = list(dict.fromkeys(ids))[:32]
        else:
            n = 8 if n is None else n
            if type(n) is not int or not 1 <= n <= 32 or not self.practice:
                raise ValueError("n_random must be 1..32 and practice data must be nonempty.")
            chosen = random.Random(0).sample(range(len(self.practice)), min(n, len(self.practice)))
        self._spend()
        from .benchmark import is_correct

        rows, self.last = [], []
        for i in chosen:
            case = self.practice[i]
            out = self.generator(case["prompt"])
            ok = is_correct(out["answer"], case["answer"])
            self.last.append({"id": i, "prompt": case["prompt"], **out, "correct": ok})
            rows.append({"id": i, "correct": ok, "reached_token_limit": out["reached_token_limit"],
                         "tokens": out["generated_tokens"], "ending": out["raw_output"][-160:]})
        result = {
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "reached_token_limit": sum(r["reached_token_limit"] for r in rows),
            "n": len(rows), "results": rows, "passes_left": self.budget - self.used,
        }

        self.measurements[(self.weights.state(), tuple(chosen))] = result
        return {**result, "validation": self.validation()}

    def validation(self):
        state = self.weights.state()
        comparisons = []
        tested = False
        for (measured_state, ids), result in self.measurements.items():
            if measured_state != state:
                continue
            tested = True
            baseline = self.measurements.get(((), ids))
            if baseline is not None:
                comparisons.append({"ids": list(ids), "before_accuracy": baseline["accuracy"],
                                    "after_accuracy": result["accuracy"],
                                    "before_token_limit_hits": baseline["reached_token_limit"],
                                    "after_token_limit_hits": result["reached_token_limit"]})
        status = "unmodified" if not state else "untested"
        if state and tested:
            status = "paired" if comparisons else "tested_without_matching_baseline"
        return {"status": status, "weight_state": list(state), "comparisons": comparisons}

    def _group(self, spec):
        """'looping' | 'finished' | 'correct' | 'wrong' (from the last run), or a list of texts."""
        if isinstance(spec, list) and all(isinstance(t, str) for t in spec):
            return spec[:16]
        keys = {"looping": lambda r: r["reached_token_limit"], "finished": lambda r: not r["reached_token_limit"],
                "correct": lambda r: r["correct"], "wrong": lambda r: not r["correct"]}
        if spec not in keys:
            raise ValueError("A group is 'looping', 'finished', 'correct', 'wrong', or a list of texts.")
        texts = [self.generator.format(r["prompt"]) + r["raw_output"] for r in self.last if keys[spec](r)]
        return texts[:16]

    @torch.inference_mode()
    def _mean_activation(self, texts, layer):
        vecs = []
        for text in texts:
            ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][:, -512:]
            hidden = self.model(input_ids=ids.to(self.model.device), output_hidden_states=True).hidden_states
            vecs.append(hidden[layer + 1][0].float().mean(0))  # output of `layer`, averaged over tokens
        return torch.stack(vecs).mean(0)

    def find_direction(self, request):
        name, layer = request.get("name"), request.get("layer")
        layers = sorted(set(self.weights.layer_of.values()))
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Give the direction a name.")
        if layer not in layers:
            raise ValueError(f"layer must be one of {layers}.")
        group_a, group_b = self._group(request.get("group_a")), self._group(request.get("group_b"))
        if not group_a or not group_b:
            raise ValueError("Both groups need at least one text. Run practice problems first.")
        self._spend()
        diff = self._mean_activation(group_a, layer) - self._mean_activation(group_b, layer)
        self.weights.directions[name] = (diff / diff.norm().clamp_min(1e-12)).cpu()
        return {"stored": name, "layer": layer, "group_sizes": [len(group_a), len(group_b)],
                "raw_norm": round(diff.norm().item(), 4), "passes_left": self.budget - self.used}
