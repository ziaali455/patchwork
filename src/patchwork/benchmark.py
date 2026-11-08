"""Evaluation: generate answers greedily and score them.

Cases come from a JSON file ([{"prompt", "answer"}, ...]) or a built-in spec:
  gsm8k        30 held-out GSM8K test problems
  gsm8k:100    100 of them
Practice cases (what Claude gets to try during editing) default to the GSM8K
*train* split, so Claude never sees the evaluation set.
"""
import json
import re
from pathlib import Path
from time import perf_counter

MATH_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _check(cases, source):
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{source} must contain a nonempty JSON array.")
    for i, case in enumerate(cases, 1):
        if not isinstance(case, dict) or any(
            not isinstance(case.get(key), str) or not case[key].strip() for key in ("prompt", "answer")
        ):
            raise ValueError(f"{source}: case {i} needs nonempty prompt and answer strings.")
    return cases


def _gsm8k(split, n=None, seed=0):
    from datasets import load_dataset

    data = load_dataset("openai/gsm8k", "main", split=split)
    if n is not None:
        data = data.shuffle(seed=seed).select(range(min(n, len(data))))
    return [
        {"prompt": row["question"] + MATH_SUFFIX, "answer": row["answer"].split("####")[-1].strip().replace(",", "")}
        for row in data
    ]


def read_cases(spec, default_n=30):
    """A JSON file path, or 'gsm8k' / 'gsm8k:N' for held-out GSM8K test problems."""
    spec = str(spec)
    if spec.split(":")[0] == "gsm8k":
        n = int(spec.split(":")[1]) if ":" in spec else default_n
        return _gsm8k("test", n)
    return _check(json.loads(Path(spec).read_text(encoding="utf-8")), spec)


def read_practice(spec):
    if spec is None or spec == "gsm8k":
        return _gsm8k("train")
    return _check(json.loads(Path(spec).read_text(encoding="utf-8")), spec)


def normalize(text):
    return " ".join(text.casefold().split())


def _number(text):
    match = NUMBER.findall(text.replace("$", ""))
    if not match:
        return None
    try:
        return float(match[-1].replace(",", ""))
    except ValueError:
        return None


def is_correct(answer: str, expected: str) -> bool:
    """Numeric answers compare by value (prefers the last \\boxed{}); text by exact match."""
    if NUMBER.fullmatch(expected.strip().replace(",", "")):
        boxed = re.findall(r"\\boxed\{([^{}]*)\}", answer)
        got, want = _number(boxed[-1] if boxed else answer), _number(expected)
        return got is not None and want is not None and abs(got - want) < 1e-6
    return normalize(answer) == normalize(expected)


class Generator:
    """Greedy decoding: deterministic, so before/after differences come from the edit alone."""

    def __init__(self, model, tokenizer, *, max_new_tokens, thinking):
        from transformers import GenerationConfig

        self.model, self.tokenizer = model, tokenizer
        self.max_new_tokens, self.thinking = max_new_tokens, thinking
        original = model.generation_config
        self.config = GenerationConfig(
            eos_token_id=original.eos_token_id, pad_token_id=original.pad_token_id,
            bos_token_id=original.bos_token_id, do_sample=False, use_cache=True,
        )
        self.think_end = tokenizer.convert_tokens_to_ids("</think>") if "</think>" in tokenizer.get_vocab() else None

    def format(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        if not getattr(self.tokenizer, "chat_template", None):
            return prompt
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.thinking,
        )

    def __call__(self, prompt, limit=None):
        import torch

        limit = limit or self.max_new_tokens
        inputs = self.tokenizer(self.format(prompt), return_tensors="pt", add_special_tokens=False).to(self.model.device)
        self.config.max_new_tokens = limit
        with torch.inference_mode():
            start = perf_counter()
            generated = self.model.generate(**inputs, generation_config=self.config)
            seconds = perf_counter() - start
        tokens = generated[0, inputs["input_ids"].shape[1]:].tolist()
        raw = self.tokenizer.decode(tokens, skip_special_tokens=False)
        if self.think_end is not None and self.think_end in tokens:
            answer_tokens = tokens[tokens.index(self.think_end) + 1:]
        elif self.thinking:
            answer_tokens = []  # never finished thinking: no answer
        else:
            answer_tokens = tokens
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        return {"raw_output": raw, "answer": answer, "generation_seconds": seconds,
                "generated_tokens": len(tokens), "reached_token_limit": len(tokens) == limit}


def evaluate(generator, cases, *, progress=None):
    results = []
    for i, case in enumerate(cases, 1):
        out = generator(case["prompt"])
        results.append({"prompt": case["prompt"], "expected": case["answer"], **out,
                        "correct": is_correct(out["answer"], case["answer"])})
        if progress:
            progress(i, len(cases))
    seconds = sum(row["generation_seconds"] for row in results)
    tokens = sum(row["generated_tokens"] for row in results)
    correct = sum(row["correct"] for row in results)
    return {
        "accuracy": correct / len(results),
        "correct": correct,
        "total": len(results),
        "reached_token_limit": sum(row["reached_token_limit"] for row in results),
        "mean_generation_seconds": seconds / len(results),
        "generated_tokens_per_second": tokens / seconds if seconds else 0.0,
        "cases": results,
    }
