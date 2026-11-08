"""The editing session: Claude investigates, edits, tests, keeps or reverts.

Claude only calls the validated tools below. It never runs code on your
machine. Practice outputs are sent to Claude; the evaluation set is not.
"""
import json
import os
import time

from anthropic import Anthropic

SYSTEM = """You are editing the weights of Qwen3.5-0.8B to achieve the user's goal, without training or gradients.

You can only touch MLP down-projection matrices (shape [hidden, intermediate]); each column is a vector the MLP writes into the residual stream. Tools:
- inspect_columns / align_columns: read weights (free).
- run_practice: Qwen answers practice problems (costs 1 forward pass). Shows accuracy, which answers hit the token limit (looping), and how each ends.
- find_direction: average activation difference at a layer between two groups of texts (e.g. 'looping' vs 'finished' from the last practice run), stored under a name (costs 1 forward pass).
- apply_edit: scale columns, or project a direction (stored name or reference column) out of columns. Applied immediately.
- revert_edit: undo the last edit or all edits, exactly.
- finish: end the session.

Forward passes are scarce, so plan each one. Work like a scientist: measure the failure, form a hypothesis about which component causes it, make a small targeted edit, then check it with a practice run. Keep edits that help, revert ones that don't. Before finishing, confirm with a practice run that the edits you keep actually help. If nothing helped, revert everything and say so honestly."""

COLUMN_SELECT = {
    "tensor": {"type": "string"},
    "columns": {"type": "array", "items": {"type": "integer", "minimum": 0}, "description": "explicit column indices (max 256)"},
    "start": {"type": "integer", "minimum": 0},
    "count": {"type": "integer", "minimum": 1},
}
GROUP = {"description": "'looping', 'finished', 'correct', 'wrong' (from the last practice run) or a list of texts",
         "anyOf": [{"type": "string", "enum": ["looping", "finished", "correct", "wrong"]},
                   {"type": "array", "items": {"type": "string"}}]}

TOOLS = [
    {"name": "inspect_columns", "description": "Norms and the first eight rows of up to eight columns.",
     "input_schema": {"type": "object", "properties": COLUMN_SELECT, "required": ["tensor"]}},
    {"name": "align_columns", "description": "Columns of a tensor most aligned (cosine) with a stored direction.",
     "input_schema": {"type": "object", "properties": {
         "tensor": {"type": "string"}, "direction": {"type": "string"},
         "top_k": {"type": "integer", "minimum": 1, "maximum": 128}}, "required": ["tensor", "direction"]}},
    {"name": "run_practice", "description": "Costs 1 forward pass. Qwen answers up to 32 practice problems.",
     "input_schema": {"type": "object", "properties": {
         "ids": {"type": "array", "items": {"type": "integer"}},
         "n_random": {"type": "integer", "minimum": 1, "maximum": 32}}}},
    {"name": "find_direction", "description": "Costs 1 forward pass. Stores normalized mean(group_a) - mean(group_b) activations at a layer's output.",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string"}, "layer": {"type": "integer"}, "group_a": GROUP, "group_b": GROUP},
         "required": ["name", "layer", "group_a", "group_b"]}},
    {"name": "apply_edit", "description": "Apply one edit now. scale: multiply columns by strength (0-2). project_out: remove strength (0-1) of a direction from columns; give `direction` (stored name) or `reference` (column index).",
     "input_schema": {"type": "object", "properties": {
         **COLUMN_SELECT,
         "operation": {"enum": ["scale", "project_out"]},
         "strength": {"type": "number", "minimum": 0, "maximum": 2},
         "direction": {"type": "string"}, "reference": {"type": "integer", "minimum": 0},
         "rationale": {"type": "string"}}, "required": ["tensor", "operation", "strength", "rationale"]}},
    {"name": "revert_edit", "description": "Undo the last edit, or all edits.",
     "input_schema": {"type": "object", "properties": {"which": {"enum": ["last", "all"]}}}},
    {"name": "finish", "description": "End the session.",
     "input_schema": {"type": "object", "properties": {
         "summary": {"type": "string", "description": "What you found, what you kept, and the evidence."},
         "helped": {"type": "boolean"}}, "required": ["summary", "helped"]}},
]


def client():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("Set ANTHROPIC_API_KEY with a key from the Claude Console.")
    return Anthropic(api_key=key, timeout=300, max_retries=2)


def session(api, model, goal, revision, weights, probe, *, log_path, max_turns=60, token_budget=3_000_000, say=print):
    context = {
        "goal": goal, "checkpoint": revision, "tensors": weights.catalog,
        "parameter_budget": weights.budget, "forward_pass_budget": probe.budget,
        "practice_problems": len(probe.practice),
    }
    messages = [{"role": "user", "content": json.dumps(context)}]
    usage = {"input_tokens": 0, "output_tokens": 0, "turns": 0}
    handlers = {
        "inspect_columns": weights.inspect, "align_columns": weights.align,
        "run_practice": probe.run_practice, "find_direction": probe.find_direction,
        "apply_edit": weights.apply, "revert_edit": lambda r: weights.revert(r.get("which", "last")),
    }
    finished, nudged = None, False
    with open(log_path, "a", encoding="utf-8") as log:
        def record(kind, data):
            log.write(json.dumps({"time": time.time(), "kind": kind, "data": data}, default=str) + "\n")
            log.flush()

        for _ in range(max_turns):
            response = api.messages.create(model=model, max_tokens=8000, system=SYSTEM, tools=TOOLS, messages=messages)
            usage["turns"] += 1
            usage["input_tokens"] += response.usage.input_tokens
            usage["output_tokens"] += response.usage.output_tokens
            record("assistant", [block.model_dump() for block in response.content])
            messages.append({"role": "assistant", "content": response.content})
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    say(f"  claude: {block.text.strip()}")
            if response.stop_reason == "refusal":
                say("  The Claude model declined this task.")
                break
            calls = [block for block in response.content if block.type == "tool_use"]
            if not calls:
                if nudged:
                    break
                nudged = True
                messages.append({"role": "user", "content": "Continue with the tools, or call finish."})
                continue

            results = []
            for call in calls:
                if call.name == "finish":
                    finished, content, error = call.input, "Session finished.", False
                elif call.name in handlers:
                    say(f"  → {call.name}" + (f" ({probe.used + 1}/{probe.budget} passes)" if call.name in ("run_practice", "find_direction") else ""))
                    try:
                        content, error = json.dumps(handlers[call.name](call.input)), False
                    except (TypeError, ValueError, KeyError) as exc:
                        content, error = str(exc), True
                else:
                    content, error = f"Unknown tool {call.name}", True
                record("tool", {"name": call.name, "input": call.input, "output": content, "error": error})
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": content, "is_error": error})
            messages.append({"role": "user", "content": results})
            if finished:
                break
            if usage["input_tokens"] + usage["output_tokens"] > token_budget:
                say(f"  Token budget ({token_budget:,}) reached.")
                break
    return finished, usage
