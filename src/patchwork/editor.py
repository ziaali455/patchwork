"""Provider-independent tool schemas and editing loop."""
import json
import time
from dataclasses import dataclass

SYSTEM = """You are editing the weights of the target model identified in the session context to achieve the user's goal, without training or gradients.

You can only touch MLP down-projection matrices (shape [hidden, intermediate]); each column is a vector the MLP writes into the residual stream. Tools:
- inspect_columns / align_columns: read weights (free).
- run_practice: The target model answers practice problems (costs 1 forward pass). Shows accuracy, which answers hit the token limit (looping), and how each ends.
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
    {"name": "run_practice", "description": "Costs 1 forward pass. The target model answers up to 32 practice problems.",
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


@dataclass
class Turn:
    history: list
    log: list
    calls: list
    text: list
    input_tokens: int
    output_tokens: int
    refused: bool = False


def session(provider, model, goal, checkpoint, weights, probe, *, log_path,
            max_turns=60, token_budget=3_000_000, say=print):
    context = {
        "goal": goal, "checkpoint": checkpoint, "tensors": weights.catalog,
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
            turn = provider.turn(model, messages)
            usage["turns"] += 1
            usage["input_tokens"] += turn.input_tokens
            usage["output_tokens"] += turn.output_tokens
            record("assistant", turn.log)
            messages.extend(turn.history)
            for message in turn.text:
                if message.strip():
                    say(f"  editor: {message.strip()}")
            if turn.refused:
                say("  The editor model declined this task.")
                break

            results = []
            for call_id, name, raw_input in turn.calls:
                args = raw_input
                try:
                    args = json.loads(raw_input) if isinstance(raw_input, str) else raw_input
                    if not isinstance(args, dict):
                        raise ValueError("Tool arguments must be an object.")
                    if name == "finish":
                        if not isinstance(args.get("summary"), str) or type(args.get("helped")) is not bool:
                            raise ValueError("finish requires a summary and a boolean helped value.")
                        finished, content = args, "Session finished."
                    elif name in handlers:
                        say(f"  → {name}" + (f" ({probe.used + 1}/{probe.budget} passes)"
                                             if name in ("run_practice", "find_direction") else ""))
                        content = json.dumps(handlers[name](args))
                    else:
                        raise ValueError(f"Unknown tool {name}")
                    error = False
                except (TypeError, ValueError, KeyError) as exc:
                    content, error = str(exc), True
                record("tool", {"name": name, "input": args, "output": content, "error": error})
                results.append((call_id, content, error))

            if finished:
                break
            if usage["input_tokens"] + usage["output_tokens"] > token_budget:
                say(f"  Token budget ({token_budget:,}) reached.")
                break
            if results:
                messages.extend(provider.tool_results(results))
            elif nudged:
                break
            else:
                nudged = True
                messages.append({"role": "user", "content": "Continue with the tools, or call finish."})
    return finished, usage
