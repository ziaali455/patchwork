"""Rebuild the rounded actor shown in the paper and parse numeric edits."""
import ast
from pathlib import Path
import re
import numpy as np

PROMPT = Path(__file__).parent / 'data' / 'car-prompt.txt'


def base_actor():
    text = PROMPT.read_text().split('THE WEIGHTS (complete actor).', 1)[1]
    pattern = r'^(embed|actor[0-4]|actor_head)\.([Wb]) \(([^)]+)\), scale (1e-?\d+):[^\n]*\n'
    matches = list(re.finditer(pattern, text, re.M))
    weights = {}
    for index, match in enumerate(matches):
        shape = tuple(map(int, match[3].split(' x ')))
        block = text[match.end():matches[index+1].start() if index+1 < len(matches) else len(text)]
        if len(shape) == 2:
            rows = re.findall(r'^\d+: ([-\d ]+)$', block, re.M)
            values = [list(map(int, row.split())) for row in rows]
        else:
            line = next(line for line in block.splitlines() if re.fullmatch(r'[-\d ]+', line))
            values = list(map(int, line.split()))
        array = np.array(values, np.float64) * float(match[4])
        if array.shape != shape:
            raise ValueError("Invalid bundled actor shape")
        weights[f'{match[1]}.{match[2]}'] = array
    if len(weights) != 14 or sum(x.size for x in weights.values()) != 171664:
        raise ValueError("Incomplete bundled actor")
    return weights


def apply_reply(reply, base):
    from .breakout import index_value
    blocks = re.findall(r'^```edits\s*\n(.*?)^```', reply, re.M | re.S)
    if not blocks:
        raise ValueError('No complete edits block')
    weights = {key: value.copy() for key, value in base.items()}
    block = re.sub(r'^ZERO ([a-z0-9_]+\.[Wb])\s*$', r'\1[:] = 0', blocks[-1], flags=re.M)
    statements = ast.parse(block).body
    for statement in statements:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, operation = statement.targets[0], None
        elif isinstance(statement, ast.AugAssign):
            target = statement.target
            operation = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply}[type(statement.op)]
        else:
            raise ValueError('Only numerical assignments are allowed')
        if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute)
                and isinstance(target.value.value, ast.Name)):
            raise ValueError("Expected a named actor tensor and indices")
        key = target.value.value.id + '.' + target.value.attr
        index = index_value(target.slice)
        value = np.asarray(ast.literal_eval(statement.value))
        if value.dtype.kind not in 'ifu' or not np.isfinite(value).all():
            raise ValueError('Edit values must be finite numbers')
        weights[key][index] = value if operation is None else operation(weights[key][index], value)
        if not np.isfinite(weights[key]).all():
            raise ValueError("Edit produced nonfinite weights")
    return weights, len(statements)



def load_actor(path):
    expected = base_actor()
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError("Checkpoint must contain all 14 actor tensors")
        weights = {key: archive[key].copy() for key in expected}
    for key, value in weights.items():
        if value.shape != expected[key].shape or value.dtype.kind != 'f' or not np.isfinite(value).all():
            raise ValueError(f"Invalid actor tensor {key}")
    return weights


def save_actor(path, weights):
    with path.open('xb') as handle:
        np.savez(handle, **weights)
