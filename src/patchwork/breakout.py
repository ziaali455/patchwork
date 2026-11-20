"""Run and inspect Experiment 5 policies. LLM replies are parsed as data only."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parent
PROMPT = ROOT / 'data' / 'breakout-prompt.txt'
KEYS = dict(zip(('W1', 'b1', 'W2', 'b2', 'W3', 'b3'), (
    'actor/dense1/kernel', 'actor/dense1/bias', 'actor/dense2/kernel',
    'actor/dense2/bias', 'actor/logits/kernel', 'actor/logits/bias')))
SHAPES = dict(zip(KEYS, ((600, 32), (32,), (32, 32), (32,), (32, 6), (6,))))


def zero_weights():
    return {name: np.zeros(shape, np.float32) for name, shape in SHAPES.items()}


def load_weights(path):
    with np.load(path, allow_pickle=False) as archive:
        weights = {name: archive[key].copy() for name, key in KEYS.items()}
    validate(weights)
    return weights


def validate(weights):
    for name, value in weights.items():
        if value.shape != SHAPES[name] or not np.isfinite(value).all():
            raise ValueError(f'Invalid tensor {name}')
        if np.max(np.abs(value)) > 1e6:
            raise ValueError(f'{name} exceeds the weight magnitude limit')


def index_value(node):
    if isinstance(node, ast.Tuple):
        return tuple(index_value(item) for item in node.elts)
    if isinstance(node, ast.Slice):
        return slice(*(index_value(item) if item is not None else None
                       for item in (node.lower, node.upper, node.step)))
    value = ast.literal_eval(node)
    if type(value) is not int:
        raise ValueError('Indices must be integers or slices')
    return value


def parse_reply(text, weights=None):
    blocks = re.findall(r'^```edits\s*\n(.*?)^```', text, re.M | re.S)
    if not blocks:
        raise ValueError('No fenced edits block found')
    weights = zero_weights() if weights is None else {k: v.copy() for k, v in weights.items()}
    statements = ast.parse(blocks[-1]).body
    for statement in statements:
        try:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                operation = None
            elif isinstance(statement, ast.AugAssign):
                target = statement.target
                operation = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply}[type(statement.op)]
            else:
                raise ValueError('Only tensor assignments are supported')
            if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
                raise ValueError('Expected W1, b1, W2, b2, W3 or b3 with indices')
            array = weights[target.value.id]
            index = index_value(target.slice)
            literal = ast.literal_eval(statement.value)
            values = np.asarray(literal)
            if values.dtype.kind not in 'ifu' or not np.isfinite(values).all():
                raise ValueError('Expected finite numeric values')
            array[index] = values if operation is None else operation(array[index], values)
            validate(weights)
        except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
            raise ValueError(f'Invalid edit on line {statement.lineno}: {error}') from error
    return weights, len(statements)


def evaluate(weights, episodes, seed, max_steps):
    from minatar import Environment
    rows = []
    for episode in range(episodes):
        env = Environment('breakout', sticky_action_prob=0.1)
        env.seed(seed + episode)
        env.reset()
        rng = np.random.default_rng(np.random.SeedSequence([seed, episode, 12345]))
        score = 0
        observation = np.zeros((10, 10, 6), np.float32)
        for step in range(1, max_steps + 1):
            state = env.state()
            observation[:, :, [0, 2, 3, 4]] = state[:, :, [0, 3, 1, 2]]
            hidden = np.tanh(observation.reshape(600) @ weights['W1'] + weights['b1'])
            hidden = np.tanh(hidden @ weights['W2'] + weights['b2'])
            logits = hidden @ weights['W3'] + weights['b3']
            probabilities = np.exp(logits.astype(np.float64) - np.max(logits))
            probabilities /= probabilities.sum()
            reward, terminal = env.act(int(rng.choice(6, p=probabilities)))
            score += reward
            if terminal:
                break
        rows.append(dict(episode=episode, seed=seed + episode, score=score,
                         steps=step, truncated=not terminal))
        if (episode + 1) % 10 == 0:
            print(f'  {episode + 1}/{episodes} episodes', flush=True)
    scores = np.array([row['score'] for row in rows])
    return dict(mean=float(scores.mean()),
                standard_error=float(scores.std(ddof=1) / np.sqrt(episodes)) if episodes > 1 else None,
                mean_steps=float(np.mean([row['steps'] for row in rows])), episodes=rows)


def run_experiment(args):
    """Import manual replies or request tool-free API replies, then score them."""
    from contextlib import nullcontext
    from . import providers

    if args.episodes < 1 or args.max_steps < 1 or args.rounds < 1 or args.editor_max_tokens < 1:
        raise ValueError('Episodes, steps, rounds and editor token limit must be positive')
    for seed in (args.seed, args.eval_seed):
        if not 0 <= seed <= 2**32 - args.episodes:
            raise ValueError('Episode seeds must fit uint32')
    if abs(args.seed - args.eval_seed) < args.episodes:
        raise ValueError('Practice and final evaluation seed ranges must be disjoint')
    if args.editor_mode == 'manual':
        if not args.reply:
            raise ValueError('Manual mode requires --reply with a saved response')
        if args.rounds != 1:
            raise ValueError('Manual mode uses one candidate per --reply file; omit --rounds')
        replies = [path.read_text() for path in args.reply]
        # Catch malformed manual responses before starting the run.
        for reply in replies:
            parse_reply(reply)
        connection = nullcontext(None)
        model = None
    else:
        if args.reply:
            raise ValueError('--reply is only supported in manual mode')
        model = args.editor_model
        if model is None:
            if args.editor_provider != 'anthropic':
                raise ValueError('Provide --editor-model for this provider')
            from .cli import SHORTCUTS
            model = SHORTCUTS['sonnet']
        if not model.strip():
            raise ValueError('Editor model must be nonempty')
        replies = [None] * args.rounds
        connection = providers.connect(args.editor_provider, model, base_url=args.editor_base_url,
                                       key_env=args.editor_key_env)
    if args.output.exists():
        raise ValueError(f'Output already exists: {args.output}')
    # Check the optional dependency before making a paid request.
    from minatar import Environment  # noqa: F401

    with connection as provider:
        args.output.mkdir(parents=True)
        prompt = PROMPT.read_text()
        (args.output / 'prompt.txt').write_text(prompt)
        report = dict(status='running', editor_mode=args.editor_mode,
                      editor_provider=args.editor_provider if provider else None, editor_model=model,
                      editor_max_tokens=args.editor_max_tokens, notes=args.notes,
                      settings=dict(episodes=args.episodes, practice_seed=args.seed, eval_seed=args.eval_seed,
                                    max_steps=args.max_steps, sticky_action_probability=0.1,
                                    minatar_version='1.0.15'),
                      candidates=[], selected='zero', usage=dict(input_tokens=0, output_tokens=0))
        report_path = args.output / 'report.json'

        def save_report():
            temporary = args.output / 'report.json.tmp'
            temporary.write_text(json.dumps(report, indent=2) + '\n')
            temporary.replace(report_path)

        save_report()
        try:
            print('Evaluating zero-weight practice baseline', flush=True)
            best_weights = zero_weights()
            baseline = evaluate(best_weights, args.episodes, args.seed, args.max_steps)
            report['practice_baseline'] = baseline
            best_mean = baseline['mean']
            messages = [{'role': 'user', 'content': prompt}]
            save_report()
            for number, reply in enumerate(replies, 1):
                label = f'attempt-{number:02d}'
                if provider:
                    print(f'Requesting {label} from {model}', flush=True)
                    turn = provider.turn(model, messages,
                                         system='Follow the supplied network specification. Return an explanation and a final fenced edits block. Use no tools.',
                                         use_tools=False, max_output_tokens=args.editor_max_tokens)
                    (args.output / f'{label}-provider.json').write_text(json.dumps(turn.log, indent=2))
                    report['usage']['input_tokens'] += turn.input_tokens
                    report['usage']['output_tokens'] += turn.output_tokens
                    reply = '\n\n'.join(turn.text)
                    (args.output / f'{label}.md').write_text(reply)
                    if turn.refused or turn.calls:
                        raise ValueError('Editor refused or returned an unexpected tool call; response saved')
                    messages.extend(turn.history)
                else:
                    (args.output / f'{label}.md').write_text(reply)
                weights, statements = parse_reply(reply)
                checkpoint = args.output / f'{label}.npz'
                with checkpoint.open('xb') as handle:
                    np.savez(handle, **{KEYS[name]: value for name, value in weights.items()})
                print(f'Evaluating {label} on practice seeds', flush=True)
                result = evaluate(weights, args.episodes, args.seed, args.max_steps)
                kept = result['mean'] > best_mean
                if kept:
                    best_weights, best_mean = weights, result['mean']
                    report['selected'] = label
                report['candidates'].append(dict(name=label, statements=statements, practice=result,
                                                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                                  improved_on_best=kept))
                save_report()
                print(f"{label}: reward {result['mean']:.2f}; best {best_mean:.2f}", flush=True)
                if provider and number < len(replies):
                    feedback = {key: result[key] for key in ('mean', 'standard_error', 'mean_steps')}
                    messages.append({'role': 'user', 'content': (
                        f'Your candidate scored {json.dumps(feedback)} on {args.episodes} practice episodes. '
                        f'The best mean reward so far is {best_mean}. Improve your design. '
                        'Return a COMPLETE replacement network in a final edits block, starting again from all zeros. '
                        'You now have gameplay feedback, so this is an iterative extension of the first attempt.'
                    )})
            with (args.output / 'selected.npz').open('xb') as handle:
                np.savez(handle, **{KEYS[name]: value for name, value in best_weights.items()})
            print('Evaluating zero and selected policy on fresh seeds', flush=True)
            report['before'] = evaluate(zero_weights(), args.episodes, args.eval_seed, args.max_steps)
            report['after'] = evaluate(best_weights, args.episodes, args.eval_seed, args.max_steps)
            report['status'] = 'complete'
            save_report()
            print(f"Held-out reward: {report['before']['mean']:.2f} -> {report['after']['mean']:.2f}")
            print(f'Saved run to {args.output}')
        except BaseException as error:
            report['status'] = 'failed'
            report['error'] = str(error)
            save_report()
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(prog="patchwork breakout", description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prompt = commands.add_parser('prompt', help='Print the original one-shot prompt')
    apply = commands.add_parser('apply', help='Import the last edits block from an LLM response')
    apply.add_argument('reply', type=Path)
    apply.add_argument('output', type=Path)
    apply.add_argument('--base', type=Path, help='Apply an incremental revision to this checkpoint')
    run = commands.add_parser('evaluate')
    run.add_argument('--weights', nargs='+', default=['zero'], help='zero, reference, or .npz paths')
    run.add_argument('--episodes', type=int, default=100)
    run.add_argument('--seed', type=int, default=0)
    run.add_argument('--max-steps', type=int, default=10000)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--reference', type=Path, help='Published hand.npz checkpoint for the reference policy')
    experiment = commands.add_parser('run', help='Author/import weights and benchmark them')
    experiment.add_argument('--editor-mode', choices=('api', 'manual'), required=True)
    experiment.add_argument('--reply', type=Path, action='append', help='Manual response file; repeat for multiple candidates')
    experiment.add_argument('--editor-provider', choices=('anthropic', 'openai', 'openai-compatible'), default='anthropic')
    experiment.add_argument('--editor-model', help='Provider model ID; default is Claude Sonnet for Anthropic')
    experiment.add_argument('--editor-base-url')
    experiment.add_argument('--editor-key-env')
    experiment.add_argument('--editor-max-tokens', type=int, default=16384)
    experiment.add_argument('--rounds', type=int, default=1, help='API attempts, with practice feedback after the first')
    experiment.add_argument('--episodes', type=int, default=100, help='Episodes for each practice and held-out evaluation')
    experiment.add_argument('--seed', type=int, default=0, help='First practice seed')
    experiment.add_argument('--eval-seed', type=int, default=10000, help='First held-out seed')
    experiment.add_argument('--max-steps', type=int, default=10000)
    experiment.add_argument('--notes', default='', help='Record provenance, such as exposure to published weights')
    experiment.add_argument('--output', type=Path, required=True, help='New directory for responses, networks and report')
    args = parser.parse_args(argv)
    if args.command == 'run':
        run_experiment(args)
    elif args.command == 'prompt':
        print(PROMPT.read_text(), end="")
    elif args.command == 'apply':
        weights, count = parse_reply(args.reply.read_text(), load_weights(args.base) if args.base else None)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as handle:
            np.savez(handle, **{KEYS[name]: value for name, value in weights.items()})
        print(f'Saved {count} statements to {args.output}')
    else:
        if args.episodes < 1 or args.max_steps < 1 or not 0 <= args.seed <= 2**32 - args.episodes:
            parser.error('Use positive episodes/max-steps and seeds within uint32 range')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Reserve the result filename before doing potentially lengthy evaluations.
        with args.output.open('x') as handle:
            results = dict(episodes=args.episodes, seed=args.seed, max_steps=args.max_steps,
                           sticky_action_probability=0.1, minatar_version='1.0.15', policies={})
            for label in args.weights:
                path = args.reference if label == 'reference' else Path(label)
                if path is None:
                    raise ValueError('Supply --reference with the published hand.npz path')
                weights = zero_weights() if label == 'zero' else load_weights(path)
                print(f'Evaluating {label}', flush=True)
                result = evaluate(weights, args.episodes, args.seed, args.max_steps)
                result['checkpoint_sha256'] = None if label == 'zero' else hashlib.sha256(path.read_bytes()).hexdigest()
                results['policies'][label] = result
                handle.seek(0)
                json.dump(results, handle, indent=2)
                handle.truncate()
                handle.flush()
                print(f"{label}: mean reward {result['mean']:.2f}, SE {result['standard_error']}", flush=True)


if __name__ == '__main__':
    main()
