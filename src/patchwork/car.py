"""Repair and evaluate the Kinetix car policy from Experiment 4."""
import argparse
from contextlib import nullcontext
import hashlib
import json

import numpy as np
from pathlib import Path

from .car_weights import PROMPT, apply_reply, base_actor, load_actor, save_actor


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def evaluator():
    try:
        from .car_sim import CarEvaluator
    except ImportError as error:
        raise ValueError("Install car support with: uv pip install -e '.[car]'") from error
    return CarEvaluator()


def run(args):
    from . import providers

    if args.rounds is None:
        args.rounds = 1 if args.editor_mode == "manual" else 2
    if min(args.practice_episodes, args.eval_episodes, args.rounds, args.editor_max_tokens) < 1:
        raise ValueError('Episode counts, rounds and editor token limit must be positive')
    for seed, count in ((args.seed, args.practice_episodes), (args.eval_seed, args.eval_episodes)):
        if not 0 <= seed <= 2**32-count:
            raise ValueError('Seeds must fit uint32')
    if max(args.seed, args.eval_seed) < min(args.seed+args.practice_episodes, args.eval_seed+args.eval_episodes):
        raise ValueError('Practice and held-out seed ranges must be disjoint')
    if args.output.exists() or args.output.is_symlink():
        raise ValueError(f'Output already exists: {args.output}')
    base = base_actor()
    initial = load_actor(args.base) if args.base else base
    if args.editor_mode == 'manual':
        if not args.reply or args.rounds != 1:
            raise ValueError('Manual mode needs --reply files and --rounds 1')
        replies = [path.read_text() for path in args.reply]
        for reply in replies:
            apply_reply(reply, base)
        connection = nullcontext(None)
        model = None
    else:
        if args.reply:
            raise ValueError('--reply requires manual mode')
        model = args.editor_model
        if model is None:
            if args.editor_provider != 'anthropic':
                raise ValueError('Provide --editor-model')
            from .cli import SHORTCUTS
            model = SHORTCUTS['sonnet']
        if not model.strip():
            raise ValueError('Editor model must be nonempty')
        connection = providers.connect(args.editor_provider, model, base_url=args.editor_base_url,
                                       key_env=args.editor_key_env, timeout=1200)
        replies = [None]*args.rounds
    simulator = evaluator()
    args.output.mkdir(parents=True)
    report = dict(status='running', editor_mode=args.editor_mode, editor_model=model,
                  editor_provider=args.editor_provider if model else None, notes=args.notes,
                  settings=vars(args).copy(), simulator=simulator.metadata, candidates=[], selected='initial',
                  usage=dict(input_tokens=0, output_tokens=0))
    report['settings'] = {key: str(value) if isinstance(value, Path) else value
                          for key, value in report['settings'].items() if key != 'reply'}
    report_path = args.output/'report.json'
    (args.output/'prompt.txt').write_text(PROMPT.read_text())
    save_actor(args.output/'base.npz', base)
    save_actor(args.output/'initial.npz', initial)
    report['base_sha256'] = hashlib.sha256((args.output/'base.npz').read_bytes()).hexdigest()
    report['initial_sha256'] = hashlib.sha256((args.output/'initial.npz').read_bytes()).hexdigest()

    def measure(weights, stage, count, seed):
        def progress(result):
            report[stage] = result
            write_json(report_path, report)
            print(f"{stage}: {result['successes']}/{len(result['episodes'])} successes", flush=True)
        return simulator.evaluate(weights, count, seed, progress=progress)

    write_json(report_path, report)
    try:
        best = initial
        baseline = measure(initial, 'practice_baseline', args.practice_episodes, args.seed)
        best_score = baseline['success_rate']
        messages = [{'role':'user', 'content':PROMPT.read_text()}]
        if args.base:
            # Describe the starting candidate exactly for feedback-driven continuation.
            edits = []
            for key in base:
                for index in zip(*np.nonzero(initial[key] != base[key])):
                    edits.append(f"{key}[{', '.join(map(str,index))}] = {initial[key][index].item()}")
            messages.append({'role':'assistant','content':'Current repair from the original weights.\n```edits\n'+'\n'.join(edits)+'\n```'})
            initial_feedback = feedback(simulator, initial, baseline, args.seed)
            (args.output/'initial-feedback.txt').write_text(initial_feedback)
            messages.append({'role':'user','content':initial_feedback})
        with connection as provider:
            for number, reply in enumerate(replies, 1):
                label = f'attempt-{number:02d}'
                if provider:
                    print(f'Requesting {label} from {model}', flush=True)
                    if args.editor_provider == 'openai':
                        turn = provider.text_turn(model, messages, max_output_tokens=args.editor_max_tokens,
                                                  reasoning_effort=args.reasoning_effort)
                    else:
                        turn = provider.turn(model, messages, use_tools=False, max_output_tokens=args.editor_max_tokens,
                                             system='Follow the policy specification and return a final edits block. Use no tools.')
                    (args.output/f'{label}-provider.json').write_text(json.dumps(turn.log, indent=2))
                    reply = '\n\n'.join(turn.text)
                    (args.output/f'{label}.md').write_text(reply)
                    report['usage']['input_tokens'] += turn.input_tokens
                    report['usage']['output_tokens'] += turn.output_tokens
                    if turn.refused or turn.calls:
                        raise ValueError('Editor refused or returned unexpected tool calls')
                    messages.extend(turn.history)
                else:
                    (args.output/f'{label}.md').write_text(reply)
                weights, count = apply_reply(reply, base)
                checkpoint = args.output/f'{label}.npz'
                save_actor(checkpoint, weights)
                result = measure(weights, label, args.practice_episodes, args.seed)
                kept = result['success_rate'] > best_score
                if kept:
                    best, best_score = weights, result['success_rate']
                    report['selected'] = label
                changed = sum(int(np.count_nonzero(weights[key] != base[key])) for key in base)
                report['candidates'].append(dict(name=label, statements=count, practice=result, kept=kept,
                    parameters_changed=changed, fraction_changed=changed / sum(value.size for value in base.values()),
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
                write_json(report_path, report)
                if provider and number < len(replies):
                    text = feedback(simulator, weights, result, args.seed)
                    (args.output/f'{label}-feedback.txt').write_text(text)
                    messages.append({'role':'user', 'content':text})
        save_actor(args.output/'selected.npz', best)
        report['selected_sha256'] = hashlib.sha256((args.output/'selected.npz').read_bytes()).hexdigest()
        measure(initial, 'before', args.eval_episodes, args.eval_seed)
        measure(best, 'after', args.eval_episodes, args.eval_seed)
        report['status'] = 'complete'
        write_json(report_path, report)
        print(f"Held-out success: {report['before']['success_rate']:.1%} -> {report['after']['success_rate']:.1%}")
    except BaseException as error:
        report.update(status='failed', error=str(error))
        write_json(report_path, report)
        raise


def feedback(simulator, weights, result, seed):
    summary = {key:value for key,value in result.items() if key != 'episodes'}
    trajectory = simulator.trajectory(weights, seed)
    return ('Your policy was tested on practice episodes. Diagnose the failure and revise it. '
            'Return a complete edits block applied to the ORIGINAL rounded policy in the first prompt, '
            'including all earlier changes you want to retain. You have no tools.\n'
            +json.dumps(dict(practice=summary, recorded_episode=trajectory), indent=2))


def main(argv=None):
    from . import providers
    parser = argparse.ArgumentParser(prog='patchwork car', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prompt', help='Print the original public task and trained weights')
    apply = commands.add_parser('apply', help='Import numerical edits')
    apply.add_argument('reply', type=Path)
    apply.add_argument('output', type=Path)
    apply.add_argument('--base', type=Path, help='Apply incremental edits to this actor')
    ev = commands.add_parser('evaluate', help='Evaluate a saved actor')
    ev.add_argument('--weights', type=Path, required=True)
    ev.add_argument('--episodes', type=int, default=1024)
    ev.add_argument('--seed', type=int, default=40000)
    ev.add_argument('--output', type=Path, required=True)
    command = commands.add_parser('run', help='Repair a trained policy and benchmark it')
    command.add_argument('--editor-mode', choices=('api','manual'), required=True)
    command.add_argument('--reply', type=Path, action='append')
    command.add_argument('--base', type=Path, help='Start from an existing repair, retaining it as a candidate')
    command.add_argument('--editor-provider', choices=providers.PROVIDERS, default='anthropic')
    command.add_argument('--editor-model')
    command.add_argument('--editor-base-url')
    command.add_argument('--editor-key-env')
    command.add_argument('--editor-max-tokens', type=int, default=16384)
    command.add_argument('--reasoning-effort', choices=('low','medium','high','xhigh','max'), default='high', help='OpenAI editor reasoning effort')
    command.add_argument('--rounds', type=int, help='API default: 2; manual default: 1')
    command.add_argument('--practice-episodes', type=int, default=64)
    command.add_argument('--eval-episodes', type=int, default=1024)
    command.add_argument('--seed', type=int, default=30000)
    command.add_argument('--eval-seed', type=int, default=40000)
    command.add_argument('--notes', default='')
    command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prompt':
        print(PROMPT.read_text(), end='')
    elif args.command == 'apply':
        weights, count = apply_reply(args.reply.read_text(), load_actor(args.base) if args.base else base_actor())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_actor(args.output, weights)
        print(f'Saved {count} edits to {args.output}')
    elif args.command == 'evaluate':
        if args.episodes < 1 or not 0 <= args.seed <= 2**32-args.episodes:
            raise ValueError('Use a positive episode count and uint32 seeds')
        if args.output.exists():
            raise ValueError('Output already exists')
        weights = load_actor(args.weights)
        simulator = evaluator()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = simulator.evaluate(weights, args.episodes, args.seed,
                                    progress=lambda result:write_json(args.output,dict(status='running', **result)))
        write_json(args.output, dict(status='complete', simulator=simulator.metadata, **result))
        print(f"Success: {result['successes']}/{args.episodes}")
    else:
        run(args)
