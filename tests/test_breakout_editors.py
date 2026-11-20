"""Exercise manual and all provider modes without paid requests."""
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from patchwork import breakout, providers, claude
from patchwork.cli import main

REPLY = 'A simple test policy.\n```edits\nb3[1] = 1\n```'
REVISION = '```edits\nb3[1] = 0\n```'


def score(weights, episodes, seed, max_steps):
    reward = float(weights['b3'][1])
    return dict(mean=reward, standard_error=0, mean_steps=1,
                episodes=[dict(seed=seed, score=reward)])


def run():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        reply = root / 'reply.md'
        reply.write_text(REPLY)
        with patch.object(providers, 'connect', side_effect=AssertionError('Manual mode contacted API')), patch.object(breakout, 'evaluate', score):
            assert main(['breakout', 'run', '--editor-mode', 'manual', '--reply', str(reply),
                         '--output', str(root / 'manual'), '--episodes', '2']) == 0
        manual = json.loads((root / 'manual/report.json').read_text())
        assert manual['status'] == 'complete' and manual['after']['mean'] == 1
        assert manual['usage']['output_tokens'] == 0
        assert main(['breakout', 'run', '--editor-mode', 'manual', '--output', str(root / 'missing')]) == 1
        assert not (root / 'missing').exists()
        assert main(['breakout', 'run', '--editor-mode', 'manual', '--reply', str(reply),
                     '--eval-seed', '0', '--output', str(root / 'overlap')]) == 1

        client = httpx.Client
        for provider in ('openai', 'openai-compatible', 'anthropic'):
            requests = []

            def respond(request):
                payload = json.loads(request.content)
                assert payload['tools'] == []
                assert payload.get('max_output_tokens', payload.get('max_tokens')) == 1234
                requests.append(payload)
                text = REPLY if len(requests) == 1 else REVISION
                if len(requests) == 2:
                    history = payload.get('input', payload.get('messages'))
                    assert 'practice episodes' in history[-1]['content']
                if provider == 'openai':
                    return httpx.Response(200, json={'output': [{'type': 'message', 'role': 'assistant',
                        'content': [{'type': 'output_text', 'text': text}]}],
                        'usage': {'input_tokens': 10, 'output_tokens': 20}})
                return httpx.Response(200, json={'choices': [{'message': {'role': 'assistant', 'content': text}}],
                                                 'usage': {'prompt_tokens': 10, 'completion_tokens': 20}})

            def anthropic_create(**payload):
                assert payload['tools'] == [] and payload['max_tokens'] == 1234
                requests.append(payload)
                text = REPLY if len(requests) == 1 else REVISION
                block = SimpleNamespace(type='text', text=text, model_dump=lambda: {'type': 'text', 'text': text})
                return SimpleNamespace(content=[block], stop_reason='end_turn',
                                       usage=SimpleNamespace(input_tokens=10, output_tokens=20))

            api = SimpleNamespace(models=SimpleNamespace(retrieve=lambda model: None),
                                  messages=SimpleNamespace(create=anthropic_create))
            from contextlib import nullcontext
            flags = ['--editor-base-url', 'https://editor.example/v1'] if provider == 'openai-compatible' else []
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test'}), patch.object(
                providers.httpx, 'Client', side_effect=lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs)
            ), patch.object(claude, 'client', return_value=nullcontext(api)), patch.object(breakout, 'evaluate', score):
                assert main(['breakout', 'run', '--editor-mode', 'api', '--editor-provider', provider,
                             '--editor-model', 'test-model', '--rounds', '2', '--editor-max-tokens', '1234',
                             '--episodes', '2', '--output', str(root / provider), *flags]) == 0
            report = json.loads((root / provider / 'report.json').read_text())
            assert len(requests) == 2 and report['selected'] == 'attempt-01'
            assert len(report['candidates']) == 2 and report['after']['mean'] == 1
            assert report['usage'] == {'input_tokens': 20, 'output_tokens': 40}
            assert breakout.load_weights(root / provider / 'selected.npz')['b3'][1] == 1

        # Invalid API output is saved, and the run is explicitly marked failed.
        turn = SimpleNamespace(log=[], text=['invalid response'], history=[], calls=[], refused=False,
                               input_tokens=1, output_tokens=1)
        with patch.object(providers, 'connect', return_value=nullcontext(SimpleNamespace(turn=lambda *a, **kw: turn))), patch.object(breakout, 'evaluate', score):
            assert main(['breakout', 'run', '--editor-mode', 'api', '--output', str(root / 'invalid')]) == 1
        assert json.loads((root / 'invalid/report.json').read_text())['status'] == 'failed'
        assert (root / 'invalid/attempt-01.md').read_text() == 'invalid response'
    print('Manual/API modes, all adapters, revisions, selection and failure persistence passed.')


if __name__ == '__main__':
    run()
