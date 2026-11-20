"""Exercise the integrated car workflow and streaming transport without paid calls."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import nullcontext

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from patchwork import car, providers, claude
from patchwork.cli import main
from patchwork.car_weights import base_actor, apply_reply, load_actor

REPLY = '```edits\nactor_head.b[1] = 1000\n```'
BAD = '```edits\nactor_head.b[1] = -1000\n```'


class Simulator:
    metadata = {'test':True}

    def evaluate(self, weights, episodes, seed, progress=None):
        success = float(weights['actor_head.b'][1]) > 999
        result = dict(successes=episodes if success else 0, red_collisions=0,
                      success_rate=float(success), mean_steps=73,
                      episodes=[dict(seed=seed+i,success=success) for i in range(episodes)])
        if progress:
            progress(result)
        return result

    def trajectory(self, weights, seed):
        return dict(seed=seed,trajectory=[{'step':32,'x':1.4}])


def run():
    base=base_actor()
    assert len(base)==14
    for text in ('```edits\nimport os\n```','```edits\nactor_head.b[0] = float("nan")\n```'):
        try:
            apply_reply(text,base)
        except (ValueError,TypeError):
            pass
        else:
            raise AssertionError('Unsafe code accepted')
    reset,_=apply_reply('```edits\nZERO actor_head.W\n```',base)
    assert np.count_nonzero(reset['actor_head.W'])==0
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        reply=root/'reply.md'
        reply.write_text(REPLY)
        with patch.object(car,'evaluator',return_value=Simulator()), patch.object(providers,'connect',side_effect=AssertionError('manual called provider')):
            assert main(['car','run','--editor-mode','manual','--reply',str(reply),'--output',str(root/'manual'),
                         '--practice-episodes','2','--eval-episodes','3'])==0
        report=json.loads((root/'manual/report.json').read_text())
        assert report['before']['successes']==0 and report['after']['successes']==3
        assert report['status']=='complete' and report['selected']=='attempt-01'
        assert load_actor(root/'manual/selected.npz')['actor_head.b'][1]==1000
        assert main(['car','run','--editor-mode','manual','--reply',str(reply),'--seed','40000','--output',str(root/'invalid')])==1
        assert not (root/'invalid').exists()

        for provider_name in providers.PROVIDERS:
            messages=[]
            def turn(model, history, **kwargs):
                messages.append(list(history))
                text=REPLY if len(messages)==1 else BAD
                return SimpleNamespace(log=[],text=[text],history=[{'role':'assistant','content':text}],
                                       calls=[],refused=False,input_tokens=4,output_tokens=8)
            provider=SimpleNamespace(turn=turn,text_turn=turn)
            with patch.object(car,'evaluator',return_value=Simulator()), patch.object(providers,'connect',return_value=nullcontext(provider)):
                assert main(['car','run','--editor-mode','api','--editor-provider',provider_name,
                             '--editor-model','test','--output',str(root/provider_name),
                             '--practice-episodes','2','--eval-episodes','3'])==0
            report=json.loads((root/provider_name/'report.json').read_text())
            assert report['selected']=='attempt-01' and report['after']['successes']==3
            assert len(messages)==2 and 'recorded_episode' in messages[1][-1]['content']
            assert 'published' not in json.dumps(messages).lower()
            assert report['usage']==dict(input_tokens=8,output_tokens=16)

        def refused(*a,**kw):
            return SimpleNamespace(log=[],text=['refusal'],history=[],calls=[],refused=True,input_tokens=1,output_tokens=1)
        with patch.object(car,'evaluator',return_value=Simulator()), patch.object(providers,'connect',return_value=nullcontext(SimpleNamespace(text_turn=refused))):
            assert main(['car','run','--editor-mode','api','--editor-provider','openai','--editor-model','test',
                         '--output',str(root/'failed'),'--practice-episodes','2'])==1
        assert json.loads((root/'failed/report.json').read_text())['status']=='failed'
        assert (root/'failed/attempt-01.md').read_text()=='refusal'

    response={'output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':REPLY}]}],
              'usage':{'input_tokens':5,'output_tokens':6}}
    def transport(request):
        payload=json.loads(request.content)
        assert payload['stream'] and payload['tools']==[] and payload['reasoning']=={'effort':'high'}
        return httpx.Response(200,text='data: '+json.dumps({'type':'response.completed','response':response})+'\n\n',
                              headers={'Content-Type':'text/event-stream'})
    with httpx.Client(base_url='https://example.com',transport=httpx.MockTransport(transport)) as client:
        result=providers.OpenAIEditor(client).text_turn('test',[{'role':'user','content':'test'}])
        assert result.text==[REPLY] and result.input_tokens==5
    print('Car parsing, selection, feedback, modes, failure reports and streaming passed')


if __name__=='__main__':
    run()
