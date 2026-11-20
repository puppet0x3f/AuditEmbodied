"""Training-seed plumbing test with a deterministic stand-in, never used to fit."""
import importlib.util
import json
from pathlib import Path
from queue import Queue

from auditembodied.l0 import _import_robosuite

root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('round_live',root/'live.py')
live=importlib.util.module_from_spec(spec);spec.loader.exec_module(live)


class StandIn:
    def __init__(self):
        self.messages=[];self.answers=Queue();self.stdin=self;self.stdout=self
    def write(self,line):
        m=json.loads(line);self.messages.append(m)
        if m['op']=='scope':result={'candidate':'R_left','parallel_local':True}
        elif m['op']=='cache_start':result={'start_valid_score':1.,'start_cache_hit':False}
        elif m['op']=='choose_examples_shared':
            assert m['image']==self.messages[-2]['image']
            assert m['plans'][0]['prefix_observation_hold_steps']>0
            result={'records':[{'plan':p,'accepted':p['local_relevel_steps']>=40} for p in m['plans']]}
        else:raise ValueError(m['op'])
        self.answers.put(json.dumps(result)+'\n')
    def flush(self):pass
    def readline(self):return self.answers.get(timeout=60)


output=root/'integration-check';(output/'observations').mkdir(parents=True,exist_ok=True)
_import_robosuite(True);worker=StandIn()
row=live.live_trial(48,'left_loss','vlm_shared',worker,output)
assert not row.get('error'),row.get('error')
assert row['local_steps']['local_relevel_steps']==40
assert row['duration_selection']['prefix_observation_wait_steps']>0
assert not row['duration_selection']['deadline_missed']
assert abs(row['total_post_fault_seconds']-row['inference_wait_seconds']-row['skill_seconds']-row['hold_seconds'])<1e-7
(output/'report.json').write_text(json.dumps({'valid':True,'seed':48,'model':'deterministic stand-in',
    'scope':'One engineering integration rollout, excluded from training/dev/independent evaluation.',
    'messages':[m['op'] for m in worker.messages],'wait_seconds':row['inference_wait_seconds'],
    'total_post_fault_seconds':row['total_post_fault_seconds'],'duration_selection':row['duration_selection']},indent=2)+'\n')
print(json.dumps({'integration_passed':True,'wait_seconds':row['inference_wait_seconds']}),flush=True)
