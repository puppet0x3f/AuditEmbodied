"""Prospective waiting-decision intervention; original models and skills frozen."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'frozen-source'))
from auditembodied import l4_lift as runner
from policy import WaitingPolicy,MODES

SCENARIOS=('left_loss','right_loss','coupled_height','sham')
WAITS={'actual':0,'950ms':19}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def admission(row):
    keeper={'R_left':1,'R_right':0}.get(row['candidate'])
    t=row['wait_decision']['observed_time']
    e=row['post_fault'] if abs(t-row['prefix_state']['time'])<1e-8 else next(f['evaluator'] for f in row['trace'] if abs(f['time']-t)<1e-8)
    observed=bool(e['holds'][keeper]) if keeper is not None else True
    entered=bool(row['post_wait_evaluator']['holds'][keeper]) if keeper is not None else True
    p=not row['protection_violation'];accepted=bool(row['success'])
    label='E' if not entered else 'A' if accepted and p else 'B' if accepted else 'C' if p else 'D'
    return {'complete':True,'V_obs':observed,'V_entry':entered,'protected_completion':bool(entered and accepted and p),
            'label':label,'cost_identity':abs(row['total_post_fault_seconds']-row['inference_wait_seconds']-row['skill_seconds']-row['hold_seconds'])<1e-8}


def verify(p):
    assert sha(ROOT/'direct-selected.json')==p['model_sha256']
    assert sha(REPO/'outputs/p0-20261109/direct-selected.json')==p['model_sha256']
    assert all(sha(ROOT/path)==digest for path,digest in p['source_sha256'].items())


def one(seed,scenario,mode,window,index,protocol):
    setup_started=time.perf_counter()
    values=json.loads((ROOT/'direct-selected.json').read_text())['mlp']
    policy=WaitingPolicy(values,mode)
    warmup=ROOT/'warmup-observation.json'
    if warmup.exists():
        port=json.loads(warmup.read_text())
        policy.initial(port)
        policy([{'elapsed_seconds':j/20,'observation':port} for j in range(21)])
    setup_seconds=time.perf_counter()-setup_started
    started=time.perf_counter()
    row=runner.run_trial(seed,scenario,config=runner.LiftConfig(**protocol['test_config']),
        selector=policy.initial,account_selector_latency=True,wait_policy=policy,decision_hold_steps=WAITS[window])
    row.update(method=mode,window=window,execution_index=index,wall_seconds=time.perf_counter()-started,
        model_setup_wall_seconds=setup_seconds,model_prewarmed=warmup.exists())
    if 'wait_decision' in row:row['audit']=admission(row)
    return row


def seed_job(job):
    seed,index,engineering=job
    p=json.loads((ROOT/'protocol.json').read_text());verify(p)
    path=ROOT/('engineering' if engineering else 'seed-shards')/f'{seed}.jsonl';path.parent.mkdir(exist_ok=True)
    assert not path.exists()
    count=0
    with path.open('x') as f:
        for qi in range(4):
            q=SCENARIOS[(qi+index)%4]
            for wi in range(2):
                w=tuple(WAITS)[(wi+index)%2];shift=(index+qi+wi)%3
                for m in MODES[shift:]+MODES[:shift]:
                    row=one(seed,q,m,w,count,p);f.write(json.dumps(row)+'\n');f.flush();count+=1
                    print(json.dumps({'seed':seed,'completed':count,'scenario':q,'window':w,'method':m,
                        'candidate':row['candidate'],'label':row.get('audit',{}).get('label'),'error':row.get('error')}),flush=True)
    verify(p)
    return seed,count


def prepare():
    assert not (ROOT/'protocol.json').exists()
    prior=json.loads((REPO/'outputs/p1-load-transfer-20261109/protocol.json').read_text())
    model=json.loads((ROOT/'direct-selected.json').read_text())['mlp']
    p={'question':'Does using the observed waiting interval change prospective frozen-MLP recovery decisions and their full-path outcomes?',
        'test_seeds':list(range(60000,60020)),'engineering_seeds':[60],
        'methods':MODES,'scenarios':SCENARIOS,'waits':WAITS,'expected_episodes':480,
        'model_sha256':sha(ROOT/'direct-selected.json'),'threshold':model['threshold'],
        'test_config':prior['test_config'],
        'observations':'All methods record the same20Hz positions/fingers/registered-reference stream during the uncommitted physical hold. No object truth, seed, fault identity or future outcome reaches selection.',
        'policies':{'cached':'Use initial observation only.',
            'refreshed':'Use final observed state before the final decision callback.',
            'wait_conditioned':'Use minimum candidate probability and conjunction of start masks over the entire observed wait; same MLP, threshold and nominal-cost order. A conservative history wrapper, not a trained delay predictor.'},
        'timing':'Initial callback and final history/decision service each replay ceil(measured seconds/.05) holding steps. The independent intervention adds0 or19 physical holds. All extra service, skill motion and final1s verification are charged; simulator throughput wall time is separate.',
        'roles':'Uncommitted initial/budget holding has shared guards. After final choice, its local keeper obligation covers the final service hold and local execution. Explicit per-frame skill tags prevent retroactive role assignment.',
        'controls':'Healthy sham is separate. No skill gain, waypoint, force limit, duration, model or score threshold changes. No tuning after engineering or fresh-seed outcomes.',
        'statistics':'20 whole initialization seeds, preserving fault/window/method dependence;2000 bootstrap draws with seed20261110. Failures stay in all denominators; common-protected cost separately.',
        'source_sha256':{str(f.relative_to(ROOT)):sha(f) for f in [ROOT/'campaign.py',ROOT/'policy.py',*sorted((ROOT/'frozen-source/auditembodied').glob('*.py'))]}}
    (ROOT/'protocol.json').write_text(json.dumps(p,indent=2)+'\n');print('Protocol prepared before engineering and formal outcomes.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','engineering','engineering_warm','run'));parser.add_argument('--workers',type=int,default=6);args=parser.parse_args()
    if args.phase=='prepare':prepare();return
    p=json.loads((ROOT/'protocol.json').read_text());verify(p)
    if args.phase=='engineering':seed_job((60,0,True));return
    if args.phase=='engineering_warm':seed_job((61,1,True));return
    for name in ('seed-audit-local.json','seed-audit-remote.json'):
        check=json.loads((ROOT/name).read_text());assert check['fresh'] and not check['overlap'] and not check['errors'],name
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        list(pool.map(seed_job,[(s,i,False) for i,s in enumerate(p['test_seeds'])]))
    with (ROOT/'trials.jsonl').open('x') as f:
        for s in p['test_seeds']:f.write((ROOT/'seed-shards'/f'{s}.jsonl').read_text())
    verify(p)


if __name__=='__main__':main()
