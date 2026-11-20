"""Frozen lifting policies under a prespecified, uniformly denser pot."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'frozen-source'))
from auditembodied import l4_lift as runner
from auditembodied.admissible_mlp import StartGatedMLP
from auditembodied.l4_selector import ProprioRule

METHODS=('protected','gated_direct','global')
SCENARIOS=('left_loss','right_loss','coupled_height','sham')
WINDOWS=('actual','1000ms')
LOADS={'native':1.,'density150':1.5}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def scaled_factory(original, scale):
    def create(*args,**kwargs):
        kwargs['density']=float(kwargs.get('density',1000.))*scale
        return original(*args,**kwargs)
    return create


def physical_parameters(env):
    model=env.sim.model;body=env.raw.pot_body_id
    bodies=[]
    for i in range(model.nbody):
        ancestor=i
        while ancestor and ancestor!=body:ancestor=int(model.body_parentid[ancestor])
        if ancestor==body:bodies.append(i)
    geoms=[i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in bodies]
    actuators=[int(a) for r in env.raw.robots for a in r._ref_joint_gripper_actuator_indexes['right']]
    return {'density_kg_per_m3':float(env.raw.pot.density),
        'body_names':[model.body_id2name(i) for i in bodies],
        'body_mass_kg':np.asarray(model.body_mass)[bodies].tolist(),
        'body_inertia_kg_m2':np.asarray(model.body_inertia)[bodies].tolist(),
        'body_ipos':np.asarray(model.body_ipos)[bodies].tolist(),
        'body_iquat':np.asarray(model.body_iquat)[bodies].tolist(),
        'total_object_mass_kg':float(np.asarray(model.body_mass)[bodies].sum()),
        'geom_names':[model.geom_id2name(i) for i in geoms],
        **{name:np.asarray(getattr(model,name))[geoms].tolist()
           for name in ('geom_type','geom_size','geom_pos','geom_quat','geom_friction')},
        **{name:np.asarray(getattr(model,name))[actuators].tolist()
           for name in ('actuator_forcerange','actuator_gainprm','actuator_biasprm')}}


def admission(row):
    if 'post_wait_evaluator' not in row:return {'complete':False,'protected_completion':False,'label':'initialization_failure'}
    keeper={'R_left':1,'R_right':0}.get(row['candidate'])
    observed=bool(row['post_fault']['holds'][keeper]) if keeper is not None else True
    entered=bool(row['post_wait_evaluator']['holds'][keeper]) if keeper is not None else True
    p=not row['protection_violation']
    accepted=bool(row['success'])
    label='E' if not entered else 'A' if accepted and p else 'B' if accepted else 'C' if p else 'D'
    return {'complete':True,'V_obs':observed,'V_entry':entered,'V_path':entered,
        'required_keeper':keeper,'terminal_goal':bool(row['task_complete']),
        'accepted_completion':accepted,'protection':p,
        'protected_completion':bool(entered and accepted and p),'label':label,
        'observation_valid_entry_invalid':observed and not entered,
        'cost_identity':abs(row['total_post_fault_seconds']-row['inference_wait_seconds']-row['skill_seconds']-row['hold_seconds'])<1e-8}


def one(spec):
    seed,scenario,method,load,window,index=spec
    p=json.loads((ROOT/'protocol.json').read_text())
    values=json.loads((ROOT/'direct-selected.json').read_text())['mlp']
    selector=(ProprioRule(**p['rule']) if method=='protected' else StartGatedMLP(values)
              if method=='gated_direct' else lambda port:{'candidate':'R_global'})
    # Import the original simulator once, then modify only the object's density
    # at construction. MuJoCo compiles consistent mass and inertia itself.
    runner._import_robosuite(False)
    from robosuite.environments.manipulation import two_arm_lift as suite
    original=suite.PotWithHandlesObject
    seen=[]
    class Environment(runner.TwoArmLiftEnv):
        def reset(self):
            super().reset();seen.append(physical_parameters(self))
    started=time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(suite,'PotWithHandlesObject',scaled_factory(original,LOADS[load])))
        stack.enter_context(patch.object(runner,'TwoArmLiftEnv',Environment))
        row=runner.run_trial(seed,scenario,config=runner.LiftConfig(**p['test_config']),selector=selector,
            account_selector_latency=window=='actual',fixed_selector_wait_steps=None if window=='actual' else 20)
    row.update(method=method,load=load,window=window,execution_index=index,
        wall_seconds=time.perf_counter()-started,physical_parameters=seen[-1] if seen else None)
    row['audit']=admission(row)
    if window=='1000ms' and row['selector_wall_seconds']>1.:
        raise RuntimeError('prespecified wait exceeded; preserve shard and inspect timing before proceeding')
    return row


def seed_job(job):
    seed,index=job;folder=ROOT/'seed-shards';folder.mkdir(exist_ok=True)
    output=folder/f'{seed}.jsonl'
    assert not output.exists(),'do not overwrite observed episodes'
    specs=[]
    for qi in range(4):
        q=SCENARIOS[(qi+index)%4]
        for wi in range(2):
            w=WINDOWS[(wi+index)%2]
            for li in range(2):
                load=tuple(LOADS)[(li+index+qi)%2]
                shift=(index+qi+wi+li)%3
                for m in METHODS[shift:]+METHODS[:shift]:specs.append((seed,q,m,load,w,len(specs)))
    with output.open('x') as f:
        for spec in specs:
            row=one(spec);f.write(json.dumps(row)+'\n');f.flush()
            print(json.dumps({'seed':seed,'finished':spec[-1]+1,'total':len(specs),
                **{k:row.get(k) for k in ('scenario','method','load','window','candidate')},
                'label':row['audit']['label'],'error':row.get('error')}),flush=True)
    return seed,len(specs)


def verify(p):
    assert sha(ROOT/'direct-selected.json')==p['model_sha256']
    assert sha(REPO/'outputs/p0-20261109/direct-selected.json')==p['model_sha256']
    assert all(sha(ROOT/path)==digest for path,digest in p['source_sha256'].items())
    values=json.loads((ROOT/'direct-selected.json').read_text())['mlp']
    assert values['threshold']==p['threshold']


def prepare():
    assert not (ROOT/'protocol.json').exists()
    prior=json.loads((REPO/'outputs/agentchord-style-20261109/protocol.json').read_text())
    p={'scope':'Prespecified 1.5x object density transfer with same-seed native reference; no fitting or skill adaptation',
        'test_seeds':list(range(50000,50020)),'development_check_seed':60,
        'methods':METHODS,'scenarios':SCENARIOS,'loads':LOADS,'windows':WINDOWS,'expected_episodes':960,
        'model_parent':str(REPO/'outputs/p0-20261109/direct-selected.json'),
        'model_sha256':sha(ROOT/'direct-selected.json'),'threshold':prior['threshold'],'rule':prior['rule'],
        'test_config':prior['test_config'],
        'physical_intervention':'PotWithHandlesObject density1000 ->1500kg/m3 before model compilation. All object mass and inertia scale uniformly; geometry, friction, actuators, skills and thresholds unchanged.',
        'condition_selection':'Density1.5x and1s wait declared before development mechanics check or test. No grid search or outcome-conditioned replacement.',
        'physical_novelty':'All prior lifting protocols retain native pot density/mass/friction. Native1.0x reference is executed on the same new seeds, without changing the1.5x condition.',
        'observations':{'keys':['positions','fingers','reference_positions'],'rgb':False,
            'excluded':['object state','contact truth','load identity','fault name','seed','future samples']},
        'waiting':'Actual callback wall time rounded up to20Hz physical holding, or prespecified20-step(1s) physical holding after the same cached scope decision. Same registered feedback primitive and previous gripper signs; no free refresh.',
        'fixed_wait_service_check':'All measured callbacks must fit within1s; otherwise stop after preserving the evidence. No uncharged delay.',
        'admission':'Truth V_obs at common fault prefix; V_entry after actual wait. Main Y requires V_entry; truth never routes decisions. Continue/global retain no additional keeper prerequisite.',
        'protection':'Original20Hz zero-tolerance local keeper, registered-grasp/table absence>100ms, penetration>10mm, original task/global-interface acceptance and1s final hold.',
        'cost':'Full post-fault physical time=actual blocking hold+all skill motion+final1s hold. Selector callback timing, process wall time, and incomplete-cost cases recorded separately. No GPU or RGB service.',
        'healthy_control':'sham retains both grasps and executes fault-length holding with no perturbation; all three methods tested including fixed global.',
        'statistics':'20 initialization seeds; each load/window/method has60 fault states and20 healthy controls.2000 paired cluster-bootstrap draws; repeated methods/windows/loads do not increase independent seeds.',
        'order':'Rotate scenario, load and window order by seed; rotate three-method order within each condition. No method-conditioned seed selection.',
        'deployment':{'python':sys.executable,'control_hz':20,'workers':8,'camera_rendering':False,'GPU':False},
        'source_parent':'outputs/agentchord-style-20261109/frozen-source/auditembodied',
        'source_sha256':{str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'frozen-source').rglob('*.py'))}}
    p['source_sha256']['campaign.py']=sha(__file__)
    assert p['model_sha256']==prior['model_sha256']
    save(ROOT/'protocol.json',p);verify(p)
    print('Prespecified native/1.5x density, 20 new seeds, three frozen policies, two waits, four conditions:960 episodes.')


def engineering(workers):
    p=json.loads((ROOT/'protocol.json').read_text());verify(p)
    assert not (ROOT/'engineering.jsonl').exists()
    specs=[(60,q,m,load,w,i) for i,(load,q,w,m) in enumerate(
        (load,q,w,m) for load in LOADS for q in ('left_loss','sham') for w in WINDOWS for m in METHODS)]
    with (ROOT/'engineering.jsonl').open('x') as f:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for r in pool.map(one,specs):
                f.write(json.dumps(r)+'\n');f.flush()
                print(json.dumps({k:r.get(k) for k in ('load','scenario','method','window','candidate','error')}
                    |{'audit':r['audit'],'mass_kg':r['physical_parameters']['total_object_mass_kg']}),flush=True)
    verify(p)


def run(workers):
    p=json.loads((ROOT/'protocol.json').read_text());verify(p)
    for name in ('local','remote'):
        a=json.loads((ROOT/f'seed-audit-{name}.json').read_text())
        assert a['fresh'] and a['proposed_test_seeds']==p['test_seeds']
    gate=json.loads((ROOT/'engineering-check.json').read_text());assert gate['mechanics_pass']
    assert not (ROOT/'seed-shards').exists()
    save(ROOT/'formal-freeze.json',{'protocol_sha256':sha(ROOT/'protocol.json'),
        'model_sha256':p['model_sha256'],'campaign_sha256':sha(__file__),
        'engineering_sha256':sha(ROOT/'engineering-check.json'),
        'local_seed_audit_sha256':sha(ROOT/'seed-audit-local.json'),
        'remote_seed_audit_sha256':sha(ROOT/'seed-audit-remote.json')})
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(seed_job,job) for job in zip(p['test_seeds'],range(20))]
        for future in as_completed(futures):print(json.dumps({'seed_complete':future.result()}),flush=True)
    verify(p)
    with (ROOT/'trials.jsonl').open('x') as out:
        for seed in p['test_seeds']:
            with (ROOT/'seed-shards'/f'{seed}.jsonl').open() as src:shutil.copyfileobj(src,out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','engineering','run'))
    parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    if args.phase=='prepare':prepare()
    elif args.phase=='engineering':engineering(args.workers)
    else:run(args.workers)
