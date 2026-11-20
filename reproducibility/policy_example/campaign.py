"""Develop, then freeze and compare two recovery execution strategies on CPU."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'frozen-source'))
from auditembodied import l4_lift as runner
from observation import Observer,render
from geometry import fit
from native import NativeRecorder
from closed_loop import recover

CONFIG=json.loads((HERE/'config.json').read_text())


class RecordedObserver(Observer):
    def __init__(self):super().__init__(True);self.bundles=[]
    def __call__(self,*args):
        d=super().__call__(*args)
        self.bundles.append((d,self.images))
        return d


def one(seed,scenario,method,load,window,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    case=f'{seed}-{scenario}-{load}-{window}-{method}'
    destination=folder/(case+'.json')
    assert not destination.exists()
    fit(json.loads((HERE/'warmup-port.json').read_text()))
    runner._import_robosuite(False)
    from robosuite.environments.manipulation import two_arm_lift as suite
    factory=suite.PotWithHandlesObject;observer=RecordedObserver();recorders=[]

    class Environment(runner.TwoArmLiftEnv):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);self.raw.has_offscreen_renderer=True
        def reset(self):
            super().reset();before=runner.snapshot(self);render(self)
            assert runner.snapshot(self)==before
            recorder=NativeRecorder(self);recorders.append(recorder)

    def scaled(*a,**kw):kw['density']=1000.*(1.5 if load=='density150' else 1.);return factory(*a,**kw)
    started=time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(runner,'TwoArmLiftEnv',Environment))
        stack.enter_context(patch.object(suite,'PotWithHandlesObject',scaled))
        row=runner.run_trial(seed,scenario,config=runner.LiftConfig(**CONFIG),
            selector=lambda port:{'candidate':'R_global'},account_selector_latency=window=='actual',
            fixed_selector_wait_steps=None if window=='actual' else 20,
            regrasp_hook=observer,global_executor=recover if method=='closed_loop' else None)
    row.update(method=method,load=load,window=window,wall_seconds=time.perf_counter()-started)
    native_path=folder/(case+'-native.npz')
    row['native_samples']=recorders[0].save(native_path);row['native_file']=native_path.name
    image_folder=folder/(case+'-rgb');image_folder.mkdir(exist_ok=False)
    import imageio.v3 as iio
    from geometry import CAMERAS
    for j,(decision,images) in enumerate(observer.bundles):
        for c,rgb in zip(CAMERAS,images):iio.imwrite(image_folder/f'{j:02d}-{c}.png',rgb)
    row['rgb_directory']=image_folder.name;row['rgb_observations']=len(observer.bundles)
    row['audit']={'label':'error' if row.get('error') else 'A' if row['success'] and not row['protection_violation'] else 'B' if row['success'] else 'D' if row['protection_violation'] else 'C',
                  'protected_completion':bool(row['success'] and not row['protection_violation'])}
    goal=bool(row.get('task_complete',False))
    protected=bool(not row['protection_violation'] and row.get('table_cycle_verified',False))
    row['uniform_audit']={'V_path':True,'C':goal,'P':protected,
        'I':bool(row.get('table_cycle_verified',False)),
        'Y':bool(goal and protected),
        'label':'A' if goal and protected else 'B' if goal else 'C' if protected else 'D'}
    destination.write_text(json.dumps(row,separators=(',',':'))+'\n')
    print(json.dumps({'case':case,'label':row['uniform_audit']['label'],'legacy_label':row['audit']['label'],'complete':row.get('task_complete'),
        'time':row.get('total_post_fault_seconds'),'error':row.get('error'),
        'holds':row.get('final',{}).get('holds'),'tilt':row.get('final',{}).get('tilt_deg'),
        'failure':row.get('failure_reason'),'observations':len(observer.bundles),
        'max_gap':row['protection']['max_unsupported_seconds'],'wall':row['wall_seconds']}),flush=True)
    return row


def seed_job(job):
    seed,index,phase=job
    if phase=='formal':
        protocol=json.loads((HERE/'protocol.json').read_text())
        for name,digest in protocol['source_sha256'].items():
            assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    folder=HERE/('formal' if phase=='formal' else 'development/engineering-v2')
    results=[]
    for qi in range(3):
        scenario=('left_loss','right_loss','sham')[(qi+index)%3]
        for wi in range(2):
            window=('actual','1000ms')[(wi+index)%2]
            for li in range(2):
                load=('native','density150')[(li+index+qi)%2]
                methods=('retarget','closed_loop') if (index+qi+wi+li)%2==0 else ('closed_loop','retarget')
                for method in methods:
                    results.append(one(seed,scenario,method,load,window,folder))
    shard=folder/f'seed-{seed}.jsonl'
    with shard.open('x') as f:
        for row in results:f.write(json.dumps(row,separators=(',',':'))+'\n')
    return {'seed':seed,'completed':len(results)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=63)
    p.add_argument('--scenario',choices=('left_loss','right_loss','sham'),default='left_loss')
    p.add_argument('--method',choices=('retarget','closed_loop'),default='closed_loop')
    p.add_argument('--load',choices=('native','density150'),default='density150')
    p.add_argument('--window',choices=('actual','1000ms'),default='actual')
    p.add_argument('--out',type=Path,default=HERE/'development/pilot1')
    p.add_argument('--phase',choices=('single','engineering','formal'),default='single')
    p.add_argument('--workers',type=int,default=10)
    args=p.parse_args()
    if args.phase=='single':
        one(args.seed,args.scenario,args.method,args.load,args.window,args.out);return
    seeds=[64,65] if args.phase=='engineering' else list(range(63000,63020))
    if args.phase=='formal':
        assert (HERE/'engineering-check.json').exists() and (HERE/'protocol.json').exists()
        assert not (HERE/'formal').exists()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for status in pool.map(seed_job,[(s,i,args.phase) for i,s in enumerate(seeds)]):
            print(json.dumps(status),flush=True)


if __name__=='__main__':main()
