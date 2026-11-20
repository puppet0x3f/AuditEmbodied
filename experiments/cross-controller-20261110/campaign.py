"""Two independent global executors, shared physics and passive audit."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib,json,multiprocessing,sys,time
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'frozen-source'))
from auditembodied import l4_lift as runner
from observation import Observer,render
from geometry import fit
from controller import VisualRecovery
from sampling import NativeRecorder

METHODS=('retarget','visual')
SCENARIOS=('left_loss','right_loss','sham')
LOADS={'native':1.,'density150':1.5}
WINDOWS=('actual','1000ms')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def config():return json.loads((REPO/'outputs/interface-retarget-20261110/protocol.json').read_text())['test_config']

def one(seed,scenario,method,load,window,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    setup_started=time.perf_counter();fit(json.loads((ROOT/'warmup-port.json').read_text()))
    runner._import_robosuite(False)
    from robosuite.environments.manipulation import two_arm_lift as suite
    original=suite.PotWithHandlesObject
    observer=Observer(True);controller=VisualRecovery();instances=[]
    class Environment(runner.TwoArmLiftEnv):
        def __init__(self,*a,**k):
            super().__init__(*a,**k);self.raw.has_offscreen_renderer=True;instances.append(self)
        def reset(self):
            super().reset();before=runner.snapshot(self);render(self)
            assert runner.snapshot(self)==before
            self.recorder=NativeRecorder(self,runner.evaluate)
    def scaled(*a,**k):k['density']=1000.*LOADS[load];return original(*a,**k)
    setup=time.perf_counter()-setup_started;started=time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(suite,'PotWithHandlesObject',scaled))
        stack.enter_context(patch.object(runner,'TwoArmLiftEnv',Environment))
        row=runner.run_trial(seed,scenario,config=runner.LiftConfig(**config()),
            selector=lambda p:{'candidate':'R_global'},account_selector_latency=window=='actual',
            fixed_selector_wait_steps=None if window=='actual' else 20,
            regrasp_hook=observer if method=='retarget' else None,
            global_executor=controller if method=='visual' else None)
    wall=time.perf_counter()-started
    archive=time.perf_counter();env=instances[0]
    row.update(method=method,load=load,window=window,wall_seconds=wall,algorithm_setup_wall_seconds=setup,
        native_recording=env.recorder.save(folder/'native500.npz'))
    images=controller.images if method=='visual' else [np.stack(observer.images)] if observer.images else []
    if images:np.savez_compressed(folder/'rgb.npz',images=np.stack(images))
    row['archive_wall_seconds']=time.perf_counter()-archive
    row['safe_success']=bool(row['success'] and not row['protection_violation'])
    row['label']=('A' if row['safe_success'] else 'B') if row['success'] else ('D' if row['protection_violation'] else 'C')
    row['full_process_wall_seconds']=time.perf_counter()-setup_started
    (folder/'trial.json').write_text(json.dumps(row)+'\n')
    print(json.dumps({k:row.get(k) for k in ['seed','scenario','method','load','window','label','failure_reason','error','controller_stop_reason','total_post_fault_seconds','inference_wait_seconds','wall_seconds']}),flush=True)
    return row

def verify(p):
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in p['preserved_files'].items():assert sha(REPO/n)==h,n

def seed_job(job):
    seed,index=job;p=json.loads((ROOT/'protocol.json').read_text());verify(p)
    folder=ROOT/'formal'/str(seed);folder.mkdir(parents=True,exist_ok=False)
    rows=[]
    with (folder/'trials.jsonl').open('x') as stream:
        for qi in range(3):
            q=SCENARIOS[(qi+index)%3]
            for wi in range(2):
                w=WINDOWS[(wi+index)%2]
                for li in range(2):
                    load=tuple(LOADS)[(li+index+qi)%2];shift=(index+qi+wi+li)%2
                    for m in METHODS[shift:]+METHODS[:shift]:
                        row=one(seed,q,m,load,w,folder/f'{q}-{load}-{w}-{m}')
                        stream.write(json.dumps(row)+'\n');stream.flush();rows.append(row)
    verify(p);return seed,len(rows)

def freeze():
    assert not (ROOT/'protocol.json').exists()
    files=[ROOT/n for n in ['campaign.py','controller.py','sampling.py','geometry.py','observation.py','warmup-port.json']]
    files+=sorted((ROOT/'frozen-source/auditembodied').glob('*.py'))
    p={'question':'Do cumulative path and waiting effects persist with an independently implemented RGB feedback executor?',
        'seeds':list(range(62000,62020)),'methods':METHODS,'scenarios':SCENARIOS,'loads':LOADS,'windows':WINDOWS,
        'expected_episodes':480,'test_config':config(),'source_sha256':{str(f.relative_to(ROOT)):sha(f) for f in files},
        'preserved_files':json.loads((ROOT/'preserved-files.json').read_text()),
        'observations':'Same four calibrated320x320 RGB cameras, known CAD/color prior, registered robot poses, robot positions/orientations/finger encoders. No object/contact truth, seed/fault/load labels or future data enter either policy.',
        'methods_definition':{'retarget':'Frozen prior position-only correction: original full global controller, one RGB fit and2s inspection slot.',
            'visual':'Independent feedback controller starts at fault: retain loaded wrist orientation; RGB-height-guided vertical setdown; table-supported planar recentering and repeated height checks; geometric release check; repeated6DOF pickup pose estimates and contact-alignment refresh; error-driven wrist/translation servo, target-reach check and at most2grasp attempts with8mm encoder load test; lift preserves established wrist relation.'},
        'timing':'Initial actual callback is charged by ceil(service/50ms), or total20steps in1000ms condition. Every visual service replays ceil(measured/50ms) physical holds before consuming the snapshot. Baseline preserves its original2s inspection slot. All controls, retries and final1s hold count. Passive audit, image archive and total process wall are separately measured.',
        'audit':'Unchanged20Hz primary guards and global interface acceptance. Passive500Hz grasp/table/contact force norms/center/tilt recorded from the first physical closing tick. Force norms do not imply upward support or absence of freefall.',
        'statistics':'20 whole initialization seeds per paired contrast;40fault states and20healthy per load/window/method. No unseen-load claim; loads were used in development.',
        'scope':'One registered global skill call; internal grasp attempts are recorded and are not relabeled as separate registered calls.'}
    (ROOT/'protocol.json').write_text(json.dumps(p,indent=2)+'\n');print('Frozen',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['trial','freeze','run'])
    p.add_argument('--seed',type=int,default=60);p.add_argument('--scenario',default='left_loss',choices=SCENARIOS)
    p.add_argument('--method',default='visual',choices=METHODS);p.add_argument('--load',default='density150',choices=LOADS)
    p.add_argument('--window',default='actual',choices=WINDOWS);p.add_argument('--tag');p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    if a.phase=='trial':
        if not a.tag:p.error('--tag required')
        one(a.seed,a.scenario,a.method,a.load,a.window,ROOT/'development'/a.tag);return
    if a.phase=='freeze':freeze();return
    protocol=json.loads((ROOT/'protocol.json').read_text());verify(protocol)
    for name in ['seed-audit-local.json','seed-audit-remote.json']:
        audit=json.loads((ROOT/name).read_text());assert audit['fresh'] and not audit['overlap'] and not audit['errors']
    assert not (ROOT/'trials.jsonl').exists()
    (ROOT/'formal-start.json').write_text(json.dumps({'protocol_sha256':sha(ROOT/'protocol.json'),
        'workers':a.workers,'seeds':protocol['seeds'],'performance_gate':False},indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(seed_job,[(s,i) for i,s in enumerate(protocol['seeds'])]):print('Finished seed',result,flush=True)
    with (ROOT/'trials.jsonl').open('x') as out:
        for seed in protocol['seeds']:out.write((ROOT/'formal'/str(seed)/'trials.jsonl').read_text())
    verify(protocol)

if __name__=='__main__':main()
