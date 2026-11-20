"""Three controls for observation-driven pickup: 720 new episodes, no training."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
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
from observation import Observer,render
from geometry import CAMERAS,fit

METHODS=('fixed','refresh','retarget')
SCENARIOS=('left_loss','right_loss','sham')
LOADS={'native':1.,'density150':1.5}
WINDOWS=('actual','1000ms')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(p):
    assert all(sha(ROOT/name)==digest for name,digest in p['source_sha256'].items())
    for name,digest in p['original_model_sha256'].items():assert sha(REPO/name)==digest


def admission(row):
    if 'post_wait_evaluator' not in row:return {'complete':False,'protected_completion':False,'label':'initialization_failure'}
    accepted=bool(row['success']);protected=not row['protection_violation']
    return {'complete':True,'V_entry':True,'label':'A' if accepted and protected else 'B' if accepted else 'C' if protected else 'D',
        'protected_completion':bool(accepted and protected),
        'cost_identity':abs(row['total_post_fault_seconds']-row['inference_wait_seconds']-row['skill_seconds']-row['hold_seconds'])<1e-8}


def one(seed,scenario,method,load,window,index,p,engineering):
    setup_started=time.perf_counter()
    fit(json.loads((ROOT/'warmup-port.json').read_text()))
    runner._import_robosuite(False)
    from robosuite.environments.manipulation import two_arm_lift as suite
    original=suite.PotWithHandlesObject
    observation=Observer(method=='retarget') if method!='fixed' else None
    class Environment(runner.TwoArmLiftEnv):
        def __init__(self,*a,**k):
            super().__init__(*a,**k);self.raw.has_offscreen_renderer=True
        def reset(self):
            super().reset()
            before=runner.snapshot(self);render(self)
            assert runner.snapshot(self)==before
            self.render_setup_done=True
    def scaled(*a,**k):k['density']=1000.*LOADS[load];return original(*a,**k)
    setup=time.perf_counter()-setup_started
    started=time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(suite,'PotWithHandlesObject',scaled))
        stack.enter_context(patch.object(runner,'TwoArmLiftEnv',Environment))
        row=runner.run_trial(seed,scenario,config=runner.LiftConfig(**p['test_config']),
            selector=lambda port:{'candidate':'R_global'},account_selector_latency=window=='actual',
            fixed_selector_wait_steps=None if window=='actual' else 20,regrasp_hook=observation)
    row.update(method=method,load=load,window=window,execution_index=index,
        wall_seconds=time.perf_counter()-started,algorithm_setup_wall_seconds=setup)
    row['audit']=admission(row)
    # The audit reads truth only after the observation-driven policy has completed.
    if 'regrasp_decision' in row:
        d=row['regrasp_decision'];t=d['observed_time']
        at=next(f for f in row['trace'] if abs(f['time']-t)<1e-8)
        d['observation_truth']=at['evaluator']
        d['observation_state_robot']=at['robot']
        if d['estimate']['valid']:
            truth=np.asarray(at['evaluator']['handle_positions'])
            target=np.asarray(d['estimate']['targets'])
            center=np.asarray(d['estimate']['center']);rot=np.asarray(d['estimate']['rotation'])
            handles=center+np.asarray([[0.,.155,.06],[0.,-.155,.06]])@rot.T
            d['estimated_handle_error_m']=np.linalg.norm(handles-truth,axis=1).tolist()
            d['retarget_grip_to_handle_error_m']=np.linalg.norm(target-truth,axis=1).tolist()
            d['fixed_grip_to_handle_error_m']=np.linalg.norm(np.asarray(d['input']['reference_targets'])-truth,axis=1).tolist()
        image_dir=ROOT/('engineering-campaign/images' if engineering else 'images')/str(seed)/f'{scenario}-{load}-{window}-{method}'
        image_dir.mkdir(parents=True,exist_ok=False)
        for camera,rgb in zip(CAMERAS,observation.images):
            import imageio.v3 as iio
            iio.imwrite(image_dir/f'{camera}.png',rgb)
        d['image_directory']=str(image_dir.relative_to(ROOT))
    return row


def seed_job(job):
    seed,index,engineering=job;p=json.loads((ROOT/'protocol.json').read_text());verify(p)
    folder=ROOT/('engineering-campaign' if engineering else 'seed-shards');folder.mkdir(exist_ok=True)
    path=folder/f'{seed}.jsonl';assert not path.exists()
    count=0
    with path.open('x') as f:
        for qi in range(3):
            q=SCENARIOS[(qi+index)%3]
            for wi in range(2):
                w=WINDOWS[(wi+index)%2]
                for li in range(2):
                    load=tuple(LOADS)[(li+index+qi)%2];shift=(index+qi+wi+li)%3
                    for m in METHODS[shift:]+METHODS[:shift]:
                        row=one(seed,q,m,load,w,count,p,engineering);f.write(json.dumps(row)+'\n');f.flush();count+=1
                        print(json.dumps({'seed':seed,'completed':count,'scenario':q,'load':load,'window':w,'method':m,
                            'label':row['audit']['label'],'error':row.get('error')}),flush=True)
    verify(p);return seed,count


def prepare():
    assert not (ROOT/'protocol.json').exists()
    old=json.loads((REPO/'outputs/p1-load-transfer-20261109/protocol.json').read_text())
    preserved=json.loads((ROOT/'preserved-original-files.json').read_text())
    p={'question':'Does RGB-based pickup position retargeting restore successor compatibility under a fixed actuator controller?',
        'test_seeds':list(range(61000,61020)),'engineering_seeds':[61,62],
        'methods':METHODS,'scenarios':SCENARIOS,'loads':LOADS,'windows':WINDOWS,'expected_episodes':720,
        'test_config':old['test_config'],
        'original_model_sha256':{n:h for n,h in preserved.items() if n.endswith(('.npz','direct-selected.json'))},
        'original_mlp_threshold':old['threshold'],'scope':'All three execute fixed global; no learned selector or training is involved.',
        'observations':'Four calibrated static RGB cameras at320x320, known pot CAD/colors and registered robot poses. No depth, object pose/contact truth, seed, fault or load enters the fitter.',
        'estimator':'32 support directions for each color silhouette, fixed RGB dominance rule and known CAD convex hulls; robust6DOF fit with60 maximum NumPy LM iterations. Camera row convention calibrated on development seed60.',
        'methods_definition':{'fixed':'Original global targets and no post-setdown inspection.',
            'refresh':'Acquire and fit RGB after global_clear; discard estimated targets.',
            'retarget':'Same acquisition/fit as refresh; update hover/descend/close positions using estimated SE3. Registered wrist targets, gains, gripper commands and phase step counts remain unchanged.'},
        'timing':'Initial measured constant-global decision rounds to50ms; delayed window has20 physical steps total instead. Refresh/retarget physically replay max(2s,ceil(full measured callback/50ms)). The common2s inspection slot contains measured acquisition/fit and explicitly charged padding. Image archival and audit are outside policy service and separately logged as process wall time. No free final refresh.',
        'prewarm':'Same camera contexts initialized before bootstrap for all arms, without stepping physics or editing state; fitter prewarmed on fixed saved development seed60 port. Post-fault acquisition/fit remains measured and charged.',
        'protection':'Original native grasp/table gap>100ms, interarm penetration>10mm, verified setdown/regrasp cycle and1s final hold. No relaxed labels or changed acceptance.',
        'feasibility':'Development seed60 confirms accurate RGB target localization and successful native/healthy execution. Heavy unilateral position-only probes reacquire the high handle while the low remains constrained. Full-frame wrist variants also fail and are preserved separately; position-only selected to isolate target correction with unchanged controller, not to claim a complete skill repair.',
        'physical_novelty':'These loads were already diagnosed. This is a new-seed causal component experiment, not a claim of unseen-load development generalization.',
        'statistics':'20 whole initialization seeds,2000 bootstrap draws seed20261111; fault denominators40 and healthy20 per load/window/method. Repeated windows/loads/methods do not increase independent seeds.',
        'source_sha256':{str(f.relative_to(ROOT)):sha(f) for f in [ROOT/'campaign.py',ROOT/'observation.py',ROOT/'geometry.py',ROOT/'warmup-port.json',*sorted((ROOT/'frozen-source/auditembodied').glob('*.py'))]}}
    (ROOT/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    print('Development-selected component frozen before engineering campaign and fresh test outcomes.')


def main():
    a=argparse.ArgumentParser();a.add_argument('phase',choices=('prepare','engineering','run'));a.add_argument('--workers',type=int,default=8);args=a.parse_args()
    if args.phase=='prepare':prepare();return
    p=json.loads((ROOT/'protocol.json').read_text());verify(p)
    seeds=p['engineering_seeds'] if args.phase=='engineering' else p['test_seeds']
    if args.phase=='run':
        for n in ('seed-audit-local.json','seed-audit-remote.json'):
            r=json.loads((ROOT/n).read_text());assert r['fresh'] and not r['overlap'] and not r['errors']
        assert (ROOT/'engineering-check.json').exists()
        (ROOT/'formal-freeze.json').write_text(json.dumps({'protocol_sha256':sha(ROOT/'protocol.json'),'seeds':seeds},indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        list(pool.map(seed_job,[(s,i,args.phase=='engineering') for i,s in enumerate(seeds)]))
    if args.phase=='run':
        with (ROOT/'trials.jsonl').open('x') as f:
            for s in seeds:f.write((ROOT/'seed-shards'/f'{s}.jsonl').read_text())


if __name__=='__main__':main()
