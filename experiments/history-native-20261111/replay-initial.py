"""Replay exactly the ten published history-only B paths at native frequency."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import json,multiprocessing,sys,time
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
SOURCE=REPO/'outputs/interface-retarget-20261110'
sys.path.insert(0,str(SOURCE/'frozen-source'))
from auditembodied import l4_lift as runner
from auditembodied.recovery_skills import actual_eef_rotations,actual_gripper_positions
from auditembodied.continuous import ContinuousActionExpert
from sampling import NativeRecorder

def one(case):
    started=time.perf_counter();name=case['name']
    row=json.loads((ROOT/'source-cases'/f'{name}.json').read_text())
    cfg=runner.LiftConfig(**json.loads((SOURCE/'protocol.json').read_text())['test_config'])
    runner._import_robosuite(False)
    from robosuite.environments.manipulation import two_arm_lift as suite
    original=suite.PotWithHandlesObject
    def scaled(*a,**k):k['density']=1500.;return original(*a,**k)
    maximum=0.;recorder=None;env=None
    report={**case,'diagnostic_only':True,'error':None,'original_frames':len(row['trace'])}
    def compare(x,y):
        nonlocal maximum
        if isinstance(y,dict):
            for k,v in y.items():compare(x[k],v)
        elif isinstance(y,(list,tuple)):
            assert len(x)==len(y)
            for a,b in zip(x,y):compare(a,b)
        elif isinstance(y,(int,float,bool)):
            d=abs(float(x)-float(y));maximum=max(maximum,d);assert d<1e-8,(x,y,d)
        else:assert x==y
    def forbidden(*a,**k):raise AssertionError('No state restoration or new control allowed')
    try:
        with patch.object(suite,'PotWithHandlesObject',scaled):env=runner.TwoArmLiftEnv(row['seed'],cfg)
        env.reset();compare(runner.snapshot(env),row['initialization_state'])
        recorder=NativeRecorder(env,runner.evaluate)
        env.bootstrap();compare(runner.snapshot(env),row['preclose_state'])
        with ExitStack() as guards:
            for k in ('reset','bootstrap','_move_arm_to_pose','_sync_after_reset','_sync_after_kinematic_edit','sync_gripper_controller'):
                guards.enter_context(patch.object(env,k,side_effect=forbidden))
            guards.enter_context(patch.object(env.raw,'reset',side_effect=forbidden))
            guards.enter_context(patch.object(ContinuousActionExpert,'_set_contact_state',side_effect=forbidden))
            for index,frame in enumerate(row['trace']):
                env.audit_context=(frame['phase'],frame['stage'],row['candidate'])
                env.step(frame['robot']['action'])
                compare(float(env.sim.data.time),frame['time'])
                compare(runner.evaluate(env),frame['evaluator'])
                compare(np.stack([env._eef_pos(i) for i in range(2)]).tolist(),frame['robot']['positions'])
                compare(actual_eef_rotations(env).tolist(),frame['robot']['rotations'])
                compare(actual_gripper_positions(env).tolist(),frame['robot']['fingers'])
                if abs(frame['time']-row['prefix_state']['time'])<1e-8:compare(runner.snapshot(env),row['prefix_state'])
        assert len(recorder.records)==25*len(row['trace'])
        report.update(all_original_frames_match=True,maximum_original_observation_error=maximum,
            original_wait_seconds=row['inference_wait_seconds'],original_total_seconds=row['total_post_fault_seconds'],
            external_wrench_at_end=float(np.linalg.norm(env.sim.data.xfrc_applied)),
            native=recorder.save(ROOT/'native'/f'{name}.npz'))
    except Exception as e:
        report.update(error=repr(e),failed_control_index=locals().get('index'),maximum_original_observation_error=maximum)
        raise
    finally:
        if env is not None:env.close()
        report['wall_seconds']=time.perf_counter()-started
        (ROOT/'native'/f'{name}.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:report.get(k) for k in ['name','error','all_original_frames_match','maximum_original_observation_error']}),flush=True)
    return report

if __name__=='__main__':
    (ROOT/'native').mkdir(exist_ok=False)
    cases=json.loads((ROOT/'cases.json').read_text())['cases']
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:reports=list(pool.map(one,cases))
    (ROOT/'replay-report.json').write_text(json.dumps({'cases':len(reports),'all_match':True,'independent_policy_trials':0,
        'maximum_error':max(r['maximum_original_observation_error'] for r in reports),'reports':reports},indent=2)+'\n')
