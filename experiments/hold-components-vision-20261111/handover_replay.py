"""Exact-action native replay of two handover role cases and paired controls."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import json
import multiprocessing
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'outputs/handover-audit-20261109'
sys.path.insert(0,str(OLD/'frozen-source'))
from auditembodied import l3_matrix as runner
from auditembodied.l0 import TwoArmL0Env,L0Config,OracleVerifier
from auditembodied.continuous import ContinuousActionExpert
from auditembodied.recovery_skills import actual_gripper_positions,actual_eef_rotations


def longest(flags):
    run=maximum=0
    for v in flags:run=run+1 if v else 0;maximum=max(maximum,run)
    return maximum


def one(row):
    name=f"{row['seed']}-{row['scenario']}-{row['method']}-{row['window']}"
    folder=HERE/'handover-native';folder.mkdir(exist_ok=True)
    assert not (folder/(name+'.json')).exists()
    maximum=0.;records=[];context={};oracle=OracleVerifier()
    def compare(current,original):
        nonlocal maximum
        if isinstance(original,dict):
            for k,v in original.items():compare(current[k],v)
        elif isinstance(original,(tuple,list)):
            assert len(current)==len(original)
            for a,b in zip(current,original):compare(a,b)
        elif isinstance(original,(int,float,bool)):
            error=abs(float(current)-float(original));maximum=max(maximum,error)
            assert error<1e-8,(current,original,error)
        else:assert current==original
    env=TwoArmL0Env(row['seed'],L0Config(render=False,horizon=2000,expert_mode='continuous'))
    try:
        env.reset();compare(runner.state(env) if hasattr(runner,'state') else {
            **{k:np.asarray(getattr(env.sim.data,k)).tolist() for k in ('qpos','qvel','ctrl','act')},'time':float(env.sim.data.time)},row['initialization_state'])
        env.bootstrap_handover()
        compare({**{k:np.asarray(getattr(env.sim.data,k)).tolist() for k in ('qpos','qvel','ctrl','act')},'time':float(env.sim.data.time)},row['initialization_end_state'])
        step=env.sim.step
        def native_step(*args,**kwargs):
            answer=step(*args,**kwargs)
            if context.get('phase') not in ('recovery','release','hold'):return answer
            truth=oracle.evaluate(env);table=runner._table_contact(env)
            penetration=0.
            for c in env.sim.data.contact[:env.sim.data.ncon]:
                a,b=(env.sim.model.geom_id2name(int(i)) for i in (c.geom1,c.geom2))
                if a.startswith('robot0') and b.startswith('robot1') or a.startswith('robot1') and b.startswith('robot0'):
                    penetration=max(penetration,-float(c.dist))
            role=context.get('audit',{}).get('role_obligation')
            records.append((float(env.sim.data.time),context['stage'],context['phase'],truth['giver_holds'],
                truth['receiver_holds'],truth['receiver_supports'],table,role or '',
                bool(role and not truth[role]),penetration))
            return answer
        env.sim.step=native_step
        def forbidden(*a,**k):raise AssertionError('No state edit or new control in exact-action replay')
        wrench_index=0;wrench=row.get('physical_wrench')
        with ExitStack() as guards:
            for key in ('reset','bootstrap_handover','_move_arm_to_pose','_move_arm_to_position','_set_hammer_pose',
                        '_sync_after_reset','_sync_after_kinematic_edit','sync_gripper_controller'):
                guards.enter_context(patch.object(env,key,side_effect=forbidden))
            guards.enter_context(patch.object(ContinuousActionExpert,'_set_contact_state',side_effect=forbidden))
            for frame in row['trace']:
                context=frame
                if frame['stage']=='fault_wrench':
                    amplitude=np.r_[wrench['world_force_n'],wrench['world_torque_nm']]
                    scale=float(np.sin(np.pi*(wrench_index+.5)/wrench['control_steps'])) if wrench['profile']=='half_sine' else 1.
                    env.sim.data.xfrc_applied[env.raw.hammer_body_id]=amplitude*scale;wrench_index+=1
                else:env.sim.data.xfrc_applied[env.raw.hammer_body_id]=0.
                env.step(frame['robot']['action14'])
                truth={**oracle.evaluate(env),'table_contact':runner._table_contact(env),
                    'object_position':np.asarray(env.sim.data.body_xpos[env.raw.hammer_body_id]).tolist(),
                    'object_quaternion':np.asarray(env.sim.data.body_xquat[env.raw.hammer_body_id]).tolist()}
                compare(float(env.sim.data.time),frame['time']);compare(truth,frame['evaluator'])
                compare(runner._proprio(env).tolist(),frame['robot']['proprio20'])
                compare(actual_eef_rotations(env).tolist(),frame['robot']['actual_eef_rotations2x3x3'])
                compare(actual_gripper_positions(env).reshape(-1).tolist(),frame['robot']['actual_finger_qpos4'])
        names=('time','stage','phase','giver_holds','receiver_holds','receiver_supports','table','role','role_bad','penetration')
        kinds=(float,str,str,bool,bool,bool,bool,str,bool,float)
        arrays={k:np.asarray([r[i] for r in records],dtype=d) for i,(k,d) in enumerate(zip(names,kinds))}
        frames=[f for f in row['trace'] if f['phase'] in ('recovery','release','hold')]
        assert len(records)==25*len(frames)
        indices=np.arange(24,len(records),25)
        assert np.array_equal(arrays['role_bad'][indices],[not f['audit']['role_obligation_satisfied'] for f in frames])
        missing=~(arrays['giver_holds']|arrays['receiver_holds']|arrays['table'])
        report={'name':name,'seed':row['seed'],'method':row['method'],'window':row['window'],
            'diagnostic_only':True,'independent_policy_trials':0,'original_label':row['audit']['label'],
            'all_original_frames_match':True,'original_frames':len(row['trace']),
            'maximum_original_observation_error':maximum,'native_samples':len(records),
            'native_registered_gap_ms':longest(missing)*2,
            'native_role_loss_samples':int(arrays['role_bad'].sum()),
            'native_longest_role_gap_ms':longest(arrays['role_bad'])*2,
            'sampled_role_loss_samples':sum(not f['audit']['role_obligation_satisfied'] for f in frames),
            'native_shared_bad':bool(longest(missing)>50 or np.any(arrays['penetration']>.01))}
        np.savez_compressed(folder/(name+'.npz'),**arrays)
        (folder/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
        return report
    finally:env.close()


if __name__=='__main__':
    rows=[]
    for line in (OLD/'trials.jsonl').open():
        r=json.loads(line)
        if r['scenario']=='object_rotation' and (
            r['seed']==40001 and r['method'] in ('rule','proprio') or
            r['seed']==40017 and r['method']=='proprio' and r['window']=='2000ms'):
            rows.append(r)
    assert len(rows)==7
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        reports=list(pool.map(one,rows))
    (HERE/'handover-replay-report.json').write_text(json.dumps({'cases':7,'independent_policy_trials':0,
        'all_original_frames_match':True,'maximum_error':max(r['maximum_original_observation_error'] for r in reports),
        'reports':reports},indent=2)+'\n')
