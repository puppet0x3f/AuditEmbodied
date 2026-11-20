"""Render real frames by replaying unchanged actions of already observed paths."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import imageio.v3 as iio

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'keyframes'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('lift_graph','lift_heavy','handover'),required=True);args=parser.parse_args()
    hand=args.case=='handover';heavy=args.case=='lift_heavy'
    study=ROOT/'outputs'/('handover-audit-20261109' if hand else 'p1-load-transfer-20261109' if heavy else 'agentchord-style-20261109')
    sys.path.insert(0,str(study/'frozen-source'))
    protocol=json.loads((study/'protocol.json').read_text())
    match=(40001,'object_rotation','rule','1000ms') if hand else (50000,'left_loss','global','actual') if heavy else (12006,'right_loss','agentchord_style',None)
    with (study/'trials.jsonl').open() as f:
        row=next(r for r in map(json.loads,f) if (r['seed'],r['scenario'],r['method'],r.get('window'))==match and (not heavy or r['load']=='density150'))
    trace=row['trace'];prefix=max(i for i,f in enumerate(trace) if f['time']<=row['prefix_state']['time']+1e-8)
    if hand:
        call=next(c for c in row['skill_calls'] if c['skill']=='R1_retarget')
        local=next(i for i,f in enumerate(trace) if abs(f['time']-call['end_time'])<1e-8)
        lost=next(i for i,f in enumerate(trace) if f['phase']=='release' and not f['evaluator']['receiver_supports'])
        picks=[prefix,local,lost,len(trace)-1]
        labels=['Fault prefix','Local catch complete','Receiver support lost','Goal complete; contract B']
    elif heavy:
        picks=[prefix,max(i for i,f in enumerate(trace) if f['stage']=='global_table_settle'),
               max(i for i,f in enumerate(trace) if f['stage']=='global_close'),len(trace)-1]
        labels=['Unilateral fault','Verified table setdown','Fixed pickup misses handles','No dual regrasp; outcome D']
    else:
        lost=next(i for i,f in enumerate(trace) if f['phase']=='recovery' and f.get('skill')=='R_right' and not f['evaluator']['holds'][0])
        picks=[prefix,lost,max(i for i,f in enumerate(trace) if f['stage']=='global_table_settle'),len(trace)-1]
        labels=['Valid local start','Keeper grasp lost','Later global setdown','Goal restored; contract B']
    if hand:
        from auditembodied.l0 import L0Config,TwoArmL0Env,OracleVerifier
        from auditembodied.l3_matrix import _table_contact
        from auditembodied.continuous import _proprio
        from auditembodied.recovery_skills import actual_eef_rotations,actual_gripper_positions
        env=TwoArmL0Env(row['seed'],L0Config(render=True,horizon=2000,expert_mode='continuous'))
        def snapshot():return {**{k:np.asarray(getattr(env.sim.data,k)).tolist() for k in ('qpos','qvel','ctrl','act')},'time':float(env.sim.data.time)}
        oracle=OracleVerifier()
        def evaluate():
            return {**oracle.evaluate(env),'table_contact':_table_contact(env),
                'object_position':np.asarray(env.sim.data.body_xpos[env.raw.hammer_body_id]).tolist(),
                'object_quaternion':np.asarray(env.sim.data.body_xquat[env.raw.hammer_body_id]).tolist()}
    else:
        from auditembodied import l4_lift as runner
        from auditembodied.recovery_skills import actual_eef_rotations,actual_gripper_positions
        runner._import_robosuite(False)
        from robosuite.environments.manipulation import two_arm_lift as suite
        original=suite.PotWithHandlesObject
        def scaled(*a,**k):k['density']=1500.;return original(*a,**k)
        scaling=patch.object(suite,'PotWithHandlesObject',scaled if heavy else original)
        scaling.start()
        env=runner.TwoArmLiftEnv(row['seed'],runner.LiftConfig(**protocol['test_config']))
        env.raw.has_offscreen_renderer=True
        snapshot=lambda:runner.snapshot(env)
        evaluate=lambda:runner.evaluate(env)
    maximum=0.;metadata=[]
    def compare(a,b):
        nonlocal maximum
        if isinstance(b,dict):
            for k,v in b.items():compare(a[k],v)
        elif isinstance(b,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):compare(x,y)
        elif isinstance(b,(int,float,bool)):
            error=abs(float(a)-float(b));maximum=max(maximum,error);assert error<1e-8,(a,b,error)
        else:assert a==b
    try:
        OUT.mkdir(exist_ok=True);env.reset();compare(snapshot(),row['initialization_state'])
        if hand:env.bootstrap_handover();compare(snapshot(),row['initialization_end_state'])
        else:env.bootstrap();compare(snapshot(),row['preclose_state'])
        def forbidden(*a,**k):raise AssertionError('No state restoration after initialization')
        with ExitStack() as guards:
            names=('reset','bootstrap_handover','_set_hammer_pose','_move_arm_to_pose','_sync_after_reset','_sync_after_kinematic_edit') if hand else ('reset','bootstrap','_move_arm_to_pose','_sync_after_reset','_sync_after_kinematic_edit')
            for name in names:guards.enter_context(patch.object(env,name,side_effect=forbidden))
            wrench_index=0
            for i,f in enumerate(trace):
                if hand:
                    wrench=row.get('physical_wrench',{})
                    scale=(float(np.sin(np.pi*(wrench_index+.5)/wrench['control_steps']))
                        if wrench.get('profile')=='half_sine' else 1.)
                    env.sim.data.xfrc_applied[env.raw.hammer_body_id]=np.r_[wrench.get('world_force_n',[0,0,0]),wrench.get('world_torque_nm',[0,0,0])]*scale if f['stage']=='fault_wrench' else np.zeros(6)
                    if f['stage']=='fault_wrench':wrench_index+=1
                env.step(f['robot']['action14' if hand else 'action'])
                truth=evaluate();compare(truth,f['evaluator']);compare(float(env.sim.data.time),f['time'])
                robot=f['robot']
                if hand:
                    compare(_proprio(env).tolist(),robot['proprio20'])
                    compare(actual_gripper_positions(env).reshape(-1).tolist(),robot['actual_finger_qpos4'])
                    compare(actual_eef_rotations(env).tolist(),robot['actual_eef_rotations2x3x3'])
                else:
                    compare(np.stack([env._eef_pos(j) for j in range(2)]).tolist(),robot['positions'])
                    compare(actual_gripper_positions(env).tolist(),robot['fingers'])
                    compare(actual_eef_rotations(env).tolist(),robot['rotations'])
                if i in picks:
                    from robosuite import macros
                    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
                    convention=IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
                    before=snapshot()
                    for camera in ('frontview','agentview'):
                        rgb=env.sim.render(camera_name=camera,width=512,height=512)[::convention].copy()
                        path=OUT/f'{args.case}-{picks.index(i)}-{camera}.png';iio.imwrite(path,rgb)
                    compare(snapshot(),before)
                    metadata.append({'index':picks.index(i),'trace_index':i,'time':f['time'],
                        'post_fault_seconds':f['time']-row['prefix_state']['time'],'stage':f['stage'],
                        'caption':labels[picks.index(i)],'evaluator':truth})
        result={'case':args.case,'seed':row['seed'],'scenario':row['scenario'],'method':row['method'],
            'original_trace_frames':len(trace),'maximum_original_observation_error':maximum,
            'all_original_observations_match':True,'frames':metadata,
            'purpose':'Known executed trajectory illustration, not new policy evaluation. Original actions and physical parameters retained. CPU OSMesa images; fixed native cameras; no state restoration.'}
        (OUT/f'{args.case}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    finally:
        env.close()
        if not hand:scaling.stop()


if __name__=='__main__':main()
