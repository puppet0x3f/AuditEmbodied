"""Read-only contact instrumentation of three previously observed heavy-load paths."""
from contextlib import ExitStack
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'outputs/p1-load-transfer-20261109'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(STUDY))
import campaign as original
from auditembodied.recovery_skills import actual_eef_rotations,actual_gripper_positions


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scenario',choices=('left_loss','right_loss','sham'),required=True)
    args=parser.parse_args();seed=50000
    protocol=json.loads((STUDY/'protocol.json').read_text());original.verify(protocol)
    with (STUDY/'seed-shards'/f'{seed}.jsonl').open() as f:
        row=next(r for r in map(json.loads,f) if (r['scenario'],r['load'],r['window'],r['method'])==(args.scenario,'density150','actual','global'))
    original.runner._import_robosuite(False)
    import mujoco
    from robosuite.environments.manipulation import two_arm_lift as suite
    factory=suite.PotWithHandlesObject
    maximum=0.;records=[];contacts=[]
    def compare(a,b):
        nonlocal maximum
        if isinstance(b,dict):
            for k,v in b.items():compare(a[k],v)
        elif isinstance(b,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):compare(x,y)
        elif isinstance(b,(float,int,bool)):
            error=abs(float(a)-float(b));maximum=max(maximum,error);assert error<1e-8,(a,b,error)
        else:assert a==b
    case=f'{seed}-{args.scenario}-density150-actual-global'
    folder=OUT/'contacts';folder.mkdir(exist_ok=True)
    assert not (folder/(case+'.json')).exists()
    with patch.object(suite,'PotWithHandlesObject',original.scaled_factory(factory,1.5)):
        env=original.runner.TwoArmLiftEnv(seed,original.runner.LiftConfig(**protocol['test_config']))
        try:
            env.reset();compare(original.runner.snapshot(env),row['initialization_state'])
            env.bootstrap();compare(original.runner.snapshot(env),row['preclose_state'])
            compare(original.physical_parameters(env),row['physical_parameters'])
            model=env.sim.model;data=env.sim.data
            pot=set(env.raw.pot.contact_geoms)
            finger_groups={}
            for arm in range(2):
                gripper=env.raw.robots[arm].gripper['right']
                for finger,name in enumerate(('left_finger','right_finger')):
                    for geom in gripper.important_geoms[name]:finger_groups[geom]=(arm,finger)
            actuator_ids=[int(i) for robot in env.raw.robots for i in robot._ref_joint_gripper_actuator_indexes['right']]
            def forbidden(*a,**k):raise AssertionError('No reset/state restoration after closing starts')
            with ExitStack() as guards:
                for name in ('reset','bootstrap','_move_arm_to_pose','_sync_after_reset','_sync_after_kinematic_edit','sync_gripper_controller'):
                    guards.enter_context(patch.object(env,name,side_effect=forbidden))
                guards.enter_context(patch.object(env.raw,'reset',side_effect=forbidden))
                for index,f in enumerate(row['trace']):
                    env.step(f['robot']['action'])
                    truth=original.runner.evaluate(env)
                    compare(truth,f['evaluator']);compare(float(data.time),f['time'])
                    compare(np.stack([env._eef_pos(i) for i in range(2)]).tolist(),f['robot']['positions'])
                    compare(actual_eef_rotations(env).tolist(),f['robot']['rotations'])
                    compare(actual_gripper_positions(env).tolist(),f['robot']['fingers'])
                    if abs(f['time']-row['prefix_state']['time'])<1e-8:compare(original.runner.snapshot(env),row['prefix_state'])
                    if f['phase'] not in ('recovery','final_hold'):continue
                    sums={'table':0.,'arm0_fingers':0.,'arm1_fingers':0.,'other_robot':0.,'other':0.}
                    finger_forces=np.zeros((2,2));handle_finger_contacts=np.zeros((2,2),dtype=int)
                    handles=[set(env.raw.pot.handle0_geoms),set(env.raw.pot.handle1_geoms)]
                    ncontacts=0
                    for ci,contact in enumerate(data.contact[:data.ncon]):
                        a,b=(model.geom_id2name(int(g)) for g in (contact.geom1,contact.geom2))
                        if a not in pot and b not in pot:continue
                        object_geom,other=(a,b) if a in pot else (b,a)
                        wrench=np.zeros(6);mujoco.mj_contactForce(model._model,data._data,ci,wrench)
                        magnitude=float(np.linalg.norm(wrench[:3]));ncontacts+=1
                        if other in finger_groups:
                            arm,finger=finger_groups[other];group=f'arm{arm}_fingers'
                            finger_forces[arm,finger]+=magnitude
                            handle_finger_contacts[arm,finger]+=int(object_geom in handles[arm])
                        elif other.startswith('table'):group='table'
                        elif other.startswith(('robot','gripper')):group='other_robot'
                        else:group='other'
                        sums[group]+=magnitude
                        contacts.append({'time':float(data.time),'stage':f['stage'],'object_geom':object_geom,
                            'other_geom':other,'normal_force_N':float(wrench[0]),'force_norm_N':magnitude,
                            'distance_m':float(contact.dist),'position':np.asarray(contact.pos).tolist()})
                    records.append({'time':float(data.time),'stage':f['stage'],'phase':f['phase'],
                        **truth,'positions':f['robot']['positions'],'rotations':f['robot']['rotations'],
                        'fingers':f['robot']['fingers'],'gripper_actuator_force_N':np.asarray(data.actuator_force)[actuator_ids].tolist(),
                        'gripper_ctrl':np.asarray(data.ctrl)[actuator_ids].tolist(),
                        'object_rotation':np.asarray(data.body_xmat[env.raw.pot_body_id]).reshape(3,3).tolist(),
                        'finger_contact_force_norm_N':finger_forces.tolist(),'handle_finger_contact_counts':handle_finger_contacts.tolist(),
                        'contact_force_norm_sums_N':sums,'object_contact_count':ncontacts})
            assert not np.any(data.xfrc_applied)
            result={'case':case,'seed':seed,'scenario':args.scenario,'source_load':'density150',
                'purpose':'Known-state contact diagnosis only; no controller, action, weight, threshold, parameter or state intervention.',
                'all_original_20hz_observations_match':True,'maximum_original_observation_error':maximum,
                'physical_samples':len(records),'object_contacts':len(contacts),'state_restorations':0,
                'original_label':row['audit']['label'],'original_seconds':row['total_post_fault_seconds'],
                'source_model_sha256':protocol['model_sha256'],
                'force_interpretation':'Contact force norms are magnitudes, not net upward support or grasp-capacity certificates.'}
            for filename,items in [(case+'-samples.jsonl',records),(case+'-contacts.jsonl',contacts)]:
                with (folder/filename).open('x') as f:
                    for x in items:f.write(json.dumps(x)+'\n')
            (folder/(case+'.json')).write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result,indent=2))
        finally:env.close()
    original.verify(protocol)


if __name__=='__main__':main()
