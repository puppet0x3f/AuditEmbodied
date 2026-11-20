"""Record native physics ticks while replaying one frozen, already observed rollout."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT=Path('/home/lc/AuditEmbodied')
OUT=Path(__file__).resolve().parent
SOURCES={'scope':ROOT/'outputs/agentchord-style-20261109',
         'supervision':ROOT/'outputs/supervision-controls-20261109'}


def main():
    parser=argparse.ArgumentParser()
    for name in ('cohort','scenario','method'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--seed',type=int,required=True);parser.add_argument('--contacts',action='store_true');args=parser.parse_args()
    base=SOURCES[args.cohort];frozen=base/'frozen-source'
    sys.path.insert(0,str(frozen))
    from auditembodied.l4_lift import TwoArmLiftEnv,LiftConfig,evaluate,snapshot
    from auditembodied.recovery_skills import actual_eef_rotations,actual_gripper_positions
    import mujoco
    source=base/('trials.jsonl' if args.cohort=='scope' else 'closed-loop/trials.jsonl')
    with source.open() as stream:
        row=next(json.loads(line) for line in stream if (lambda r:(r['seed'],r['scenario'],r['method'])==(args.seed,args.scenario,args.method))(json.loads(line)))
    config=json.loads((base/'protocol.json').read_text())['test_config'] if args.cohort=='scope' else {}
    case=f'{args.cohort}-{args.seed}-{args.scenario}-{args.method}'
    folder=OUT/('contact-diagnostic' if args.contacts else 'frequency');folder.mkdir(exist_ok=True)
    env=TwoArmLiftEnv(args.seed,LiftConfig(**config));started=time.perf_counter()
    report={'case':case,'diagnostic_only':True,'source':str(source),'source_package':str(frozen),
            'original_safe':bool(row['success'] and not row['protection_violation']),
            'accepted_completion':bool(row['success']),'original_protection':row['protection'],
            'low_frequency_frames':len(row['trace']),'all_20hz_frames_match':False,'error':None}
    recorded=[];max_error=0.;current=None;control_index=-1;tick=0

    def compare(actual,expected):
        nonlocal max_error
        if isinstance(expected,dict):
            for key,value in expected.items():compare(actual[key],value)
        elif isinstance(expected,(list,tuple)):
            assert len(actual)==len(expected)
            for a,b in zip(actual,expected):compare(a,b)
        elif isinstance(expected,(int,float,bool)):
            error=abs(float(actual)-float(expected));max_error=max(max_error,error)
            assert error<1e-8,(actual,expected,error)
        else:assert actual==expected

    def forbidden(*args,**kwargs):raise AssertionError('Initialization-only method called during replay')

    try:
        env.reset();compare(snapshot(env),row['initialization_state'])
        env.bootstrap();compare(snapshot(env),row['preclose_state'])
        dt=float(env.sim.model.opt.timestep);assert abs(dt-.002)<1e-12
        report['physics_timestep_seconds']=dt;report['native_recording_hz']=1/dt
        original_step=env.sim.step

        def sampled_step(*step_args,**step_kwargs):
            nonlocal tick
            result=original_step(*step_args,**step_kwargs);tick+=1
            if current['phase'] in ('recovery','final_hold'):
                truth=evaluate(env)
                skill=current.get('skill',row['candidate'])
                keeper=(1 if skill=='R_left' else 0 if skill=='R_right' else -1) if current['phase']=='recovery' else -1
                penetration=0.;pot_force=0.;pot_contacts=0
                pot_geoms=set(env.raw.pot.contact_geoms)
                for contact_index,contact in enumerate(env.sim.data.contact[:env.sim.data.ncon]):
                    a,b=(env.sim.model.geom_id2name(int(g)) for g in (contact.geom1,contact.geom2))
                    if ((a.startswith('robot0') and b.startswith('robot1')) or (a.startswith('robot1') and b.startswith('robot0'))):
                        penetration=max(penetration,-float(contact.dist))
                    if args.contacts and (a in pot_geoms or b in pot_geoms):
                        wrench=np.zeros(6)
                        mujoco.mj_contactForce(env.sim.model._model,env.sim.data._data,contact_index,wrench)
                        magnitude=float(np.linalg.norm(wrench[:3]));pot_force+=magnitude
                        pot_contacts+=int(magnitude>1e-6)
                recorded.append({'time':float(env.sim.data.time),'control_index':control_index,'tick':tick,
                    'phase':current['phase'],'stage':current['stage'],'keeper':keeper,
                    'holds':truth['holds'],'table_contact':truth['table_contact'],
                    'pot_center':truth['pot_center'],'tilt_deg':truth['tilt_deg'],
                    'clearance':truth['clearance'],'interarm_penetration_m':penetration})
                if args.contacts:recorded[-1].update(active_pot_contacts=pot_contacts,pot_contact_force_norm_sum=pot_force)
            return result

        with ExitStack() as guards:
            for name in ('reset','bootstrap','_move_arm_to_pose','_sync_after_reset','_sync_after_kinematic_edit','sync_gripper_controller'):
                guards.enter_context(patch.object(env,name,side_effect=forbidden))
            guards.enter_context(patch.object(env.raw,'reset',side_effect=forbidden))
            guards.enter_context(patch.object(env.sim,'step',side_effect=sampled_step))
            for control_index,current in enumerate(row['trace']):
                tick=0;env.step(current['robot']['action']);assert tick==25,tick
                compare(float(env.sim.data.time),current['time'])
                compare(evaluate(env),current['evaluator'])
                compare(np.stack([env._eef_pos(i) for i in range(2)]).tolist(),current['robot']['positions'])
                compare(actual_eef_rotations(env).tolist(),current['robot']['rotations'])
                compare(actual_gripper_positions(env).tolist(),current['robot']['fingers'])
                if abs(current['time']-row['prefix_state']['time'])<1e-8:compare(snapshot(env),row['prefix_state'])
        assert len(recorded)==25*sum(f['phase'] in ('recovery','final_hold') for f in row['trace'])
        report.update(all_20hz_frames_match=True,maximum_original_observation_error=max_error,
                      high_frequency_samples=len(recorded),initialization_and_fault_states_match=True,
                      state_restorations_after_initialization=0,external_wrench_at_end=float(np.linalg.norm(env.sim.data.xfrc_applied)))
        arrays={key:np.asarray([r[key] for r in recorded]) for key in recorded[0]}
        np.savez_compressed(folder/f'{case}.npz',**arrays)
    except Exception as error:
        report.update(error=repr(error),failed_control_index=control_index,maximum_original_observation_error=max_error)
        raise
    finally:
        env.close();report['wall_seconds']=time.perf_counter()-started
        (folder/f'{case}.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
