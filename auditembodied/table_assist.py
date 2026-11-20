"""Sensor-guided table support and bounded regrasp after local search failure."""
from __future__ import annotations

from dataclasses import asdict,dataclass
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .recovery_skills import actual_eef_rotations,actual_eef_wrenches,actual_gripper_positions


@dataclass(frozen=True)
class TableAssistConfig:
    rgbd_template:str|None=None
    retreat_steps:int=24
    wrist_restore_steps:int=12
    settle_steps:int=8
    lower_steps:int=200
    lower_increment:float=.002
    table_height_tolerance:float=.015
    unload_fraction:float=.35
    stable_steps:int=4
    maximum_load_ratio:float=2.5
    contact_outward_increment:float=.002
    maximum_outward_shift:float=.20
    search_yaws_deg:tuple[float,...]=(0.,-12.,12.)
    hover_height:float=.06
    approach_steps:int=16
    close_steps:int=22
    probe_height:float=.004
    probe_steps:int=8
    load_share_fraction:float=.05
    tracking_tolerance:float=.012
    support_probe_distance:float=.0015
    support_probe_steps:int=4
    geometry_prepare_steps:int=220


def run_table_assist(env,phase,charge_compute,*,keeper,entry_positions,position_targets,
                     table_targets,registered_rotations,reference_wrench,lift_height,
                     lift_steps,record,config=None):
    """Use robot encoders, wrist sensors and registered robot goals only.

    No evaluator, contact array, object state or simulation state restoration
    enters this controller. Optional RGB-D registration uses rendered sensors.
    """
    cfg=config or TableAssistConfig();repair=1-keeper
    targets=np.asarray(position_targets).copy();table=np.asarray(table_targets)
    tare=np.asarray(reference_wrench);park=np.asarray(entry_positions).copy()
    work_started=time.perf_counter()
    record.update(parameters=asdict(cfg),keeper=keeper,repair=repair,
        reference_wrench=tare.tolist(),telemetry=[],attempts=[])

    def observe():
        return {'time':float(env.sim.data.time),
            'positions':np.stack([env._eef_pos(i) for i in range(2)]),
            'rotations':actual_eef_rotations(env),'fingers':actual_gripper_positions(env),
            'wrench':actual_eef_wrenches(env)}

    obs=observe()
    def command(name,p,grips):
        nonlocal targets,work_started,obs
        targets=np.asarray(p).copy()
        charge_compute(time.perf_counter()-work_started)
        phase(name,targets,1,grips)
        work_started=time.perf_counter();obs=observe()
        record['telemetry'].append({'stage':name,**{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in obs.items()},
            'position_targets':targets.tolist(),'rotation_targets':env.rotation_targets.tolist(),'grippers':list(grips)})

    def move(name,goal,steps,grips,end_rotations=None):
        start=targets.copy();r0=env.rotation_targets.copy()
        vectors=None if end_rotations is None else [Rotation.from_matrix(end_rotations[i]@r0[i].T).as_rotvec() for i in range(2)]
        for j in range(steps):
            fraction=j/(steps-1) if steps>1 else 1.
            f=fraction**3*(10+fraction*(-15+6*fraction))
            if vectors is not None:env.rotation_targets=np.stack([Rotation.from_rotvec(v*f).as_matrix()@r for v,r in zip(vectors,r0)])
            command(name,start+(np.asarray(goal)-start)*f,grips)

    def hold(name,steps,grips):
        for _ in range(steps):command(name,targets,grips)

    def loaded(i):
        width=float(np.mean(np.abs(obs['fingers'][i])))
        return .004<width<.028

    def load(i):return abs(float(obs['wrench'][i,2]-tare[i,2]))

    def done(completed,reason):
        charge_compute(time.perf_counter()-work_started)
        record.update(completed=completed,reason=reason,ended_sim_time=float(env.sim.data.time))
        return completed

    one=[1.,1.];one[repair]=-1.
    retreat=targets.copy();retreat[repair]=park[repair]
    move('assist_retreat',retreat,cfg.retreat_steps,one)
    restored=env.rotation_targets.copy();restored[repair]=registered_rotations[repair]
    move('assist_free_wrist_restore',targets.copy(),cfg.wrist_restore_steps,one,restored)
    hold('assist_carrier_settle',cfg.settle_steps,one)
    # Initial table contact can preload a closed wrist, so it is not an empty
    # force tare. Both robots use the same gripper; the now parked, open hand
    # supplies its measured static gravity baseline in world z.
    empty_z=float(np.median([t['wrench'][repair][2] for t in record['telemetry'][-4:]]))
    tare=tare.copy();tare[:,2]=empty_z
    record['world_z_empty_force']=empty_z
    initial_load=max(float(np.median([abs(t['wrench'][keeper][2]-tare[keeper,2]) for t in record['telemetry'][-4:]])),.2)
    record['initial_carrier_load']=initial_load
    initial_height=max(float(obs['positions'][keeper,2]-table[keeper,2]),.02)
    direction=1. if obs['wrench'][keeper,2]>=empty_z else -1.
    outward=table[keeper,:2]-table[repair,:2]
    outward=outward/np.linalg.norm(outward)
    contact_origin=None
    stable=0;contact_response=False;registration=None
    def visual_pose():
        from .table_rgbd import register
        prior=next((p for p in reversed(record.get('registrations',[])) if p.get('valid')),None)
        result=register(env,cfg.rgbd_template,keeper,obs['positions'],obs['rotations'],prior=prior)
        record.setdefault('registrations',[]).append(result)
        return result
    for _ in range(cfg.lower_steps):
        # Follow measured orientation, retaining the frozen inner-loop gains.
        # Small vertical goals allow the pot to pivot instead of forcing the
        # old pair of registered poses through first table contact.
        env.rotation_targets[keeper]=obs['rotations'][keeper]
        signed_load=direction*float(obs['wrench'][keeper,2]-empty_z)
        if not contact_response and signed_load<.7*initial_load:
            contact_response=True;contact_origin=obs['positions'][keeper,:2].copy()
            record['contact_response_sim_time']=float(env.sim.data.time)
        goal=targets.copy()
        if contact_response:
            # Latch the measured load response. In contact, stop accumulating
            # positional penetration; allow horizontal pivot motion and back
            # off compression while retaining the keeper's closed fingers.
            compression=signed_load<-.2*initial_load
            taper=np.clip((obs['positions'][keeper,2]-table[keeper,2])/initial_height,.25,1.)
            increment=(cfg.contact_outward_increment if compression else cfg.contact_outward_increment/2)*taper
            goal[keeper,:2]+=outward*increment
            displacement=float((goal[keeper,:2]-contact_origin)@outward)
            if displacement>cfg.maximum_outward_shift:
                return done(False,'bounded_support_pivot_exhausted')
            z=goal[keeper,2]+(.001 if compression else -cfg.lower_increment)
            goal[keeper,2]=np.clip(z,max(table[keeper,2],obs['positions'][keeper,2]-.010),obs['positions'][keeper,2]+.010)
        else:goal[keeper,2]=max(table[keeper,2],goal[keeper,2]-cfg.lower_increment)
        command('assist_guarded_lower',goal,one)
        if not loaded(keeper):return done(False,'carrier_encoder_lost_before_support')
        if load(keeper)>cfg.maximum_load_ratio*initial_load+1.:
            return done(False,'carrier_load_limit')
        if cfg.rgbd_template:
            stable=stable+1 if contact_response and load(keeper)<=.8*initial_load else 0
            if stable>=cfg.stable_steps:
                # Freeze the accumulated goals before probing partial support.
                targets=obs['positions'].copy();env.rotation_targets=obs['rotations'].copy()
                hold('assist_support_settle',cfg.settle_steps,one)
                registration=visual_pose()
                hold('assist_support_observe',1,one)
                if not registration['valid'] or registration['table_gap_m']>.035:
                    return done(False,'support_geometry_not_confirmed')
                before=direction*float(obs['wrench'][keeper,2]-empty_z)
                origin=targets.copy();probe=origin.copy();probe[keeper,2]-=cfg.support_probe_distance
                offset=len(record['telemetry'])
                move('assist_support_probe',probe,cfg.support_probe_steps,one)
                values=[direction*(t['wrench'][keeper][2]-empty_z) for t in record['telemetry'][offset:]]
                move('assist_support_probe_return',origin,cfg.support_probe_steps,one)
                reaction=before-min(values)
                record['support_probe']={'before_load':before,'minimum_load':min(values),'reaction_N':reaction,
                    'keeper_loaded':loaded(keeper),'table_gap_m':registration['table_gap_m']}
                if loaded(keeper) and reaction>=max(.1,.015*initial_load):break
                return done(False,'active_support_probe_not_confirmed')
        else:
            near_table=abs(float(obs['positions'][keeper,2]-table[keeper,2]))<=cfg.table_height_tolerance
            stable=stable+1 if near_table and load(keeper)<=cfg.unload_fraction*initial_load else 0
            if stable>=cfg.stable_steps:break
    else:return done(False,'support_transfer_not_confirmed')
    record['support_detected_sim_time']=float(env.sim.data.time)
    hold('assist_table_loaded',cfg.settle_steps,one)

    if cfg.rgbd_template:
        # Contact can be valid with the free handle almost touching the table.
        # Prepare a graspable geometry separately, while the carrier stays shut.
        for j in range(cfg.geometry_prepare_steps):
            if j%20==0:
                registration=visual_pose();hold('assist_geometry_observe',1,one)
                if registration['valid'] and registration['tilt_deg']<=12.:break
            env.rotation_targets[keeper]=obs['rotations'][keeper]
            signed=direction*float(obs['wrench'][keeper,2]-empty_z)
            compression=signed<-.2*initial_load
            goal=targets.copy()
            taper=np.clip((obs['positions'][keeper,2]-table[keeper,2])/initial_height,.25,1.)
            goal[keeper,:2]+=outward*(.002 if compression else .001)*taper
            goal[keeper,2]=np.clip(goal[keeper,2]+(.001 if compression else -.002),
                max(table[keeper,2],obs['positions'][keeper,2]-.010),obs['positions'][keeper,2]+.010)
            command('assist_geometry_prepare',goal,one)
            if not loaded(keeper):return done(False,'carrier_encoder_lost_during_geometry')
            if load(keeper)>cfg.maximum_load_ratio*initial_load+1.:return done(False,'carrier_load_limit')
        else:return done(False,'grasp_geometry_not_ready')

    span=table[repair]-table[keeper];span[2]=0.
    for yaw in cfg.search_yaws_deg:
        # Keep the carrier closed. Search only the free hand around a robot
        # anchor; the registered object/handle truth is never re-read.
        rz=Rotation.from_rotvec([0.,0.,np.radians(yaw)]).as_matrix()
        if cfg.rgbd_template:
            registration=visual_pose();hold('assist_grasp_observe',1,one)
            if not registration['valid']:return done(False,'regrasp_geometry_not_confirmed')
            grasp=np.asarray(registration['positions'])[repair].copy()
            # Bounded yaw search is centred on the observed handle, never the
            # old keeper TCP height. Rotate both approach and wrist together.
            wrist=rz@np.asarray(registration['rotations'])[repair]
        else:
            anchor=obs['positions'][keeper].copy()
            grasp=anchor+rz@span
            grasp[2]=table[repair,2]+anchor[2]-table[keeper,2]
            wrist=rz@registered_rotations[repair]
        hover=targets.copy();hover[repair]=grasp+[0.,0.,cfg.hover_height]
        desired=env.rotation_targets.copy();desired[repair]=wrist
        move('assist_search_hover',hover,cfg.approach_steps,one,desired)
        grasp_goal=targets.copy();grasp_goal[repair]=grasp
        move('assist_search_descend',grasp_goal,cfg.approach_steps,one)
        hold('assist_search_close',cfg.close_steps,(1.,1.))
        error=float(np.linalg.norm(targets[repair]-obs['positions'][repair]))
        attempt={'yaw_deg':yaw,'width':float(np.mean(np.abs(obs['fingers'][repair]))),'tracking_error':error}
        record['attempts'].append(attempt)
        if loaded(repair) and error<=cfg.tracking_tolerance:
            probe=obs['positions'].copy();probe[:,2]+=cfg.probe_height
            move('assist_probe_lift',probe,cfg.probe_steps,(1.,1.),obs['rotations'].copy())
            hold('assist_probe_settle',4,(1.,1.))
            share=load(repair)
            attempt['probe_load']=share
            attempt['probe_loaded']=loaded(repair) and share>=max(.1,cfg.load_share_fraction*initial_load)
            if attempt['probe_loaded']:
                goal=obs['positions'].copy();goal[:,2]+=lift_height
                move('assist_joint_lift',goal,lift_steps,(1.,1.),obs['rotations'].copy())
                if cfg.rgbd_template:
                    # Use observations AFTER lifting; a pre-lift pose is stale
                    # after 2.5 seconds of physical load transfer and motion.
                    for correction_index in range(2):
                        verified=visual_pose();hold('assist_postlift_observe',1,(1.,1.))
                        if not verified['valid']:return done(False,'postlift_geometry_not_confirmed')
                        predicted=np.asarray(verified['positions'])
                        if abs(predicted[0,2]-predicted[1,2])<=.015 and verified['tilt_deg']<=5.:break
                        r=np.asarray(verified['rotation']);z=r[:,2]
                        axis=np.cross(z,np.array([0.,0.,1.]));sine=np.linalg.norm(axis)
                        if sine<1e-5:break
                        correction=Rotation.from_rotvec(axis/sine*np.arctan2(sine,z[2])).as_matrix()
                        start=obs['positions'].copy();center=start.mean(0);r0=obs['rotations'].copy()
                        vector=Rotation.from_matrix(correction).as_rotvec()
                        for j in range(30):
                            f=(j+1)/30;f=f**3*(10+f*(-15+6*f))
                            delta=Rotation.from_rotvec(vector*f).as_matrix()
                            env.rotation_targets=np.einsum('ij,njk->nik',delta,r0)
                            command('assist_joint_level',(start-center)@delta.T+center,(1.,1.))
                        hold('assist_level_settle',8,(1.,1.))
                hold('assist_joint_settle',12,(1.,1.))
                return done(True,'joint_lift_executed')
            back=obs['positions'].copy();back[:,2]-=cfg.probe_height
            move('assist_probe_setdown',back,cfg.probe_steps,(1.,1.))
        clear=targets.copy();clear[repair]=obs['positions'][repair]+[0.,0.,cfg.hover_height]
        move('assist_search_clear',clear,cfg.approach_steps,one)
    return done(False,'bounded_regrasp_exhausted')
