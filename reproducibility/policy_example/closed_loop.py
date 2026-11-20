"""Independent RGB Cartesian recovery with measured service and deferred leveling.

Only RGB fits, robot kinematics, encoders and registered references reach control.
The simulator's contact/pose audit is outside this module.
"""
import time
import numpy as np
from geometry import BODY,HANDLES,exp_rotation
from observation import reference


def rotation_vector(matrix):
    # Quaternion-free SO(3) logarithm for interpolating the registered wrists.
    angle=np.arccos(np.clip((np.trace(matrix)-1)/2,-1.,1.))
    if angle<1e-8:return np.zeros(3)
    return angle/(2*np.sin(angle))*np.array([matrix[2,1]-matrix[1,2],matrix[0,2]-matrix[2,0],matrix[1,0]-matrix[0,1]])


def recover(p):
    log={'observations':[],'pickup_attempts':[],'service_steps':0,
         'policy':'RGB Cartesian feedback; preserve carrier wrist during setdown; align and defer leveling'}
    table=np.asarray(p['table_targets']);nominal_rot=np.asarray(p['registered_rotations'])
    ref_center,ref_rot=reference(table,nominal_rot)
    table_z=ref_center[2]-.07
    local_grips=(table-ref_center)@ref_rot
    grips=tuple(p['gripper_commands'])
    p['set_rotations'](p['robot']()['rotations'])

    def robot():return np.asarray(p['robot']()['positions'])

    def observe(grippers):
        at=p['time']();started=time.perf_counter();d=p['observe']()
        seconds=time.perf_counter()-started;steps=max(1,int(np.ceil(seconds/.05)))
        p['phase']('regrasp_observation_wait',robot(),steps,grippers)
        log['service_steps']+=steps
        d.update(observed_time=at,service_wall_seconds=seconds,service_wait_steps=steps)
        log['observations'].append(d)
        return d['estimate']

    def loaded():
        widths=np.mean(np.abs(p['robot']()['fingers']),axis=1)
        return [bool(.004<w<.028) for w in widths]

    # Preserve the actual carrier orientation; lower the observed object onto
    # the registered plane before opening. A second measurement closes setdown.
    for _ in range(2):
        estimate=observe(grips)
        if not estimate['valid']:break
        center=np.asarray(estimate['center']);rot=np.asarray(estimate['rotation'])
        bottom=min((template@rot.T+center)[:,2].min() for template in (BODY,*HANDLES))
        lower=float(np.clip(bottom-table_z+.005,0.,.22))
        targets=robot();targets[:,2]-=lower
        p['move']('global_lower_to_table',targets,max(12,int(np.ceil(lower/.003))),grips)
        p['phase']('global_table_loaded',targets,10,grips)
    targets=robot()
    p['phase']('global_release',targets,14,(-1.,-1.))
    p['phase']('global_table_settle',targets,10,(-1.,-1.))
    targets=robot();targets[:,2]+=.12
    p['move']('global_clear',targets,24,(-1.,-1.))

    for attempt in range(2):
        entry={'index':attempt,'start_time':p['time']()};log['pickup_attempts'].append(entry)
        estimate=observe((-1.,-1.))
        if not estimate['valid']:
            entry['stop']='insufficient_rgb';break
        center=np.asarray(estimate['center']);rot=np.asarray(estimate['rotation'])
        targets=center+local_grips@rot.T
        desired=np.asarray(estimate['rotation_targets'])
        previous=np.asarray(p['robot']()['rotations']);hold=robot()
        turn=[rotation_vector(desired[i]@previous[i].T) for i in range(2)]
        for j in range(20):
            p['set_rotations']([exp_rotation(v*(j+1)/20)@previous[i] for i,v in enumerate(turn)])
            p['phase']('global_align_wrist',hold,1,(-1.,-1.))
        p['move']('global_hover',targets+rot[:,2]*.08,28,(-1.,-1.))
        # Reobserve immediately before descending, charging all acquisition/fit.
        estimate=observe((-1.,-1.))
        if not estimate['valid']:
            entry['stop']='insufficient_preclose_rgb';break
        center=np.asarray(estimate['center']);rot=np.asarray(estimate['rotation'])
        targets=center+local_grips@rot.T
        p['set_rotations'](estimate['rotation_targets'])
        p['move']('global_descend',targets,24,(-1.,-1.))
        p['phase']('global_close',targets,22)
        entry['encoder_loaded']=loaded();entry['close_time']=p['time']()
        if not all(entry['encoder_loaded']) and attempt==0:
            p['phase']('global_release',robot(),14,(-1.,-1.))
            p['phase']('global_table_settle',robot(),8,(-1.,-1.))
            clear=robot();clear[:,2]+=.12
            p['move']('global_clear',clear,24,(-1.,-1.))
            entry.update(end_time=p['time'](),stop='encoder_retry')
            continue
        # Recenter while the table still carries the object. This avoids a
        # long vertical lift near one arm's lateral workspace boundary.
        shift=ref_center-center;shift[2]=0.
        p['move']('global_recenter_on_table',targets+shift,70)
        p['phase']('global_table_loaded',robot(),10)
        estimate=observe((1.,1.))
        center=np.asarray(estimate['center']);rot=np.asarray(estimate['rotation'])
        targets=robot();wrist=np.asarray(p['robot']()['rotations'])
        p['set_rotations'](wrist)
        raised=targets+np.array([0.,0.,.16])
        p['move']('global_lift',raised,80)
        normal=rot[:,2];axis=np.cross(normal,[0.,0.,1.])
        length=np.linalg.norm(axis)
        transform=axis/max(length,1e-12)*np.arccos(np.clip(normal[2],-1.,1.))
        raised_center=center+np.array([0.,0.,.16])
        destination=raised_center
        for j in range(50):
            q=(j+1)/50;delta=exp_rotation(transform*q)
            current_center=raised_center*(1-q)+destination*q
            current=current_center+(targets-center)@delta.T
            p['set_rotations'](np.einsum('ij,ajk->aik',delta,wrist))
            p['phase']('global_lift',current,1)
        final=destination+(targets-center)@exp_rotation(transform).T
        p['phase']('global_lift_settle',final,16)
        entry.update(end_time=p['time'](),stop='lifted')
        break
    return log
