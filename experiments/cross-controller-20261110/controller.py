"""Independent RGB/encoder feedback recovery; no evaluation-state inputs."""
import time
import numpy as np
from geometry import exp_rotation
from observation import Observer, reference
from auditembodied.recovery_skills import actual_eef_rotations,actual_gripper_positions


def rotation_vector(r):
    c=np.clip((np.trace(r)-1)/2,-1.,1.);angle=np.arccos(c)
    v=np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])
    return .5*v if angle<1e-7 else angle/(2*np.sin(angle))*v


def loaded(fingers):
    widths=np.mean(np.abs(fingers),axis=1)
    return (widths>.004)&(widths<.028)


class VisualRecovery:
    def __init__(self):
        self.sensor=Observer(True);self.services=[];self.images=[];self.events=[];self.wait_steps=0

    def __call__(self,env,table,registered,rotations,phase,move):
        ref_center,ref_rot=reference(table,rotations)
        table_z=ref_center[2]-.07
        def positions():return np.stack([env._eef_pos(i) for i in range(2)])
        def observe(name,grips):
            observed=float(env.sim.data.time);started=time.perf_counter()
            result=self.sensor(env,table,rotations)
            elapsed=time.perf_counter()-started
            steps=max(1,int(np.ceil(elapsed/.05)))
            self.images.append(np.stack(self.sensor.images))
            result.update(stage=name,observed_time=observed,service_wall_seconds=elapsed,service_wait_steps=steps)
            self.services.append(result);self.wait_steps+=steps
            phase('vision_wait_'+name,positions(),steps,grips)
            return result['estimate'] if result['estimate']['valid'] else None
        def servo(name,goal,rot,grips,limit=60,tolerance=.005):
            stable=0
            for j in range(limit):
                current=positions();dr=goal-current
                norms=np.linalg.norm(dr,axis=1)
                command=current+dr*np.minimum(1.,.020/np.maximum(norms,1e-12))[:,None]
                now=actual_eef_rotations(env);angles=[]
                for arm in range(2):
                    rv=rotation_vector(rot[arm]@now[arm].T);angles.append(np.linalg.norm(rv))
                    env.rotation_targets[arm]=exp_rotation(rv*min(1.,.06/max(np.linalg.norm(rv),1e-12)))@now[arm]
                phase(name,command,1,grips)
                stable=stable+1 if max(norms)<tolerance and max(angles)<.12 else 0
                if stable>=4:break
            return j+1
        def rigid(name,estimate,center,rot,active,grips,steps):
            start_center=np.asarray(estimate['center']);start_rot=np.asarray(estimate['rotation'])
            p0=positions();r0=actual_eef_rotations(env)
            local=(p0-start_center)@start_rot
            rv=rotation_vector(rot@start_rot.T)
            for j in range(steps):
                alpha=(j+1)/steps;q=exp_rotation(alpha*rv)@start_rot
                c=(1-alpha)*start_center+alpha*center
                target=p0.copy()
                for arm in range(2):
                    if active[arm]:
                        target[arm]=c+q@local[arm]
                        env.rotation_targets[arm]=q@start_rot.T@r0[arm]
                phase(name,target,1,grips)
        def finish(reason=None):
            result={'controller_services':self.services,'controller_events':self.events,
                'executor_wait_steps':self.wait_steps,'controller':'rgb_pose_feedback',
                'controller_stop_reason':reason}
            if reason:result.update(abstained=True,abstention_reason=reason)
            return result

        active=loaded(actual_gripper_positions(env))
        grips=tuple(1. if yes else -1. for yes in active)
        # Clear the open hand while the encoder-loaded hand retains its target.
        clear=positions();clear[~active,2]+=.08
        servo('vision_open_hand_clear',clear,actual_eef_rotations(env),grips,limit=18)
        estimate=observe('entry',grips)
        if estimate is None:return finish('vision_entry_unavailable')
        # A one-handed tilted object need not follow wrist rotation rigidly.
        # Retain the loaded wrist orientation; approach the observed table plane
        # vertically and reobserve instead of imposing a free-space level pose.
        for segment in range(4):
            half=float(np.abs(np.asarray(estimate['rotation'])[2])@np.array([.07,.07,.07]))
            gap=estimate['center'][2]-half-table_z
            if gap<=.004:break
            target=positions();target[active,2]-=min(.04,gap+.003)
            move('vision_lower_feedback',target,24,grips)
            estimate=observe('lower_'+str(segment),grips)
            if estimate is None:return finish('vision_lower_unavailable')
        phase('vision_setdown_loaded',positions(),8,grips)
        estimate=observe('release_check',grips)
        if estimate is None:return finish('vision_release_unavailable')
        shift=ref_center[:2]-np.asarray(estimate['center'])[:2]
        distance=np.linalg.norm(shift)
        if distance>.02 and any(active):
            # Move the still-grasped, table-supported object back into the
            # registered bimanual workspace before opening the supporting hand.
            shift*=min(1.,.12/distance)
            centered=positions();centered[active,:2]+=shift
            move('vision_table_recenter',centered,40,grips)
            phase('vision_recenter_settle',positions(),8,grips)
            estimate=observe('recenter_check',grips)
            if estimate is None:return finish('vision_recenter_unavailable')
            for segment in range(4):
                half=float(np.abs(np.asarray(estimate['rotation'])[2])@np.array([.07,.07,.07]))
                gap=estimate['center'][2]-half-table_z
                if gap<=.004:break
                target=positions();target[active,2]-=min(.04,gap+.003)
                move('vision_recenter_lower',target,24,grips)
                estimate=observe('recenter_lower_'+str(segment),grips)
                if estimate is None:return finish('vision_recenter_lower_unavailable')
        body_half_height=float(np.abs(np.asarray(estimate['rotation'])[2])@np.array([.07,.07,.07]))
        if estimate['center'][2]-body_half_height>table_z+.008:
            return finish('visual_setdown_height_unconfirmed')
        phase('global_release',positions(),16,(-1.,-1.))
        phase('global_table_settle',positions(),8,(-1.,-1.))
        clear=positions();clear[:,2]+=.09
        servo('vision_clear',clear,actual_eef_rotations(env),(-1.,-1.),limit=32)

        for attempt in range(2):
            estimate=observe('pickup_'+str(attempt),(-1.,-1.) if attempt==0 else grips)
            if estimate is None:return finish('vision_pickup_unavailable')
            pickup=np.asarray(estimate['targets']);rot=np.asarray(estimate['rotation_targets'])
            normal=np.asarray(estimate['rotation'])[:,2]
            hover=pickup+.07*normal
            if attempt:
                active=loaded(actual_gripper_positions(env));hover[active]=positions()[active]
                rot[active]=actual_eef_rotations(env)[active]
                grips=tuple(1. if v else -1. for v in active)
            else:grips=(-1.,-1.)
            event={'attempt':attempt+1,'start_time':float(env.sim.data.time)}
            self.events.append(event)
            servo('vision_hover',hover,rot,grips,limit=64)
            estimate=observe('preclose_'+str(attempt),grips)
            if estimate is None:return finish('vision_preclose_unavailable')
            pickup=np.asarray(estimate['targets']);rot=np.asarray(estimate['rotation_targets'])
            if attempt:
                pickup[active]=positions()[active];rot[active]=actual_eef_rotations(env)[active]
            servo('vision_descend',pickup,rot,grips,limit=48,tolerance=.003)
            estimate=observe('contact_align_'+str(attempt),grips)
            if estimate is None:return finish('vision_contact_alignment_unavailable')
            pickup=np.asarray(estimate['targets']);rot=np.asarray(estimate['rotation_targets'])
            if attempt:
                pickup[active]=positions()[active];rot[active]=actual_eef_rotations(env)[active]
            servo('vision_contact_align',pickup,rot,grips,limit=20,tolerance=.003)
            error=np.linalg.norm(positions()-pickup,axis=1)
            if max(error)>.015:
                event.update(end_time=float(env.sim.data.time),tracking_error_m=error.tolist(),
                    encoder_loaded=[False,False],reason='pickup_target_not_reached')
                continue
            phase('vision_close',pickup,22)
            env.rotation_targets=actual_eef_rotations(env).copy()
            test=positions();test[:,2]+=.008
            move('vision_grasp_load_test',test,8)
            active=loaded(actual_gripper_positions(env));grips=tuple(1. if v else -1. for v in active)
            event.update(end_time=float(env.sim.data.time),encoder_loaded=active.tolist())
            if all(active):break
        if not all(active):return finish('visual_regrasp_encoder_unconfirmed')
        # Preserve the grasp actually established. Re-fitting a grasped,
        # occluded object must not impose a new rigid wrist relation in flight.
        env.rotation_targets=actual_eef_rotations(env).copy()
        lift=positions();lift[:,2]+=max(.10,float(np.mean(registered[:,2]-lift[:,2])))
        move('global_lift',lift,60)
        phase('global_lift_settle',positions(),12)
        return finish()
