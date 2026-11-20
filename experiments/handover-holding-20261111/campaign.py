"""Position-reference interventions on two unchanged handover recovery paths."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'frozen-source'))
from auditembodied import handover_audit as base
from auditembodied import l3_matrix as runner
from auditembodied.continuous import ContinuousActionExpert
from auditembodied.recovery_skills import actual_eef_rotations
from native import NativeRecorder

PROTOCOL=json.loads((HERE/'protocol.json').read_text())


def one(job):
    seed,scenario,steps,hold,method,phase=job
    name=f'{seed}-{scenario}-{steps:02d}-{hold}-{method}'
    folder=HERE/phase;folder.mkdir(parents=True,exist_ok=True)
    path=folder/(name+'.json');assert not path.exists(),path
    observers=[]

    class Observer(base.AuditObserver):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);self.holding_input=None;self.hold_tracking=[]
            observers.append(self)

        def after_reset(self,env):
            super().after_reset(env)
            self.native=NativeRecorder(env,self.oracle,runner._table_contact)
            env._holding_recorder=self.native

        def capture(self,env,packet,force=False):
            # Preserve the completed final packet when the caller requests a
            # duplicate observation after a skill has returned.
            if force and 'audit' in packet:return
            super().capture(env,packet,force)

        def decision(self,env,episode,capture,started,plan,evidence,observed_state):
            if self.events:
                # Original measured-pose service protocol, independent of the
                # initial holding/window treatment.
                self.native.stage='decision_wait'
                return super().decision(env,episode,capture,started,plan,evidence,observed_state)
            service=time.perf_counter()-started
            n=max(steps,int(math.ceil(service/.05)))
            before=self.oracle.evaluate(env)
            required='receiver_supports' if plan=='continue' else 'giver_holds' if plan=='R1_retarget' else None
            positions=np.stack([env._eef_pos(i) for i in range(2)])
            registered=env._last_command_position_targets.copy()
            rotations=actual_eef_rotations(env)
            grips=self.last_action[[6,13]].copy()
            self.holding_input={'measured_positions':positions.tolist(),'registered_positions':registered.tolist(),
                'shared_rotation_targets':rotations.tolist(),'gripper_commands':grips.tolist(),
                'smooth_transition_seconds':PROTOCOL['smooth_transition_seconds']}
            event={'index':0,'kind':'initial_decision','observation_state':observed_state,
                'observed_time':observed_state['time'],'plan':plan,'evidence':evidence,
                'service_wall_seconds':service,'minimum_window_steps':steps,'physical_wait_steps':n,
                'service_exceeds_fixed_window':bool(service>steps*.05),
                'required_entry_predicate':required,
                'prerequisite_at_observation':bool(before[required]) if required else True,
                'start_time':float(env.sim.data.time),'previous_action14':self.last_action.tolist(),
                'hold_reference_positions':registered.tolist() if hold=='registered' else positions.tolist(),
                'hold_reference_rotations':rotations.tolist(),'held_grippers':grips.tolist()}
            previous_role=self.wait_role;self.wait_role=base.required_role(plan,False)
            self.native.enabled=True;self.native.initial_wait=True
            extra_total=0

            def advance(stage):
                elapsed=float(env.sim.data.time)-event['start_time']
                alpha=0. if hold=='registered' else 1. if hold=='measured_position' else min(1.,elapsed/PROTOCOL['smooth_transition_seconds'])
                target=(1-alpha)*registered+alpha*positions
                current=np.stack([env._eef_pos(i) for i in range(2)])
                record={'time':float(env.sim.data.time),'elapsed':elapsed,'alpha':alpha,
                    'target_positions':target.tolist(),'measured_positions':current.tolist(),
                    'position_error_mm':(np.linalg.norm(target-current,axis=1)*1000).tolist()}
                self.native.stage=stage
                action=base.hold_action(env,target,rotations,grips)
                env.step(action);capture(episode,env,stage,action)
                record['arm_actuator_torques_Nm']=[np.asarray(env.sim.data.qfrc_actuator)[robot._ref_joint_vel_indexes].tolist() for robot in env.raw.robots]
                record['wrist_force_sensor_N']=[np.asarray(robot.ee_force['right']).tolist() for robot in env.raw.robots]
                record['wrist_torque_sensor_Nm']=[np.asarray(robot.ee_torque['right']).tolist() for robot in env.raw.robots]
                self.hold_tracking.append(record)

            for _ in range(n):
                tick=time.perf_counter();recording_before=self.native.overhead_seconds;advance('decision_wait')
                elapsed=time.perf_counter()-tick-(self.native.overhead_seconds-recording_before)
                self.control_times.append(elapsed)
                extra=max(0,int(math.ceil(max(0.,elapsed-.05)/.05)))
                for _ in range(extra):advance('controller_compute_wait')
                extra_total+=extra;self.control_overruns+=extra
            self.native.initial_wait=False;self.wait_role=previous_role
            after=self.oracle.evaluate(env)
            event.update(end_time=float(env.sim.data.time),post_wait_state=base.state(env),
                controller_wait_steps=extra_total,
                prerequisite_at_entry=bool(after[required]) if required else True)
            self.events.append(event)

    original_action=ContinuousActionExpert.action_towards
    original_phase=ContinuousActionExpert.execute_phase

    def recorded_action(env,targets,grippers=(1.,1.),gain=.65):
        env._last_command_position_targets=np.asarray(targets,dtype=float).copy()
        return original_action(env,targets,grippers,gain)

    def recorded_phase(instance,env,episode,stage,*a,**kw):
        recorder=getattr(env,'_holding_recorder',None)
        if recorder is not None:recorder.stage=stage
        return original_phase(instance,env,episode,stage,*a,**kw)

    started=time.perf_counter()
    class ControlClock:
        @staticmethod
        def perf_counter():
            # Passive evaluation is not computation by the deployed controller.
            observer=observers[0] if observers else None
            recording=observer.native.overhead_seconds if observer is not None and hasattr(observer,'native') else 0.
            return time.perf_counter()-recording
    with ExitStack() as stack:
        stack.enter_context(patch.object(base,'time',ControlClock))
        stack.enter_context(patch.object(base,'AuditObserver',Observer))
        stack.enter_context(patch.object(ContinuousActionExpert,'action_towards',staticmethod(recorded_action)))
        stack.enter_context(patch.object(ContinuousActionExpert,'execute_phase',recorded_phase))
        row=base.trial(seed,scenario,method,'actual',HERE,PROTOCOL)
    observer=observers[0]
    row.update(window=f'{steps*50}ms',wait_steps=steps,hold_mode=hold,cohort=phase,
        holding_input=observer.holding_input,hold_tracking=observer.hold_tracking,
        passive_recording_wall_seconds=observer.native.overhead_seconds,
        wall_seconds=time.perf_counter()-started)
    native_path=folder/(name+'-native.npz')
    row['native_samples']=observer.native.save(native_path);row['native_file']=native_path.name
    path.write_text(json.dumps(row,separators=(',',':'))+'\n')
    status={'case':name,'C':row.get('handover_completed'),'label':row['audit']['label'],
        'total':row['audit'].get('total_seconds'),'overrun_steps':observer.control_overruns,
        'initialized':row.get('initialization_supported'),'error':row.get('error'),
        'wall':row['wall_seconds']}
    print(json.dumps(status),flush=True)
    return status


def jobs(phase):
    seeds=PROTOCOL['development_seeds'] if phase.startswith('development') else PROTOCOL['formal_seeds']
    result=[]
    for i,seed in enumerate(seeds):
        scenes=tuple(PROTOCOL['scenarios']);scenes=scenes[i%3:]+scenes[:i%3]
        holds=tuple(PROTOCOL['hold_modes'])
        for j,scene in enumerate(scenes):
            for n in ((1,20) if i%2 else (20,1)):
                shifted=holds[(i+j)%3:]+holds[:(i+j)%3]
                for k,hold in enumerate(shifted):
                    methods=PROTOCOL['methods'] if (i+j+k+n)%2 else PROTOCOL['methods'][::-1]
                    for method in methods:result.append((seed,scene,n,hold,method,phase))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('development','formal'),required=True)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if a.phase=='formal':
        frozen=json.loads((HERE/'formal-freeze.json').read_text())
        for name,digest in frozen['source_sha256'].items():
            assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    assert not (HERE/a.phase).exists(),a.phase
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        statuses=list(pool.map(one,jobs(a.phase)))
    (HERE/(a.phase+'-completion.json')).write_text(json.dumps(statuses,indent=2)+'\n')
    print(json.dumps({'phase':a.phase,'episodes':len(statuses),
        'healthy_initialized':sum(s['initialized'] for s in statuses),
        'errors':sum(bool(s['error']) for s in statuses)}),flush=True)


if __name__=='__main__':main()
