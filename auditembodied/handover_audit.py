"""Frozen handover policies with physical decision windows and path-level audit."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from . import l3_matrix as runner
from .continuous import ContinuousActionExpert, ContinuousR1Recovery
from .l0 import OracleVerifier
from .l3_matrix import MatrixConfig
from .l3_visual_test import DecisionObserver
from .recovery_skills import ContinuousR2Recovery, ContinuousR3Recovery, PoseRetargetedR1Recovery, actual_gripper_positions, actual_eef_rotations
from .visual_recovery import MultiSupportVerifier, VisualRecoverySelector


METHODS=('visual','proprio','rule','R3')
SCENARIOS=('nominal','object_sham','object_rotation')
WINDOWS=('actual','1000ms','2000ms')
WAIT_STAGES=('decision_wait','controller_compute_wait')


def required_role(plan, after_local=False):
    if plan=='continue':return 'receiver_supports'
    if plan=='R1_retarget' or (plan=='R3' and after_local):return 'giver_holds'
    return None


def model_plan(probabilities, thresholds, after_local=False):
    giver,receiver,coordinated=np.asarray(probabilities)>=thresholds
    ready=bool(receiver and (coordinated or not giver))
    return 'continue' if ready else 'R3' if after_local or not giver else 'R1_retarget'


def state(env):
    return {**{k:np.asarray(getattr(env.sim.data,k)).tolist() for k in ('qpos','qvel','ctrl','act')},
            'time':float(env.sim.data.time)}


def intervals(flags):
    count=maximum=0
    for flag in flags:
        count=count+1 if flag else 0;maximum=max(maximum,count)
    return maximum


def hold_action(env, positions, rotations, grippers):
    """Hold measured robot poses; repeating a nonzero OSC delta would move them."""
    from robosuite.utils import transform_utils as transforms
    action=ContinuousActionExpert.action_towards(env,positions,grippers=grippers,gain=.8)
    for arm,current in enumerate(actual_eef_rotations(env)):
        error=transforms.quat2axisangle(transforms.mat2quat(rotations[arm]@current.T))
        action[arm*7+3:arm*7+6]=np.clip(error/.5,-1,1)
    return action


class AuditObserver(DecisionObserver):
    def __init__(self, mode, window):
        super().__init__(mode)
        self.window=window;self.events=[];self.calls=[];self.active='none';self.wait_role=None
        self.last_packet=None;self.last_action=None;self.table_handoff=False;self.carrier=None
        self.oracle=OracleVerifier();self.control_times=[];self.control_overruns=0

    def after_reset(self, env):
        super().after_reset(env)
        from .l0 import CAMERAS
        # Renderer setup is common preparation, before the physical fault.
        for camera in CAMERAS:env.sim.render(camera_name=camera,width=64,height=64)
        hold_action(env,np.stack([env._eef_pos(i) for i in range(2)]),actual_eef_rotations(env),(1.,1.))
        hammer=env.raw.hammer
        self.initial_object_parameters={name:float(getattr(hammer,name)) for name in
            ('handle_length','handle_radius') if hasattr(hammer,name)}

    def capture(self, env, packet, force=False):
        # A forced observation reuses the last physical frame. Preserve the
        # obligation recorded while its skill was active, even after return.
        if force and 'audit' in packet:return
        role=None
        stage,phase=packet['stage'],packet['phase'];truth=packet['evaluator']
        if phase in ('release','hold'):role='receiver_supports'
        elif phase=='recovery':
            if stage=='decision_wait':role=self.wait_role
            elif stage=='controller_compute_wait' and self.last_packet is not None:
                role=self.last_packet.get('audit',{}).get('role_obligation')
            elif self.active in ('R1_retarget','R3_receiver_catch'):role='giver_holds'
            elif self.active=='R3_transfer_R2':role='receiver_holds'
            elif self.active=='R3':
                if stage in ('r3_table_loaded','r3_table_release','r3_table_settle') and truth['table_contact']:
                    self.table_handoff=True
                before_table=('r3_free_other_gripper','r3_park_other_arm','r3_recenter_carrier',
                    'r3_transport_above_real_table','r3_lower_to_table','r3_table_loaded','r3_table_release','r3_table_settle')
                if stage in before_table:
                    if self.carrier is None:
                        grips=np.asarray(packet['robot']['action14'])[[6,13]]
                        candidates=np.flatnonzero(grips>0)
                        if len(candidates)==1:self.carrier=int(candidates[0])
                    if not self.table_handoff and self.carrier is not None:
                        role='giver_holds' if self.carrier==0 else 'receiver_holds'
                elif stage in ('r3_return_to_handover','r3_handover_settle'):role='giver_holds'
        packet['audit']={'active_skill':self.active,'role_obligation':role,
            'role_obligation_satisfied':bool(truth[role]) if role else True,
            'table_handoff_observed':self.table_handoff}
        self.last_packet=packet;self.last_action=np.array(packet['robot']['action14'],copy=True)

    def decision(self, env, episode, capture, started, plan, evidence, observed_state):
        elapsed=time.perf_counter()-started
        minimum={'actual':0,'1000ms':20,'2000ms':40}[self.window]
        steps=max(minimum,int(math.ceil(elapsed/.05)))
        after_local=bool(self.events)
        prerequisite='receiver_supports' if plan=='continue' else 'giver_holds' if plan=='R1_retarget' else None
        before=self.oracle.evaluate(env)
        event={'index':len(self.events),'kind':'post_local_check' if after_local else 'initial_decision',
            'observation_state':observed_state,'observed_time':observed_state['time'],'plan':plan,
            'evidence':evidence,'service_wall_seconds':elapsed,'minimum_window_steps':minimum,
            'physical_wait_steps':steps,'service_exceeds_fixed_window':bool(minimum and elapsed>minimum/20),
            'required_entry_predicate':prerequisite,
            'prerequisite_at_observation':bool(before[prerequisite]) if prerequisite else True,
            'start_time':float(env.sim.data.time),'previous_action14':self.last_action.tolist()}
        previous_role=self.wait_role;self.wait_role=required_role(plan,after_local)
        positions=np.stack([env._eef_pos(i) for i in range(2)]);rotations=actual_eef_rotations(env)
        grips=self.last_action[[6,13]].copy();extra_total=0
        event.update(hold_reference_positions=positions.tolist(),hold_reference_rotations=rotations.tolist(),held_grippers=grips.tolist())
        for _ in range(steps):
            tick=time.perf_counter();action=hold_action(env,positions,rotations,grips)
            env.step(action);capture(episode,env,'decision_wait',action)
            elapsed_control=time.perf_counter()-tick
            self.control_times.append(elapsed_control)
            extra=max(0,int(math.ceil(max(0.,elapsed_control-.05)/.05)))
            for _ in range(extra):
                action=hold_action(env,positions,rotations,grips);env.step(action);capture(episode,env,'controller_compute_wait',action)
            extra_total+=extra;self.control_overruns+=extra
        self.wait_role=previous_role
        after=self.oracle.evaluate(env)
        event.update(end_time=float(env.sim.data.time),post_wait_state=state(env),controller_wait_steps=extra_total,
            prerequisite_at_entry=bool(after[prerequisite]) if prerequisite else True)
        self.events.append(event)


class TimedPolicy:
    def __init__(self, method, observer, model=None, rule_tolerance=.05):
        self.method=method;self.audit=observer;self.model=model;self.rule_tolerance=rule_tolerance

    def run(self, env, expert, episode, capture, positions, rotations, gaps, table, global_recovery, *, observe):
        audit=self.audit
        if self.method in ('visual','proprio'):
            context={}
            def observed():
                context['state']=state(env);context['started']=time.perf_counter()
                return observe()
            original=self.model.predict_proba
            def predicted(rgb,robot):
                p=original(rgb,robot)
                plan=model_plan(p[0],self.model.thresholds,bool(audit.events))
                audit.decision(env,episode,capture,context['started'],plan,
                    {'probabilities':p[0].tolist(),'thresholds':self.model.thresholds.tolist()},context['state'])
                return p
            with patch.object(self.model,'predict_proba',side_effect=predicted):
                return VisualRecoverySelector(self.model).run(env,expert,episode,capture,positions,rotations,gaps,table,global_recovery,observe=observed)
        observed_state=state(env);started=time.perf_counter()
        initial='R3' if self.method=='R3' else 'R1_retarget'
        audit.decision(env,episode,capture,started,initial,{'fixed_choice':initial},observed_state)
        if self.method=='R3':
            result=global_recovery.run(env,expert,episode,capture,positions,gaps,table)
            return {'skill':'R3','sequence':['R3'],'used_global_recovery':True,'global_result':result}
        # Same sensor rule and threshold as LocalThenTableRule; only the wait is new.
        PoseRetargetedR1Recovery().run(env,expert,episode,capture,positions,rotations)
        observed_state=state(env);started=time.perf_counter()
        fingers=np.abs(actual_gripper_positions(env)[1]);reference=float(gaps[1])
        error=float(np.max(np.abs(fingers-reference))/max(reference,1e-6))
        escalate=not bool(np.all(np.isfinite(fingers)) and error<=self.rule_tolerance)
        audit.decision(env,episode,capture,started,'R3' if escalate else 'continue',
            {'receiver_fingers':fingers.tolist(),'reference':reference,'relative_gap_error':error,'tolerance':self.rule_tolerance},observed_state)
        if escalate:global_recovery.run(env,expert,episode,capture,positions,gaps,table)
        return {'skill':'rule','sequence':['R1_retarget','R3'] if escalate else ['R1_retarget'],
                'used_global_recovery':escalate,'relative_gap_error':error,'relative_gap_tolerance':self.rule_tolerance}


def trial(seed, scenario, method, window, output, protocol):
    output=Path(output);started=time.perf_counter()
    mode={'visual':'image_robot','proprio':'robot'}.get(method)
    observer=AuditObserver(mode,window)
    model=MultiSupportVerifier.load(output/'models'/('visual-support.npz' if method=='visual' else 'support-robot.npz')) if mode else None
    policy=TimedPolicy(method,observer,model,protocol['physical_config']['rule_gap_tolerance'])
    class TimedExpert(ContinuousActionExpert):
        def execute_phase(self, env, episode, stage, targets, contact_mode, grippers=(1.,1.),gain=.65,steps=None,capture=None):
            for _ in range(self.phase_steps if steps is None else max(1,int(steps))):
                tick=time.perf_counter()
                super().execute_phase(env,episode,stage,targets,contact_mode,grippers,gain,steps=1,capture=capture)
                elapsed=time.perf_counter()-tick
                if observer.last_packet['phase'] in ('recovery','release','hold'):
                    observer.control_times.append(elapsed)
                    extra=max(0,int(math.ceil(max(0.,elapsed-.05)/.05)))
                    positions=np.stack([env._eef_pos(i) for i in range(2)]);rotations=actual_eef_rotations(env)
                    grips=observer.last_action[[6,13]].copy()
                    for _ in range(extra):
                        action=hold_action(env,positions,rotations,grips);env.step(action);capture(episode,env,'controller_compute_wait',action)
                    observer.control_overruns+=extra
    def wrapped(original, name, required):
        def run(instance,env,expert,episode,capture,*args,**kwargs):
            previous=observer.active
            actual_name=('R3_receiver_catch' if previous=='R3' else 'R1_retarget') if name=='R1' else 'R3_transfer_R2' if name=='R2' else 'R3'
            predicate=observer.oracle.evaluate(env)
            event={'index':len(observer.calls),'skill':actual_name,'parent':previous,'start_time':float(env.sim.data.time),
                'prerequisite':required,'valid_at_entry':bool(predicate[required]) if required else True,
                'start_evaluator':predicate}
            observer.calls.append(event);observer.active=actual_name
            if name=='R3':observer.table_handoff=False;observer.carrier=None
            try:
                result=original(instance,env,expert,episode,capture,*args,**kwargs)
                if name=='R3':event['pickup_attempts']=result.get('pickup_attempts',[])
                return result
            finally:
                event.update(end_time=float(env.sim.data.time),end_evaluator=observer.oracle.evaluate(env))
                observer.active=previous
        return run
    with ExitStack() as guards:
        guards.enter_context(patch.object(runner,'ContinuousActionExpert',TimedExpert))
        for cls,name,required in [(ContinuousR1Recovery,'R1','giver_holds'),(ContinuousR2Recovery,'R2','receiver_holds'),(ContinuousR3Recovery,'R3',None)]:
            guards.enter_context(patch.object(cls,'run',wrapped(cls.run,name,required)))
        row=runner.run_trial(seed,scenario,'visual',MatrixConfig(**protocol['physical_config']),observer=observer,recovery_policy=policy)
    row.update(method=method,window=window,decision_events=observer.events,skill_calls=observer.calls,
        controller_overrun_steps=observer.control_overruns,max_control_wall_seconds=max(observer.control_times,default=0.),
        object_parameters=getattr(observer,'initial_object_parameters',{}),wall_seconds=time.perf_counter()-started)
    if mode:
        folder=output/'observations';folder.mkdir(exist_ok=True)
        path=folder/f'{seed}-{scenario}-{method}-{window}.npz';observer.save_inputs(path);row['policy_input_file']=str(path)
    row['audit']=episode_audit(row)
    return row


def episode_audit(row):
    frames=[f for f in row['trace'] if f['phase'] in ('recovery','release','hold')]
    if not frames:
        return {'complete':False,'protected_completion':False,'label':'uninitialized','total_seconds':None}
    dt=.05
    wait=[f for f in frames if f['stage'] in WAIT_STAGES]
    movement=[f for f in frames if f['phase']=='recovery' and f['stage'] not in WAIT_STAGES]
    release=[f for f in frames if f['phase']=='release' and f['stage'] not in WAIT_STAGES]
    hold=[f for f in frames if f['phase']=='hold' and f['stage'] not in WAIT_STAGES]
    total=frames[-1]['time']-row['prefix_state']['time']
    assert abs(total-dt*len(frames))<1e-7
    unsupported=[not(f['evaluator']['giver_holds'] or f['evaluator']['receiver_holds'] or f['evaluator']['table_contact']) for f in frames]
    max_gap=intervals(unsupported)
    assert abs(max_gap/20-row['protection']['max_unsupported_seconds'])<1e-8
    obligations=[not f['audit']['role_obligation_satisfied'] for f in frames]
    bad_calls=[c for c in row['skill_calls'] if not c['valid_at_entry']]
    bad_decisions=[d for d in row['decision_events'] if not d['prerequisite_at_entry']]
    valid=not(bad_calls or bad_decisions)
    shared=bool(max_gap>2 or row['protection']['deep_robot_robot_contact_steps'])
    hard=bool(row['blocked_calls'] or row['failure_reason']=='numerical_instability')
    role=any(obligations)
    complete=bool(row['handover_completed'])
    cycle=not row.get('used_global_recovery') or bool(row.get('table_cycle_verified'))
    protected=bool(valid and complete and cycle and not(shared or role or hard))
    label='E' if not valid else 'A' if protected else 'B' if complete else 'D' if shared or role or hard or not cycle else 'C'
    counts=Counter(f['audit']['role_obligation'] for f in frames if not f['audit']['role_obligation_satisfied'])
    return {'complete':True,'terminal_completion':complete,'all_entry_prerequisites':valid,'global_cycle_verified':cycle,
        'protected_completion':protected,'legacy_protected_completion':bool(row['success'] and not row['protection_violation']),
        'label':label,'shared_guard_violation':shared,'role_violation':role,'interface_violation':not cycle,'role_violation_samples':dict(counts),
        'maximum_registered_support_gap_ms':max_gap*50,'role_loss_samples':sum(obligations),
        'invalid_skill_entries':len(bad_calls),'invalid_decision_entries':len(bad_decisions),
        'observation_valid_entry_invalid_decisions':sum(d['prerequisite_at_observation'] and not d['prerequisite_at_entry'] for d in row['decision_events']),
        'role_violations_during_wait':sum(not f['audit']['role_obligation_satisfied'] for f in wait),
        'total_seconds':total,'wait_seconds':len(wait)*dt,'recovery_motion_seconds':len(movement)*dt,
        'release_seconds':len(release)*dt,'verification_seconds':len(hold)*dt,
        'cost_identity':abs(total-(len(wait)+len(movement)+len(release)+len(hold))*dt)<1e-7,
        'decision_calls':len(row['decision_events']),
        'top_level_sequence':row.get('recovery',{}).get('sequence',[]),
        'local_then_global':row.get('recovery',{}).get('sequence')==['R1_retarget','R3'],
        'global_pickup_attempts':sum(len(c.get('pickup_attempts',[])) for c in row['skill_calls']),
        'max_decision_service_wall_seconds':max((d['service_wall_seconds'] for d in row['decision_events']),default=0.),
        'fixed_window_exceeded':sum(d['service_exceeds_fixed_window'] for d in row['decision_events'])}


def run_seed(job):
    seed,index,output,protocol,engineering=job
    rows=[]
    windows=('actual',) if engineering else tuple(WINDOWS[(j+index)%3] for j in range(3))
    for j,scenario in enumerate(SCENARIOS):
        for k,window in enumerate(windows):
            shift=(index+j+WINDOWS.index(window))%4
            for position,method in enumerate(METHODS[shift:]+METHODS[:shift]):
                row=trial(seed,scenario,method,window,output,protocol)
                row['execution_position']=position;rows.append(row)
                path=Path(output)/('engineering' if engineering else 'seed-shards');path.mkdir(exist_ok=True)
                with (path/f'{seed}.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                print(json.dumps({'seed':seed,'scenario':scenario,'method':method,'window':window,
                    'protected':row['audit']['protected_completion'],'total':row['audit']['total_seconds'],'error':row.get('error')}),flush=True)
    return len(rows)


def run(output,workers,engineering=False):
    output=Path(output);protocol=json.loads((output/'protocol.json').read_text())
    seeds=[60] if engineering else protocol['seeds']
    if not engineering:
        audit=json.loads((output/'seed-audit.json').read_text())
        assert not audit['used_overlap'] and not audit['unresolved_metadata_read_errors']
        assert seeds==audit['requested_seeds']
        assert all(hashlib.sha256((output/'models'/name).read_bytes()).hexdigest()==digest for name,digest in protocol['model_hashes'].items())
        assert not list((output/'seed-shards').glob('*.jsonl')) if (output/'seed-shards').exists() else True
        frozen={str(p.relative_to(output/'frozen-source')):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (output/'frozen-source/auditembodied').glob('*.py')}
        protocol_hash=hashlib.sha256((output/'protocol.json').read_bytes()).hexdigest()
        (output/'formal-freeze.json').write_text(json.dumps({'sources':frozen,'model_hashes':protocol['model_hashes'],'protocol_sha256':protocol_hash},indent=2)+'\n')
    jobs=[(s,i,str(output),protocol,engineering) for i,s in enumerate(seeds)]
    count=0
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(run_seed,job) for job in jobs]):count+=future.result()
    if not engineering:
        with (output/'trials.jsonl').open('w') as target:
            for seed in seeds:target.write((output/'seed-shards'/f'{seed}.jsonl').read_text())
        assert count==720
        assert hashlib.sha256((output/'protocol.json').read_bytes()).hexdigest()==protocol_hash
        assert frozen=={str(p.relative_to(output/'frozen-source')):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (output/'frozen-source/auditembodied').glob('*.py')}
        assert all(hashlib.sha256((output/'models'/name).read_bytes()).hexdigest()==digest for name,digest in protocol['model_hashes'].items())
        (output/'complete.json').write_text(json.dumps({'episodes':count,'models_and_source_unchanged':True})+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--engineering',action='store_true');a=p.parse_args();run(a.output_dir,a.workers,a.engineering)


if __name__=='__main__':main()
