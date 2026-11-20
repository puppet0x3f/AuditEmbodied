from types import SimpleNamespace

import numpy as np

from auditembodied.handover_audit import AuditObserver, episode_audit, model_plan, required_role


def test_same_frozen_model_routing_and_post_local_scope():
    threshold=np.full(3,.5)
    assert model_plan([.1,.9,.1],threshold)=='continue'
    assert model_plan([.9,.9,.9],threshold)=='continue'
    assert model_plan([.9,.1,.1],threshold)=='R1_retarget'
    assert model_plan([.9,.1,.1],threshold,True)=='R3'
    assert model_plan([.1,.1,.1],threshold)=='R3'


def test_wait_keeps_local_obligation_until_retry_entry():
    assert required_role('R1_retarget')=='giver_holds'
    assert required_role('R3',True)=='giver_holds'
    assert required_role('R3') is None
    assert required_role('continue')=='receiver_supports'


def test_forced_observation_cannot_erase_a_completed_skill_frame():
    observer=AuditObserver(None,'actual');observer.active='R1_retarget'
    packet={'phase':'recovery','stage':'r1_recovery','robot':{'action14':[0.]*14},
            'evaluator':{'giver_holds':False,'receiver_holds':True,'table_contact':False}}
    observer.capture(None,packet)
    observer.active='none'
    observer.capture(None,packet,force=True)
    assert packet['audit']['role_obligation']=='giver_holds'
    assert not packet['audit']['role_obligation_satisfied']


def test_decision_wait_holds_observed_poses_and_keeps_stale_evidence(monkeypatch):
    import auditembodied.handover_audit as module
    observer=AuditObserver(None,'1000ms');action=np.arange(14.)
    observer.last_action=action.copy()
    env=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(time=1.,qpos=[],qvel=[],ctrl=[],act=[])))
    env._eef_pos=lambda i:np.array([float(i),0.,0.])
    monkeypatch.setattr(module,'actual_eef_rotations',lambda e:np.tile(np.eye(3),(2,1,1)))
    references=[]
    def hold(e,positions,rotations,grips):
        references.append(positions.copy());a=np.zeros(14);a[[6,13]]=grips;return a
    monkeypatch.setattr(module,'hold_action',hold)
    applied=[]
    def step(a):applied.append(a.copy());env.sim.data.time+=.05
    env.step=step
    observer.oracle=SimpleNamespace(evaluate=lambda e:{'receiver_supports':len(applied)==0})
    import time
    observer.decision(env,None,lambda *args:None,time.perf_counter(),'continue',{'probabilities':[.9]}, {'time':1.})
    assert len(applied)==20 and all(np.array_equal(a[[6,13]],action[[6,13]]) for a in applied)
    assert all(np.array_equal(p,[[0.,0.,0.],[1.,0.,0.]]) for p in references)
    assert all(not np.array_equal(a,action) for a in applied)
    event=observer.events[0]
    assert event['prerequisite_at_observation'] and not event['prerequisite_at_entry']
    assert event['evidence']=={'probabilities':[.9]}


def test_prior_role_violation_survives_a_successful_global_retry():
    frames=[]
    phases=[('decision_wait','recovery',True),('r1_recovery','recovery',False),
            ('r3_table_settle','recovery',True),('giver_open','release',True),('post_release_hold','hold',True)]
    for i,(stage,phase,role_ok) in enumerate(phases):
        frames.append({'time':(i+1)*.05,'stage':stage,'phase':phase,
            'evaluator':{'giver_holds':True,'receiver_holds':True,'table_contact':stage=='r3_table_settle'},
            'audit':{'role_obligation':'giver_holds' if not role_ok else None,'role_obligation_satisfied':role_ok}})
    row={'trace':frames,'prefix_state':{'time':0.},'skill_calls':[], 'decision_events':[],
        'protection':{'max_unsupported_seconds':0.,'deep_robot_robot_contact_steps':0},
        'blocked_calls':[],'failure_reason':None,'handover_completed':True,'success':True,
        'protection_violation':False,'used_global_recovery':True,'table_cycle_verified':True,
        'recovery':{'sequence':['R1_retarget','R3']}}
    result=episode_audit(row)
    assert result['terminal_completion'] and not result['protected_completion'] and result['label']=='B'
    assert result['legacy_protected_completion'] and result['local_then_global']
    assert result['cost_identity']
    assert abs(result['total_seconds']-.25)<1e-9
