import copy

import pytest

from auditembodied.l4_lift import run_trial
from auditembodied.vlm_duration_data import plan
from auditembodied.vlm_duration_lora import task_prompt_for
from auditembodied.vlm_duration_temporal import input_spec


def stage_row():
    robot={'positions':[[0.,0.,1.],[0.,1.,1.]],'fingers':[[.01,-.01],[.02,-.02]],
        'reference_positions':[[0.,0.,1.],[0.,1.,1.]]}
    return {'image':'h1.npz','before_image':'h0.npz','after_prefix_image':'h2.npz',
        'robot':robot,'before_robot':copy.deepcopy(robot),'after_prefix_robot':copy.deepcopy(robot),
        'early_history':[{'stage':'inference_wait','positions':robot['positions']}],
        'late_history':[{'stage':'local_arc_close','positions':robot['positions']}],
        'plan':{**plan({'candidate':'R_left','local_steps':{'local_relevel_steps':60}}),
            'post_prefix_wait_steps':20},'seed':'PRIVATE_SEED','label':'PRIVATE_LABEL',
        'h2_holds_evaluation_only':'PRIVATE_TRUTH'}


def test_stage_inputs_are_eight_frames_and_early_cannot_read_future_prefix():
    row=stage_row();early=input_spec(row,'outcome','stage_early');late=input_spec(row,'outcome','stage_late')
    assert early[0]==['h0.npz','h1.npz'] and late[0]==['h1.npz','h2.npz']
    assert 'NOT executed' in early[1] and 'ALREADY executed' in late[1]
    assert all(x not in early[1]+late[1] for x in ('PRIVATE_SEED','PRIVATE_LABEL','PRIVATE_TRUTH'))
    changed=copy.deepcopy(row);changed['after_prefix_robot']['positions'][0][0]=99
    changed['late_history']=[{'future':'PRIVATE_FUTURE'}];changed['after_prefix_image']='future.npz'
    assert input_spec(changed,'outcome','stage_early')==early
    assert input_spec(row,'start','stage_late')==(['h1.npz'],task_prompt_for(row,'start'))


def test_zero_boundary_wait_keeps_the_existing_physics_exact():
    pytest.importorskip('mujoco')
    def duration(operation,port,candidate,prefix):
        if operation=='finish':return {'candidate':candidate,'accepted':True,'local_relevel_steps':60,'pipeline_wall_seconds':0.}
    args=dict(selector=lambda p:{'candidate':'R_left','parallel_local':True},duration_selector=duration,
        fixed_selector_wait_steps=19)
    old=run_trial(0,'left_loss',**args)
    new=run_trial(0,'left_loss',**args,refresh_duration_observation=True,fixed_duration_wait_steps=0)
    assert not old.get('error') and not new.get('error')
    assert old['trace']==new['trace'] and old['protection']==new['protection']
    assert old['total_post_fault_seconds']==new['total_post_fault_seconds']


def test_post_prefix_budget_and_late_wait_are_both_physical_and_audited():
    pytest.importorskip('mujoco')
    observed=[]
    def duration(operation,port,candidate,prefix):
        observed.append(port)
        if operation=='finish':return {'candidate':candidate,'accepted':True,'local_relevel_steps':60,'pipeline_wall_seconds':.16}
    row=run_trial(0,'left_loss',selector=lambda p:{'candidate':'R_left','parallel_local':True},
        duration_selector=duration,fixed_selector_wait_steps=19,refresh_duration_observation=True,fixed_duration_wait_steps=2)
    assert not row.get('error') and row['abstained'] and row['duration_selection']['deadline_missed']
    assert all('observed_history' in p and not any(k in p for k in ('evaluator','holds','supports')) for p in observed)
    assert len([f for f in row['trace'] if f['stage']=='duration_budget_wait'])==2
    assert len([f for f in row['trace'] if f['stage']=='duration_deadline_wait'])==2
    assert not any(f['stage']=='local_relevel' for f in row['trace'])
    assert abs(row['inference_wait_seconds']-23/20)<1e-8
    assert abs(row['total_post_fault_seconds']-row['inference_wait_seconds']-row['skill_seconds']-row['hold_seconds'])<1e-8


def test_prefix_acquisition_hold_is_physical_and_in_total_wait_cost():
    pytest.importorskip('mujoco')
    def duration(operation,port,candidate,prefix):
        if operation=='finish':return {'candidate':candidate,'accepted':False,'pipeline_wall_seconds':0.}
    row=run_trial(0,'left_loss',selector=lambda p:{'candidate':'R_left','parallel_local':True},
        duration_selector=duration,fixed_selector_wait_steps=19,refresh_duration_observation=True,
        fixed_duration_wait_steps=2,fixed_prefix_observation_wait_steps=3)
    assert not row.get('error')
    holds=[f for f in row['trace'] if f['stage']=='prefix_observation_wait']
    assert len(holds)==3 and all(f['phase']=='recovery' for f in holds)
    assert abs(row['inference_wait_seconds']-24/20)<1e-8
    assert abs(row['total_post_fault_seconds']-row['inference_wait_seconds']-row['skill_seconds']-row['hold_seconds'])<1e-8
