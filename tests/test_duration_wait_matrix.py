import pytest

from auditembodied.duration_wait_matrix import continuation_outcome
from auditembodied.l4_lift import run_trial
from auditembodied.vlm_duration_policy import score_local_family


def test_offline_family_scores_all_durations_even_when_start_gate_rejects():
    calls=[]
    def score(example,task):
        calls.append(task)
        return [.5,.5] if task=='start' else [.9,.1,0.,0.]
    result=score_local_family({},'R_left',.99,score)
    assert calls==['start','outcome','outcome','outcome']
    assert [r['plan']['local_relevel_steps'] for r in result['scored_candidates']]==[30,40,60]
    assert all(not r['accepted'] and r['joint_A_score']==.45 for r in result['scored_candidates'])


def test_continuation_label_does_not_hide_or_relabel_past_guard_damage():
    row={'candidate':'R_left','success':True,'post_wait_state':{'time':1.},
        'post_wait_evaluator':{'holds':[False,True]},
        'pre_skill_protection':{'keeper_grasp_lost_steps':1,'deep_interarm_contact_steps':2},
        'protection':{'keeper_grasp_lost_steps':1,'deep_interarm_contact_steps':2},
        'trace':[{'time':1.05,'phase':'recovery','evaluator':{'holds':[True,True],'table_contact':False}}]}
    label,guards=continuation_outcome(row)
    assert label=='A' and guards['keeper_grasp_lost_steps']==0
    row['protection']['keeper_grasp_lost_steps']+=1
    assert continuation_outcome(row)[0]=='B'
    row['post_wait_evaluator']['holds'][1]=False
    assert continuation_outcome(row)[0]=='E'


def test_fixed_wait_requires_an_explicit_selector_and_separate_latency_mode():
    pytest.importorskip('mujoco')
    with pytest.raises(ValueError):run_trial(0,fixed_selector_wait_steps=19)
    with pytest.raises(ValueError):run_trial(0,selector=lambda p:{},fixed_selector_wait_steps=19,account_selector_latency=True)


def test_nineteen_fixed_steps_preserve_fault_commands_and_whole_cost():
    pytest.importorskip('mujoco')
    row=run_trial(0,'left_loss',selector=lambda p:{'candidate':'R_left','local_relevel_steps':60},fixed_selector_wait_steps=19)
    assert not row.get('error')
    waits=[f for f in row['trace'] if f['stage']=='inference_wait']
    fault=[f for f in row['trace'] if f['phase']=='fault'][-1]
    assert len(waits)==row['inference_wait_steps']==19
    assert row['wait_source']=='fixed_intervention'
    assert all(f['phase']=='recovery' and [f['robot']['action'][i] for i in (6,13)]==[fault['robot']['action'][i] for i in (6,13)] for f in waits)
    assert abs(row['inference_wait_seconds']-.95)<1e-8
    assert abs(row['total_post_fault_seconds']-row['inference_wait_seconds']-row['skill_seconds']-row['hold_seconds'])<1e-8
    assert row['post_wait_state']['time']==waits[-1]['time']
    assert row['pre_skill_protection']['keeper_grasp_lost_steps']==sum(not f['evaluator']['holds'][1] for f in waits)
