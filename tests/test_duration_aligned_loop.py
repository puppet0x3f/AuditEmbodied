import json
from pathlib import Path

import pytest

from auditembodied.l4_lift import run_trial,wait_control_steps
from auditembodied.vlm_duration_policy import choose_scope,choose_local_duration


def test_unique_role_prefix_gate_does_not_claim_outcome_acceptance():
    calls=[]
    def score(example,task):
        calls.append(task)
        return [1.,0.] if example['plan']['repair_arm']=='left' else [0.,1.]
    result=choose_scope({},.99,score)
    assert result['candidate']=='R_left' and result['parallel_local'] and not result['accepted']
    assert calls==['start','start']


def test_ambiguous_role_uses_full_selector_with_cached_start_scores():
    calls=[]
    def score(example,task):
        calls.append(task)
        if task=='start':return [1.,0.]
        return [0.,1.,0.,0.] if example['plan']['kind']=='continue' else [1.,0.,0.,0.]
    result=choose_scope({},.99,score)
    assert result['candidate']=='R_left' and result['local_relevel_steps']==30
    assert not result['parallel_local'] and result['accepted']
    assert calls.count('start')==2


def test_local_duration_refreshed_gate_and_no_untrained_global_fallback():
    calls=[]
    def score(example,task):
        p=example['plan'];calls.append((task,p))
        assert p['kind']=='local' and p['repair_arm']=='right'
        if task=='start':return [1.,0.]
        return [1.,0.,0.,0.] if p['local_relevel_steps']==60 else [0.,1.,0.,0.]
    result=choose_local_duration({},'R_right',.99,score)
    assert result['accepted'] and result['local_relevel_steps']==60
    assert len(calls)==4
    result=choose_local_duration({},'R_right',.99,lambda e,t:[.98,.02] if t=='start' else [1.,0.,0.,0.])
    assert not result['accepted'] and not result['scored_candidates']


def test_on_time_prefix_consumption_matches_saved_physics_exactly():
    pytest.importorskip('mujoco')
    saved=Path('outputs/l4-duration-relevel-20261108/pilot-trials.jsonl')
    if not saved.is_file():
        pytest.skip('requires the recorded duration pilot trials')
    def duration(operation,port,candidate,prefix):
        if operation=='finish':return {'candidate':candidate,'accepted':True,'local_relevel_steps':60,'pipeline_wall_seconds':.5}
    row=run_trial(5,'left_loss',selector=lambda p:{'candidate':'R_left','parallel_local':True},duration_selector=duration)
    assert not row.get('error')
    old=next(r for r in map(json.loads,saved.read_text().splitlines())
        if r['seed']==5 and r['scenario']=='left_loss' and r['candidate']=='R_left' and r['time_scale']==1.5)
    assert row['trace']==old['trace']
    assert row['duration_selection']['prefix_seconds']>=1.9-1e-8
    assert row['duration_selection']['overlapped_seconds']==.5
    assert row['inference_wait_seconds']==0
    assert row['success']==old['success'] and row['protection']==old['protection']


def test_late_forecast_is_discarded_and_all_uncovered_wait_is_executed():
    pytest.importorskip('mujoco')
    def duration(operation,port,candidate,prefix):
        if operation=='finish':return {'candidate':candidate,'accepted':True,'local_relevel_steps':60,'pipeline_wall_seconds':prefix+.12}
    row=run_trial(0,'left_loss',selector=lambda p:{'candidate':'R_left','parallel_local':True},duration_selector=duration)
    assert not row.get('error')
    assert row['abstained'] and not row['success'] and row['failure_reason']=='duration_deadline_missed'
    waits=[f for f in row['trace'] if f['stage']=='duration_deadline_wait']
    assert len(waits)==wait_control_steps(.12)==3
    assert all(f['phase']=='recovery' and f['robot']['action'][6]==f['robot']['action'][13]==1. for f in waits)
    assert not any(f['stage']=='local_relevel' for f in row['trace'])
    assert row['local_steps']['local_relevel_steps'] is None
    assert abs(row['total_post_fault_seconds']-row['inference_wait_seconds']-row['skill_seconds']-row['hold_seconds'])<1e-8
