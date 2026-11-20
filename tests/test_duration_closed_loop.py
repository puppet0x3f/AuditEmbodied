import json
from pathlib import Path
import time

import pytest

from auditembodied.l4_lift import run_trial,wait_control_steps
from auditembodied.duration_pilot import STATE_KEYS,retimed_config
from auditembodied.vlm_duration_policy import candidate_bank,choose_candidate


def test_latency_rounds_up_to_complete_audited_control_steps():
    for seconds in [0.,.001,.05,.101,.35,1.003]:
        steps=wait_control_steps(seconds)
        assert steps*.05+1e-12>=seconds
        assert steps*.05-seconds<.05+1e-12
    with pytest.raises(ValueError):wait_control_steps(-1.)


def test_policy_selects_cheapest_admitted_plan_and_prunes_invalid_role():
    calls=[]
    def score(example,task):
        p=example['plan'];calls.append((task,p))
        if task=='start':return [1.,0.] if p['repair_arm']=='left' else [0.,1.]
        if p['kind']=='continue':return [0.,0.,1.,0.]
        if p['kind']=='global':return [1.,0.,0.,0.]
        return [.2,.8,0.,0.] if p['local_relevel_steps']==30 else [.9999,.0001,0.,0.]
    result=choose_candidate({},.99,score)
    assert result['candidate']=='R_left' and result['local_relevel_steps']==40
    assert not any(task=='outcome' and p.get('repair_arm')=='right' for task,p in calls)
    assert len(candidate_bank())==8
    assert [x['planned_recovery_seconds'] for x in candidate_bank()]==sorted(x['planned_recovery_seconds'] for x in candidate_bank())


def test_global_must_pass_same_threshold_and_can_abstain():
    def score(example,task):return [.98,.02] if task=='start' else [1.,0.,0.,0.]
    assert choose_candidate({},.99,lambda e,t:[0.,0.,1.,0.] if e['plan']['kind']=='continue' else score(e,t))['candidate']=='R_global'
    result=choose_candidate({},.99,lambda e,t:[.1,.9] if t=='start' else [.1,.9,0.,0.])
    assert result['candidate']=='abstain' and not result['accepted']


def test_default_runner_matches_saved_physical_reference():
    pytest.importorskip('mujoco')
    source=Path('outputs/l4-duration-relevel-20261108/pilot-trials.jsonl')
    if not source.is_file():
        pytest.skip('requires the recorded duration pilot trials')
    old=next(r for r in [json.loads(s) for s in source.read_text().splitlines()]
             if r['seed']==5 and r['scenario']=='left_loss' and r['candidate']=='R_left' and r['time_scale']==1.5)
    row=run_trial(5,'left_loss','R_left',retimed_config(1.5,'relevel_only'))
    assert row['trace']==old['trace']
    assert all(row[k]==old[k] for k in STATE_KEYS)
    assert row['success']==old['success'] and row['protection']==old['protection']
    assert row['inference_wait_steps']==0


def test_executed_inference_wait_uses_last_commands_and_counts_in_total():
    pytest.importorskip('mujoco')
    def select(port):
        time.sleep(.06)
        return {'candidate':'R_left','local_relevel_steps':60}
    row=run_trial(0,'left_loss',selector=select,account_selector_latency=True)
    assert not row.get('error')
    waits=[r for r in row['trace'] if r['stage']=='inference_wait']
    assert len(waits)==row['inference_wait_steps']>=2
    assert all(r['phase']=='recovery' for r in waits)
    last_fault=[r for r in row['trace'] if r['phase']=='fault'][-1]
    assert all([r['robot']['action'][i] for i in (6,13)]==[last_fault['robot']['action'][i] for i in (6,13)] for r in waits)
    assert row['inference_wait_seconds']+1e-9>=row['selector_wall_seconds']
    assert row['inference_wait_seconds']-row['selector_wall_seconds']<.05+1e-9
    assert abs(row['total_post_fault_seconds']-row['inference_wait_seconds']-row['skill_seconds']-row['hold_seconds'])<1e-8
    assert row['local_steps']['local_relevel_steps']==60
    expected=sum(not r['evaluator']['holds'][1] for r in row['trace'] if r['phase']=='recovery')
    assert row['protection']['keeper_grasp_lost_steps']==expected
