import copy
import json
from pathlib import Path

import pytest


def test_second_window_prompt_is_causal_and_has_correct_horizon():
    from auditembodied.vlm_mid_recovery import prefix_text,tail_text
    p=Path('outputs/vlm-two-window-rgbd-20261109/engineering-data-v2/train.jsonl')
    if not p.exists():pytest.skip('engineering data unavailable')
    row=next(json.loads(x) for x in p.read_text().splitlines() if json.loads(x)['decision_step']==40)
    text=tail_text(row)
    assert '"observed_step":30' in text and '"decision_step":40' in text
    assert '"local_remaining_s_at_decision":2.4' in text
    changed=copy.deepcopy(row);changed.update(labels={'local':'D','global':'A'},all_labels={'future':'secret'},seed=999)
    changed['future_wrench']=[999]
    assert prefix_text(changed)==prefix_text(row) and tail_text(changed)==text
    changed['query']['wrench'][0][2]+=1
    assert tail_text(changed)!=text


@pytest.mark.parametrize('first_action',['continue','R_global'])
def test_two_windows_execute_only_reached_decisions(first_action):
    pytest.importorskip('mujoco')
    from dataclasses import replace
    from auditembodied.l4_lift import run_trial,LiftConfig
    calls=[]
    def mid(op,port,candidate):
        calls.append((op,port))
        assert not any(k in port for k in ('evaluator','object_pose','holds','supports'))
        assert 'wrench' in port and 'sensor_history' in port
        if op=='finish':return {'action':first_action if port['relevel_step']==30 else 'R_global',
            'forecast_ready':True,'attempt_id':port['attempt_id'],'baseline_fallback':False}
    row=run_trial(0,'left_loss',config=replace(LiftConfig(),local_relevel_steps=60),
        selector=lambda p:{'candidate':'R_left'},fixed_selector_wait_steps=19,
        continuation_selector=mid,continuation_windows=(30,40),continuation_features=True)
    assert not row.get('error'),row.get('error')
    expected=[('start',0),('lookahead',20),('finish',30)]
    if first_action=='continue':expected.extend([('lookahead',30),('finish',40)])
    assert [(op,p['relevel_step']) for op,p in calls]==expected
    assert sum(f['stage']=='local_relevel' for f in row['trace'])==(40 if first_action=='continue' else 30)
    assert all(w['continuation_matches_forecast'] for w in row['mid_selection']['windows'])
    assert abs(row['total_post_fault_seconds']-row['inference_wait_seconds']-row['skill_seconds']-row['hold_seconds'])<1e-8
    before=row['continuation_diagnostic']['protection_before']
    assert all(row['protection'][k]>=v for k,v in before.items())


@pytest.mark.parametrize('windows',[(30,),(30,40)])
def test_deferred_global_is_executable_with_one_query_and_revisable_with_two(windows):
    pytest.importorskip('mujoco')
    from dataclasses import replace
    from auditembodied.l4_lift import run_trial,LiftConfig
    calls=[]
    def mid(op,port,candidate):
        calls.append((op,port['relevel_step']))
        if op=='finish':return {'action':'defer' if port['relevel_step']==30 else 'continue',
            'forecast_ready':True,'attempt_id':port['attempt_id'],'baseline_fallback':False}
    row=run_trial(0,'left_loss',config=replace(LiftConfig(),local_relevel_steps=60),
        selector=lambda p:{'candidate':'R_left'},fixed_selector_wait_steps=19,
        continuation_selector=mid,continuation_windows=windows,continuation_features=True)
    assert not row.get('error'),row.get('error')
    assert sum(f['stage']=='local_relevel' for f in row['trace'])==(40 if len(windows)==1 else 60)
    assert ('finish',40) in calls if len(windows)==2 else ('finish',40) not in calls
    assert row['candidate']==('R_global' if len(windows)==1 else 'R_left')
    assert abs(row['total_post_fault_seconds']-row['inference_wait_seconds']-row['skill_seconds']-row['hold_seconds'])<1e-8


def test_later_global_can_be_admitted_when_both_immediate_options_are_unsafe():
    torch=pytest.importorskip('torch')
    from auditembodied.vlm_mid_recovery import forecast
    values={'local':torch.tensor([0.,0,30.,0,0]),'global':torch.tensor([0.,30.,0,0,0]),'defer':torch.tensor([30.,0,0,0,0])}
    answer=forecast(values,list(range(5)),1.,.999667229515218)
    assert answer['action']=='defer' and answer['model_admitted']
