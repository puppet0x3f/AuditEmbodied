from dataclasses import replace

import pytest


def test_replayed_prefix_can_use_a_live_midcourse_choice():
    pytest.importorskip('mujoco')
    from auditembodied.duration_continuation import run_continuation
    from auditembodied.l4_lift import LiftConfig,run_trial
    source=run_trial(0,'left_loss',config=replace(LiftConfig(),local_relevel_steps=60),
        selector=lambda _: {'candidate':'R_left'},fixed_selector_wait_steps=19)
    assert not source.get('error')
    calls=[]
    def choose(operation,port,candidate):
        calls.append(operation)
        if operation=='finish':
            return {'action':'continue','forecast_ready':True,'attempt_id':port['attempt_id'],
                'model_admitted':True,'baseline_fallback':False}
    row=run_continuation((source,'model'),transition='defer_rotation',observer=choose)
    assert not row['diagnostic_errors'],row['diagnostic_errors']
    assert calls==['start','lookahead','finish']
    assert row['model_inference_rerun'] and row['mid_selection']['selected_branch']=='continue'
    assert row['trace']==source['trace']


def test_shared_search_failure_exit_works_without_replay_or_model():
    pytest.importorskip('mujoco')
    from auditembodied.l4_lift import LiftConfig,run_trial
    row=run_trial(28005,'right_loss',config=replace(LiftConfig(),local_relevel_steps=60),
        selector=lambda _: {'candidate':'R_right'},fixed_selector_wait_steps=21,
        search_exhaustion_recovery=True)
    assert not row.get('error'),row.get('error')
    assert row['skill_failure']['reason']=='search_exhausted'
    assert row['skill_failure']['attempts']==3
    assert not any(f['stage']=='local_relevel' for f in row['trace'])
    assert row['candidate']=='R_global' and row['source_candidate']=='R_right'
    boundary=row['continuation_diagnostic']
    assert boundary['boundary']=='search_exhausted' and boundary['transition']=='defer_rotation'
    before=boundary['protection_before']
    assert all(row['protection'][k]>=v for k,v in before.items())
    assert abs(row['total_post_fault_seconds']-row['inference_wait_seconds']-row['skill_seconds']-row['hold_seconds'])<1e-8
