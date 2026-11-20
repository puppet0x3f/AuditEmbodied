import pytest

from auditembodied.vlm_mid_recovery import select_forecast


def test_no_admission_never_invents_global_in_consistent_policy():
    rows=[{'key':k,'class_probabilities':[.02,.96,.01,.01,0.]} for k in ('local','global','defer')]
    answer=select_forecast(rows,.999667229515218)
    assert answer['action']=='reject' and answer['no_admissible_plan']
    assert not answer['explicit_rejection'] and not answer['baseline_fallback']
    assert select_forecast(rows,.999667229515218,'legacy')['action']=='R_global'
    rows[2]['class_probabilities']=[.99999,0.,0.,0.,.00001]
    assert select_forecast(rows,.999667229515218)['action']=='defer'


@pytest.mark.parametrize('kind',['late','stale','unadmitted'])
def test_consistent_physical_exit_preserves_prefix_and_cost(kind):
    pytest.importorskip('mujoco')
    from dataclasses import replace
    from auditembodied.l4_lift import run_trial,LiftConfig
    def mid(op,port,candidate):
        if op=='finish':return {'action':'R_global','forecast_ready':kind!='late',
            'attempt_id':999 if kind=='stale' else port['attempt_id'],'model_admitted':False}
    r=run_trial(0,'left_loss',config=replace(LiftConfig(),local_relevel_steps=60),
        selector=lambda p:{'candidate':'R_left'},fixed_selector_wait_steps=19,
        continuation_selector=mid,continuation_admission_policy='consistent')
    assert not r.get('error'),r.get('error')
    assert r['mid_selection']['selected_branch']=='reject' and r['abstained']
    assert sum(f['stage']=='local_relevel' for f in r['trace'])==30
    assert not any(f['stage'].startswith('global_') for f in r['trace'])
    assert abs(r['total_post_fault_seconds']-r['inference_wait_seconds']-r['skill_seconds']-r['hold_seconds'])<1e-8
    assert all(r['protection'][k]>=v for k,v in r['continuation_diagnostic']['protection_before'].items())
