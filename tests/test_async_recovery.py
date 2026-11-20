import pytest

from auditembodied.vlm_mid_recovery import select_at_step


TAU=.999667229515218


def test_writer_can_pipeline_requests_before_first_response():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from auditembodied.duration_family_loop import submit_ordered_request
    written=[];both_written=Event()
    def send(request):
        written.append(request)
        if len(written)==2:both_written.set()
    def receive():
        assert both_written.wait(1.),'writer waited for an acknowledgement before sending next request'
        return 'reply'
    with ThreadPoolExecutor(max_workers=1) as sender,ThreadPoolExecutor(max_workers=1) as reader:
        first=submit_ordered_request(sender,reader,send,receive,'prefill')
        second=submit_ordered_request(sender,reader,send,receive,'tail')
        assert first.result(timeout=2.)=='reply' and second.result(timeout=2.)=='reply'
    assert written==['prefill','tail']


def records(local=.001,glob=.99999,defer=.99999):
    return [{'key':k,'class_probabilities':[v,1-v,0.,0.,0.]} for k,v in zip(('local','global','defer'),(local,glob,defer))]


def test_expired_global30_cannot_be_relabelled_as_global40():
    assert select_at_step(records(defer=.01),TAU,20,40)['action']=='reject'
    d=select_at_step(records(),TAU,20,40)
    assert d['action']=='R_global' and d['selected_key']=='defer' and d['activation_step']==40
    assert not next(r for r in d['records'] if r['key']=='global')['feasible_now']


def test_defer_requires_its_own_admission_and_local_remains_cheapest():
    assert select_at_step(records(),TAU,20,30,True)['action']=='defer'
    assert select_at_step(records(defer=.01),TAU,20,30,True)['action']=='R_global'
    assert select_at_step(records(local=.99999),TAU,20,30,True)['action']=='continue'
    assert select_at_step(records(local=.99999),TAU,20,40)['action']=='continue'


@pytest.mark.parametrize('admitted',[True,False])
def test_physical_pending_retains_only_explicitly_admitted_local_plan(admitted):
    pytest.importorskip('mujoco')
    from auditembodied.l4_lift import run_trial
    def duration(op,port,candidate,prefix):
        if op=='finish':return {'candidate':candidate,'forecast_ready':True,'attempt_id':port['attempt_id'],
            'accepted':True,'local_relevel_steps':60,'model_admitted':admitted,'baseline_fallback':not admitted}
    def mid(op,port,candidate):
        if op=='finish':return {'forecast_ready':False,'attempt_id':30,'forecast_query_step':20,'model_admitted':False}
    row=run_trial(0,'left_loss',selector=lambda p:{'candidate':'R_left','parallel_local':True},
        fixed_selector_wait_steps=19,duration_selector=duration,refresh_duration_observation=True,duration_lookahead_steps=10,
        continuation_selector=mid,continuation_windows=(30,40),continuation_admission_policy='consistent',continuation_runtime='persistent')
    assert not row.get('error'),row.get('error')
    assert sum(f['stage']=='local_relevel' for f in row['trace'])==(60 if admitted else 30)
    assert not any(f['stage'].startswith('global_') for f in row['trace'])
    assert all(w['carried_local_admission']==admitted for w in row['mid_selection']['windows'])
    assert abs(row['total_post_fault_seconds']-row['inference_wait_seconds']-row['skill_seconds']-row['hold_seconds'])<1e-8


def test_physical_second_timeout_keeps_previously_admitted_global40():
    pytest.importorskip('mujoco')
    from auditembodied.l4_lift import run_trial
    def mid(op,port,candidate):
        if op=='finish':
            if port['relevel_step']==30:
                return {**select_at_step(records(),TAU,20,30,True),'forecast_ready':True,'attempt_id':30}
            return {'forecast_ready':False,'attempt_id':30,'forecast_query_step':20,'model_admitted':False}
    row=run_trial(0,'left_loss',selector=lambda p:{'candidate':'R_left','local_relevel_steps':60},fixed_selector_wait_steps=19,
        continuation_selector=mid,continuation_windows=(30,40),continuation_admission_policy='consistent',continuation_runtime='selective')
    assert not row.get('error'),row.get('error')
    assert sum(f['stage']=='local_relevel' for f in row['trace'])==40
    window=row['mid_selection']['windows'][-1]
    assert window['selected_branch']=='R_global' and window['global_admitted'] and window['carried_global_admission']
    assert all(row['protection'][k]>=v for k,v in row['continuation_diagnostic']['protection_before'].items())


def test_pending_also_retains_explicit_scope_admission():
    pytest.importorskip('mujoco')
    from auditembodied.l4_lift import run_trial
    def mid(op,port,candidate):
        if op=='finish':return {'forecast_ready':False,'attempt_id':30,'forecast_query_step':20,'model_admitted':False}
    row=run_trial(0,'left_loss',selector=lambda p:{'candidate':'R_left','local_relevel_steps':60,
        'accepted':True,'threshold':TAU,'plan':{'local_relevel_steps':60},'scored_candidates':[{'joint_A_score':.99999}]},
        fixed_selector_wait_steps=19,continuation_selector=mid,continuation_windows=(30,40),
        continuation_admission_policy='consistent',continuation_runtime='persistent')
    assert not row.get('error'),row.get('error')
    assert sum(f['stage']=='local_relevel' for f in row['trace'])==60
    assert all(w['carried_local_admission'] for w in row['mid_selection']['windows'])
