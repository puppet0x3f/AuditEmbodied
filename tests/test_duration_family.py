import copy

import numpy as np
import pytest

from auditembodied.vlm_duration_family import compact_state,prefix_text,tail_text
from auditembodied.vlm_duration_lora import family_set_admission_loss
from auditembodied.vlm_duration_temporal import family_updates
from test_duration_joint import example


def family_row():
    r=example()
    r.update(entry_seconds_after_h0=.95,lookahead_robot=copy.deepcopy(r['after_prefix_robot']),
        lookahead_previous_robot=copy.deepcopy(r['robot']),lookahead_delta_seconds=.2,
        lookahead_stage='local_arc_close',remaining_close_steps=10,lookahead_seconds_after_h0=3.1,
        prefix_observation_wait_steps=0)
    return r


def test_compact_state_preserves_roles_and_is_translation_invariant():
    r=family_row();f=compact_state(r['lookahead_robot'],r['robot'],'left',.2)
    mirrored=copy.deepcopy(r)
    for field in ('lookahead_robot','robot'):
        for key in ('positions','fingers','reference_positions'):mirrored[field][key].reverse()
    assert compact_state(mirrored['lookahead_robot'],mirrored['robot'],'right',.2)==f
    for field in ('lookahead_robot','robot'):
        for key in ('positions','reference_positions'):
            mirrored[field][key]=(np.asarray(mirrored[field][key])+[2.,-3.,4.]).tolist()
    assert compact_state(mirrored['lookahead_robot'],mirrored['robot'],'right',.2)==f


def test_prefix_cannot_read_lookahead_or_future_truth_and_tail_is_label_free():
    r=family_row();p=prefix_text(r);t=tail_text(r)
    r.update(label='PRIVATE_LABEL',seed='PRIVATE_SEED',future_trace='PRIVATE_TRUTH')
    assert prefix_text(r)==p and tail_text(r)==t
    r['after_prefix_robot']={'PRIVATE_FUTURE':True};r['after_prefix_image']='PRIVATE_IMAGE'
    assert prefix_text(r)==p and tail_text(r)==t
    r['lookahead_robot']['positions'][0][0]+=.01
    assert prefix_text(r)==p and tail_text(r)!=t
    assert all(s not in p+t for s in ('PRIVATE_','absolute_positions'))


def test_set_boundary_rejects_all_negatives_and_preserves_one_cheap_safe_option():
    torch=pytest.importorskip('torch')
    values=[torch.tensor([10.,0.,0.,0.],requires_grad=True) for _ in range(3)]
    plans=[{'local_relevel_steps':n} for n in (30,40,60)]
    loss=family_set_admission_loss(values,list(range(4)),['B','A','A'],[1.]*3,plans,.99)
    loss.backward()
    assert values[0].grad[0]>0 and values[1].grad[0]<0
    assert values[2].grad is None
    values=[torch.tensor([10.,0.,0.,0.],requires_grad=True) for _ in range(3)]
    family_set_admission_loss(values,list(range(4)),['D','D','C'],[1.]*3,plans,.99).backward()
    assert all(v.grad[0]>0 for v in values)


def test_safe_set_stratification_keeps_fixed_budget_and_old_new_balance():
    rows=[]
    for source in ('old','new'):
        for pattern,repeats in (('AAA',5),('BAA',2),('BBA',1),('DDC',1)):
            for j in range(repeats):
                for n,label in zip((30,40,60),pattern):
                    r=example(n);r.update(image=f'{source}-{pattern}-{j}',label=label,replay_source=source);rows.append(r)
    updates,report=family_updates(rows,512,123,balanced_replay=True,stratify_safe_sets=True)
    assert report['presentations']==1024 and report['source_presentations']=={'old':512,'new':512}
    assert report['family_safe_set_presentations']=={'111':64,'011':64,'001':64,'000':64}
    assert all(len(u['indices'])==3 if i%2==0 else len(u['indices'])==1 for i,u in enumerate(updates))


@pytest.mark.parametrize('status',['late','stale','accepted','rejected'])
def test_physical_lookahead_applies_only_timely_same_attempt_and_never_waits(status):
    pytest.importorskip('mujoco')
    from auditembodied.l4_lift import run_trial
    requests=[]
    def callback(operation,port,candidate,prefix):
        if operation=='lookahead':requests.append(port)
        if operation=='finish':
            return {'candidate':candidate,'accepted':status!='rejected','local_relevel_steps':30,
                'forecast_ready':status!='late','attempt_id':999 if status=='stale' else port['attempt_id'],
                'pipeline_wall_seconds':.1}
    r=run_trial(0,'left_loss',selector=lambda p:{'candidate':'R_left','parallel_local':True},
        duration_selector=callback,fixed_selector_wait_steps=19,refresh_duration_observation=True,duration_lookahead_steps=10)
    assert not r.get('error'),r.get('error')
    assert requests and all(p['remaining_close_steps']==10 for p in requests)
    assert all(not any(k in p for k in ('evaluator','holds','supports','object_pose')) for p in requests)
    assert not any(f['stage'] in ('duration_budget_wait','duration_deadline_wait') for f in r['trace'])
    expected=None if status=='rejected' else 30 if status=='accepted' else 60
    assert r['local_steps']['local_relevel_steps']==expected
    assert abs(r['inference_wait_seconds']-.95)<1e-8
    assert abs(r['total_post_fault_seconds']-r['inference_wait_seconds']-r['skill_seconds']-r['hold_seconds'])<1e-8
    if status in ('late','stale'):assert not r.get('abstained') and r['duration_selection']['baseline_fallback']


def test_async_launch_shares_existing_period_and_forced60_preserves_physical_trace():
    pytest.importorskip('mujoco')
    from auditembodied.l4_lift import run_trial
    baseline=run_trial(0,'left_loss',selector=lambda p:{'candidate':'R_left','local_relevel_steps':60},
        fixed_selector_wait_steps=19)
    def callback(operation,port,candidate,prefix):
        if operation=='finish':
            return {'candidate':candidate,'accepted':True,'local_relevel_steps':60,
                'forecast_ready':True,'attempt_id':port['attempt_id'],'pipeline_wall_seconds':.1}
    actual=run_trial(0,'left_loss',selector=lambda p:{'candidate':'R_left','parallel_local':True},
        duration_selector=callback,fixed_selector_wait_steps=19,refresh_duration_observation=True,
        duration_lookahead_steps=10,account_prefix_observation_latency=True)
    assert not actual.get('error'),actual.get('error')
    assert actual['duration_selection']['launch_inside_control_period']
    assert actual['duration_selection']['prefix_observation_wait_steps']==0
    assert actual['trace']==baseline['trace']
    assert actual['total_post_fault_seconds']==baseline['total_post_fault_seconds']


def test_real_control_overrun_is_physically_charged_and_invalidates_forecast():
    pytest.importorskip('mujoco')
    import time
    from auditembodied.l4_lift import run_trial
    def callback(operation,port,candidate,prefix):
        if operation=='lookahead':time.sleep(.06)
        if operation=='finish':
            return {'candidate':candidate,'accepted':True,'local_relevel_steps':30,
                'forecast_ready':True,'attempt_id':port['attempt_id'],'pipeline_wall_seconds':.1}
    r=run_trial(0,'left_loss',selector=lambda p:{'candidate':'R_left','parallel_local':True},
        duration_selector=callback,fixed_selector_wait_steps=19,refresh_duration_observation=True,
        duration_lookahead_steps=10,account_prefix_observation_latency=True,pace_recovery=True)
    assert not r.get('error'),r.get('error')
    assert r['controller_compute_wait_steps']>=1
    assert any(f['stage']=='controller_compute_wait' for f in r['trace'])
    assert not r['duration_selection']['continuation_matches_forecast']
    assert r['duration_selection']['baseline_fallback'] and r['local_steps']['local_relevel_steps']==60
    assert abs(r['inference_wait_seconds']-(19+r['controller_compute_wait_steps'])/20)<1e-8
    assert abs(r['total_post_fault_seconds']-r['inference_wait_seconds']-r['skill_seconds']-r['hold_seconds'])<1e-8
