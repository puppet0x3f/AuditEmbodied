import copy
import json
from pathlib import Path

import pytest

from auditembodied.vlm_mid_recovery import prefix_text,tail_text,forecast
from auditembodied.mid_recovery_train import set_boundary,sampling


def sample():
    path=Path('outputs/vlm-mid-continuation-20261109/engineering-data/train.jsonl')
    if not path.exists():pytest.skip('engineering observation unavailable')
    return json.loads(path.read_text().splitlines()[0])


def test_prompt_uses_only_available_observations():
    row=sample();before=(prefix_text(row),tail_text(row));changed=copy.deepcopy(row)
    changed.update(seed=99999,scenario='other',labels={'local':'A','global':'D'},
        start_valid_score=.1,costs={'local':999,'global':1},actual_global_total_seconds=123,
        boundary={'object_pose':[999]})
    assert before==(prefix_text(changed),tail_text(changed))
    changed['query']['positions'][0][2]+=.01
    assert prefix_text(changed)==prefix_text(row)
    assert tail_text(changed)!=tail_text(row)


def test_native_two_suffix_decision_and_set_boundary():
    torch=pytest.importorskip('torch');ids=list(range(5));tau=.999667229515218
    logits={k:torch.tensor([50.,0,0,0,0]) for k in ('local','global')}
    assert forecast(logits,ids,1.,tau)['action']=='continue'
    logits['local']=torch.tensor([0.,50,0,0,0])
    decision=forecast(logits,ids,1.,tau)
    assert decision['action']=='R_global' and decision['model_admitted']
    logits['global']=torch.tensor([0.,0,50,0,0])
    assert forecast(logits,ids,1.,tau)['action']=='reject'
    values=[torch.zeros(5,requires_grad=True) for _ in range(2)]
    set_boundary(values,ids,['A','A'],1.,tau).backward()
    assert values[0].grad[0]<0 and values[1].grad is None
    values=[torch.tensor([12.,0,0,0,0],requires_grad=True) for _ in range(2)]
    set_boundary(values,ids,['B','C'],1.,tau).backward()
    assert all(v.grad[0]>0 for v in values)


@pytest.mark.parametrize('status',['continue','global','late','stale','reject','short'])
def test_physical_midcourse_boundary_and_cost(status):
    pytest.importorskip('mujoco')
    from auditembodied.l4_lift import run_trial,LiftConfig
    from dataclasses import replace
    calls=[]
    def mid(operation,port,candidate):
        calls.append((operation,port))
        assert not any(k in port for k in ('evaluator','holds','supports','object_pose'))
        if operation=='finish':
            return {'action':{'continue':'continue','reject':'reject'}.get(status,'R_global'),
                'forecast_ready':status!='late','attempt_id':999 if status=='stale' else port['attempt_id']}
    cfg=replace(LiftConfig(),local_relevel_steps=40 if status=='short' else 60)
    r=run_trial(0,'left_loss',config=cfg,selector=lambda p:{'candidate':'R_left'},
        fixed_selector_wait_steps=19,continuation_selector=mid)
    assert not r.get('error'),r.get('error')
    assert abs(r['total_post_fault_seconds']-r['inference_wait_seconds']-r['skill_seconds']-r['hold_seconds'])<1e-8
    if status=='short':
        assert not calls and 'mid_selection' not in r;return
    assert [c[0] for c in calls]==['start','lookahead','finish']
    assert calls[1][1]['relevel_step']==20 and calls[2][1]['relevel_step']==30
    assert abs(calls[2][1]['sim_time']-calls[1][1]['sim_time']-.5)<1e-8
    branch='continue' if status=='continue' else 'reject' if status=='reject' else 'R_global'
    assert r['mid_selection']['selected_branch']==branch
    assert r['mid_selection']['baseline_fallback']==(status in ('late','stale'))
    before=r['continuation_diagnostic']['protection_before']
    assert all(r['protection'][k]>=v for k,v in before.items())
    if branch=='R_global':
        assert r['continuation_diagnostic']['transition']=='defer_rotation'
        assert sum(f['stage']=='local_relevel' for f in r['trace'])==30
