import json

import numpy as np
import pytest

from auditembodied.duration_temporal_data import validate
from auditembodied.duration_wait_matrix import collect_state
from auditembodied.vlm_duration_data import plan
from auditembodied.vlm_duration_lora import task_prompt_for
from auditembodied.vlm_duration_temporal import input_spec,state_updates,temporal_features,risk_quota_updates
from auditembodied.vlm_duration_policy import routed_input


def sample(wait,n,label):
    robot={'positions':[[0.,0.,1.],[0.,1.,1.]],'fingers':[[.04,-.04],[.01,-.01]],
           'reference_positions':[[0.,0.,1.],[0.,1.,1.]]}
    return {'seed':9,'scenario':'PRIVATE_FAULT','before_image':'before.npz','before_robot':robot,
        'image':f'after-{wait}.npz','robot':robot,'wait_steps':wait,
        'plan':plan({'candidate':'R_left','local_steps':{'local_relevel_steps':n}}),'label':label}


def test_temporal_inputs_do_not_change_frozen_start_question():
    row=sample(20,40,'B')
    paths,text=input_spec(row,'outcome','paired')
    assert paths==['before.npz','after-20.npz'] and 'Images 1-4' in text and 'PRIVATE_FAULT' not in text
    paths,text=input_spec(row,'start','paired')
    assert paths==['after-20.npz'] and text==task_prompt_for(row,'start')
    assert temporal_features(row)['keeper_displacement_m']==[0.,0.,0.]


def test_frozen_route_and_start_inputs_do_not_depend_on_new_adapter_context():
    row=sample(20,40,'B');context={**row,'post_wait':True}
    adapter,paths,text=routed_input(row,'start',context)
    assert adapter=='start' and paths==[row['image']] and text==task_prompt_for(row,'start')
    adapter,paths,text=routed_input(row,'outcome',{**context,'post_wait':False})
    assert adapter=='default' and paths==[row['image']] and text==task_prompt_for(row,'outcome')
    adapter,paths,text=routed_input(row,'outcome',context)
    assert adapter=='post_wait' and len(paths)==2 and 'Images 1-4' in text


def test_intervention_pairs_have_their_respective_physical_matching_rules():
    rows=[sample(0,n,'A') for n in (30,40,60)]+[sample(20,n,'B' if n!=60 else 'A') for n in (30,40,60)]
    updates,design=state_updates(rows,512,17)
    assert len(updates)==512 and design['all_valid_examples_presented']
    assert design['presentations']==640 and design['update_counts']['state_coverage']==384
    for u in updates:
        if len(u['indices'])==1:continue
        a,b=[rows[i] for i in u['indices']]
        if u['kind']=='wait_contrast':
            assert a['before_image']==b['before_image'] and a['plan']==b['plan']
            assert a['wait_steps']!=b['wait_steps'] and a['label']=='A' and b['label']!='A'
        elif u['kind']=='action_contrast':
            assert a['image']==b['image'] and a['robot']==b['robot']
            assert a['plan']['local_relevel_steps']!=b['plan']['local_relevel_steps']
        else:assert a['label']==b['label']=='A'


def test_before_and_after_capture_preserves_zero_wait_physics(tmp_path):
    pytest.importorskip('mujoco')
    rows=collect_state(('train',0,'left_loss',str(tmp_path),0,True))
    report=validate(rows)
    assert report['valid'],report['errors']
    c=rows[0]['capture']
    assert c['before_state']==c['state'] and c['before_robot']==c['robot']
    with np.load(c['before_image']) as a,np.load(c['observation_file']) as b:
        assert np.array_equal(a['rgb'],b['rgb'])


def test_strict_timing_cycle_covers_every_flip_without_changing_other_updates():
    from collections import Counter
    rows=[sample(wait,n,'A' if n==60 else 'B') for wait in (0,10,20) for n in (30,40,60)]
    rows += [sample(30,n,'A') for n in (30,40,60)]
    original,_=state_updates(rows,128,20261108)
    revised,design=state_updates(rows,128,20261108,strict_timing=True)
    assert design['presentations']==160 and design['all_valid_examples_presented']
    assert len(revised)==128 and design['update_counts']['strict_timing']==11
    covered=Counter()
    for before,after in zip(original,revised):
        if before['kind']!='action_contrast':
            assert before==after
            continue
        assert after['kind']=='strict_timing'
        a,b=[rows[i] for i in after['indices']]
        assert a['label']=='A' and b['label']=='B'
        assert a['plan']['local_relevel_steps']==60 and b['plan']['local_relevel_steps']==40
        assert a['image']==b['image'] and a['robot']==b['robot'] and a['wait_steps']==b['wait_steps']
        covered[a['wait_steps']]+=1
    assert set(covered)=={0,10,20} and sorted(covered.values())==[3,4,4]


def test_risk_quotas_preserve_pairs_budget_and_invalid_start_exclusion():
    from collections import Counter
    rows=[sample(wait,n,'A' if n==60 else 'B') for wait in (0,10,20) for n in (30,40,60)]
    rows += [sample(30,n,'A') for n in (30,40,60)]
    rows += [sample(40,30,'C'),sample(40,40,'D'),sample(40,60,'C'),sample(50,40,'E')]
    previous=[{**r,'class_probabilities':[.1,.9,0,0,0]} for r in rows]
    previous[12]['class_probabilities']=[.9999,.0001,0,0,0]
    original,_=state_updates(rows,128,17,strict_timing=True)
    revised,design=risk_quota_updates(rows,previous,original,17,.9996)
    assert len(revised)==128 and design['presentations']==160 and design['all_valid_examples_presented']
    assert design['risk_quota']['hard_indices']==[12] and design['risk_quota']['all_quotas_met']
    counts=Counter(i for u in revised for i in u['indices'])
    assert counts[12]>=8 and counts[15]==0
    assert all(counts[i]>=(1 if r['label']=='A' else 3) for i,r in enumerate(rows) if r['label']!='E')
    for a,b in zip(original,revised):
        if len(a['indices'])==2:assert a==b
        else:assert len(b['indices'])==1
    with pytest.raises(ValueError,match='fixed single-example budget'):
        risk_quota_updates(rows,previous,original[:4],17,.9996)
