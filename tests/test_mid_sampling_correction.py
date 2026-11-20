import pytest

from auditembodied.mid_recovery_train import sampling,sampling_correction,training_loss


def test_hierarchical_sampling_weights_restore_family_probability():
    # Different seed sizes distinguish correct row probabilities from a mere
    # A/non-A class-prior correction; both windows keep their original share.
    rows=[]
    for window in (30,40):
        for seed,labels,n in ((1,{'local':'A','global':'A'},5),
                              (2,{'local':'A','global':'A'},1),
                              (3,{'local':'B','global':'A'},2)):
            rows.extend({'seed':seed,'decision_step':window,'labels':labels,
                         'plan':{'repair_arm':'left'}} for _ in range(n))
    schedule,_=sampling(rows,64,20261109)
    before=list(schedule);weights=sampling_correction(rows,schedule)
    assert schedule==before
    assert sum(r['sampling_probability'] for r in weights)==pytest.approx(1.)
    assert sum(r['target_probability'] for r in weights)==pytest.approx(1.)
    for r in weights:
        assert r['sampling_probability']*r['weight']==pytest.approx(1/len(rows))
    # Five rows of seed1 each have lower proposal probability than seed2's
    # single row, although they belong to the same label/role stratum.
    assert weights[0]['weight']==pytest.approx(5*weights[5]['weight'])


def test_sampling_correction_requires_support():
    rows=[{'seed':i,'labels':{'local':c,'global':'A'}} for i,c in enumerate(('A','B'))]
    with pytest.raises(ValueError,match='support'):
        sampling_correction(rows,[0])


def test_importance_weight_does_not_scale_pair_gradient():
    torch=pytest.importorskip('torch')
    terms=[torch.tensor(2.,requires_grad=True) for _ in range(5)]
    training_loss(*terms,weight=3.).backward()
    assert [float(x.grad) for x in terms]==pytest.approx([3.,1.5,1.5,.1,.3])
    legacy=training_loss(*[x.detach() for x in terms])
    assert float(legacy)==pytest.approx(4.4)


def test_classification_only_correction_preserves_action_constraints():
    torch=pytest.importorskip('torch')
    terms=[torch.tensor(2.,requires_grad=True) for _ in range(5)]
    training_loss(*terms,weight=3.,correction_scope='classification').backward()
    assert [float(x.grad) for x in terms]==pytest.approx([3.,1.5,1.5,.1,.1])
