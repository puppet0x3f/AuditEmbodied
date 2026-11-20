import json

import numpy as np
import pytest

from auditembodied.l4_lift import CANDIDATES, LiftConfig
from auditembodied.l4_selector import ProprioRule
from auditembodied.p0 import (CandidateMLP, fit_mlp, method_metrics, paired_statistics,
                             same_prefix, training_arrays, trial_job)


def outcome(success=True, seconds=1.):
    return {'success': success, 'protection_violation': False, 'recovery_seconds': seconds}


def test_unknown_healthy_branches_are_masked_and_all_safe_actions_are_positive():
    cases = [{'x': np.zeros(7), 'outcomes': {'continue': outcome()}},
             {'x': np.ones(7), 'outcomes': {a: outcome(a != 'continue', i + 1.) for i, a in enumerate(CANDIDATES)}}]
    _, y, mask, _ = training_arrays(cases)
    np.testing.assert_array_equal(mask[0], [1, 0, 0, 0])
    np.testing.assert_array_equal(y[1], [0, 1, 1, 1])


def test_multilabel_candidate_training_and_json_roundtrip():
    cases = [{'x': np.full(7, i), 'outcomes': {a: outcome((j == i) or j == 3, j + 1.)
              for j, a in enumerate(CANDIDATES)}} for i in range(3)]
    model = fit_mlp(cases, ProprioRule(), steps=400)
    scores = model.probabilities(np.stack([c['x'] for c in cases]))
    assert np.all(scores.diagonal() > .9) and np.all(scores[:, 3] > .9)
    restored = CandidateMLP(json.loads(json.dumps(model.values)))
    np.testing.assert_array_equal(scores, restored.probabilities(np.stack([c['x'] for c in cases])))
    with pytest.raises(ValueError):
        restored({'positions': [[0, 0, 0]] * 2, 'fingers': [[.01, -.01]] * 2,
                  'reference_positions': [[0, 0, 0]] * 2, 'future_contact_truth': True})


def test_incomplete_physical_cost_is_unknown_instead_of_zero():
    rows = [{'success': False, 'protection_violation': False, 'candidate': 'abstain', 'failure_reason': 'error'},
            {'success': True, 'protection_violation': False, 'candidate': 'continue', 'total_post_fault_seconds': 1.}]
    report = method_metrics(rows)
    assert report['episodes'] == 2 and report['safe_successes'] == 1
    assert report['mean_total_seconds'] is None and report['complete_cost_observations'] == 1


def test_fast_failure_is_excluded_from_both_safe_cost_comparison():
    rows = []
    for seed in (1, 2):
        for scenario in ('left_loss', 'right_loss'):
            for method, seconds in [('protected', 2.), ('direct', 10.)]:
                failed = seed == 1 and scenario == 'left_loss' and method == 'protected'
                rows.append({'seed': seed, 'scenario': scenario, 'method': method,
                    'success': not failed, 'protection_violation': failed,
                    'total_post_fault_seconds': .1 if failed else seconds,
                    'skill_seconds': seconds, 'recovery_path_m': seconds})
    report = paired_statistics(rows, 'protected', 'direct', draws=100)
    assert report['paired_scenes'] == 4 and report['both_safe_scenes'] == 3
    assert report['safe_success_difference'] == -.25
    assert report['total_post_fault_seconds']['reduction_fraction'] == .8


def test_new_perturbation_comparison_executes_measured_wait_and_complete_hold():
    from dataclasses import asdict, replace
    pytest.importorskip('mujoco')
    config = asdict(replace(LiftConfig(), fault_withdrawal=.06, coupled_height_offset=.05))
    rule = asdict(ProprioRule())
    local = trial_job(((0, 'left_loss', 'protected', None, None), config, rule, {}))
    global_ = trial_job(((0, 'left_loss', 'global', None, None), config, rule, {}))
    assert same_prefix([local, global_])
    for row in (local, global_):
        assert not row.get('error') and row['fault_class_valid']
        assert row['hold_seconds'] >= 1 - 1e-9
        assert 0 <= row['inference_wait_seconds'] - row['selector_wall_seconds'] < .05 + 1e-9
        assert abs(row['total_post_fault_seconds'] - row['skill_seconds'] - row['inference_wait_seconds'] - row['hold_seconds']) < 1e-8
        assert row['contact_anchor_count'] == row['state_edit_count_after_initialization'] == 0
