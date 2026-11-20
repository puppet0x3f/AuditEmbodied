import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from auditembodied.admissible_mlp import StartGatedMLP, local_precondition, start_mask, trial_job
from auditembodied.l4_lift import LiftConfig
from auditembodied.l4_selector import ProprioRule
from auditembodied.p0 import CandidateMLP, same_prefix


def model_values(scores):
    return {'rule': asdict(ProprioRule()), 'mean': [0.] * 7, 'scale': [1.] * 7,
            'w1': [[0.] for _ in range(7)], 'b1': [0.], 'w2': [[0.] * 4],
            'b2': np.log(np.array(scores) / (1 - np.array(scores))).tolist(),
            'threshold': .5, 'action_cost_seconds': [0., 2.7, 2.7, 11.6]}


def port(left=True, right=False):
    return {'positions': [[0., 0., 1.], [.5, 0., 1.]],
            'reference_positions': [[0., 0., 1.], [.5, 0., 1.]],
            'fingers': [[.012, -.012] if loaded else [.001, -.001] for loaded in (left, right)]}


def test_wrong_side_is_masked_without_changing_probabilities_or_threshold():
    values = model_values([.1, .99, .8, .99])
    before = copy.deepcopy(values)
    decision = StartGatedMLP(values)(port())
    assert CandidateMLP(values)(port())['candidate'] == 'R_left'
    assert decision['candidate'] == 'R_right'
    assert decision['start_admissible'] == [False, False, True, True]
    assert decision['safety_scores'] == CandidateMLP(values)(port())['safety_scores']
    assert values == before and decision['threshold'] == .5


def test_gate_does_not_force_a_low_scoring_legal_candidate():
    values = model_values([.1, .99, .4, .9])
    assert StartGatedMLP(values)(port())['candidate'] == 'R_global'
    values = model_values([.1, .99, .4, .3])
    assert StartGatedMLP(values)(port())['candidate'] == 'abstain'


def test_gate_is_role_symmetric_and_does_not_copy_fragility_or_height_rules():
    np.testing.assert_array_equal(start_mask([0, 0, 1, 0, .99, .99, 10]), [False, False, True, True])
    np.testing.assert_array_equal(start_mask([0, 0, 0, 1, 0, 0, 0]), [False, True, False, True])
    np.testing.assert_array_equal(start_mask([0, 0, 1, 1, .99, .99, 10]), [True] * 4)


def test_truth_fields_are_rejected_and_preexisting_absence_is_not_new_loss():
    values = model_values([.1, .9, .9, .99])
    with pytest.raises(ValueError):
        StartGatedMLP(values)({**port(), 'keeper_contact_truth': True})
    row = {'candidate': 'R_left', 'post_fault': {'holds': [True, False]},
           'trace': [{'phase': 'recovery', 'stage': 'inference_wait', 'evaluator': {'holds': [True, False]}}]}
    result = local_precondition(row)
    assert not result['keeper_present_at_start'] and not result['keeper_interrupted_after_valid_start']


def test_frozen_real_model_runs_with_shared_prefix_and_charged_wait():
    pytest.importorskip('mujoco')
    root = Path('outputs/p0-20261109')
    protocol_path = root / 'protocol.json'
    model_path = root / 'direct-selected.json'
    if not protocol_path.is_file() or not model_path.is_file():
        pytest.skip('requires the original P0 protocol and frozen MLP checkpoint')
    protocol = json.loads(protocol_path.read_text())
    values = json.loads(model_path.read_text())['mlp']
    before = copy.deepcopy(values)
    config = protocol['test_config']
    gated = trial_job((0, 'left_loss', 'gated_direct', config, protocol['rule'], values))
    protected = trial_job((0, 'left_loss', 'protected', config, protocol['rule'], values))
    assert same_prefix([gated, protected])
    for row in (gated, protected):
        assert not row.get('error') and row['fault_class_valid']
        assert abs(row['total_post_fault_seconds'] - row['skill_seconds'] - row['inference_wait_seconds'] - row['hold_seconds']) < 1e-8
        assert row['inference_wait_seconds'] >= row['selector_wall_seconds']
    assert values == before
    assert gated['selection']['decision']['threshold'] == values['threshold']
