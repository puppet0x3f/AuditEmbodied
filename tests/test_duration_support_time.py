import copy

import pytest

from auditembodied.vlm_duration_lora import outcome_updates, task_label
from auditembodied.vlm_duration_support_time import support_time_target
from auditembodied.vlm_duration_temporal import input_spec


def rollout(loss_step=None, loss_stage=None):
    trace = []
    t = 1.
    for stage, count in [('duration_budget_wait', 2), ('local_relevel', 30),
                         ('local_relevel_settle', 8), ('final_stable_hold', 20)]:
        for j in range(1, count + 1):
            t += .05
            lost = stage == 'local_relevel' and j == loss_step or stage == loss_stage
            trace.append({'time': t, 'stage': stage, 'evaluator': {'holds': [True, not lost]}})
    return {'candidate': 'R_left', 'stage_capture': {'h2': {'state': {'time': 1.}}},
        'local_steps': {'local_relevel_steps': 30}, 'trace': trace, 'success': False}


@pytest.mark.parametrize(('step', 'code'), [(7, 'C'), (8, 'D'), (15, 'D'), (16, 'E'), (22, 'E'), (23, 'F')])
def test_first_loss_quarters_follow_fraction_of_planned_steps(step, code):
    target = support_time_target(rollout(loss_step=step))
    assert target['label'] == code and target['first_loss_relevel_step'] == step
    assert target['first_loss_seconds_after_observation'] == pytest.approx(.1 + step / 20)
    assert target['event_observed'] and not target['right_censored']


def test_wait_and_verification_losses_and_finite_censoring():
    assert support_time_target(rollout(loss_stage='duration_budget_wait'))['label'] == 'B'
    assert support_time_target(rollout(loss_stage='final_stable_hold'))['label'] == 'G'
    target = support_time_target(rollout())
    assert target['label'] == 'A' and target['right_censored']
    assert not target['event_observed'] and target['first_loss_seconds_after_observation'] is None
    assert target['observed_horizon_seconds'] == pytest.approx(3.)


def test_future_time_target_does_not_change_main_inputs_or_sampling():
    robot = {'positions': [[0, 0, 1], [0, 1, 1]], 'fingers': [[.01, -.01], [.02, -.02]],
        'reference_positions': [[0, 0, 1], [0, 1, 1]]}
    row = {'image': 'h1.npz', 'after_prefix_image': 'h2.npz', 'robot': robot,
        'after_prefix_robot': copy.deepcopy(robot), 'late_history': [], 'seed': 40, 'label': 'B',
        'plan': {'kind': 'local', 'repair_arm': 'left', 'keeper_arm': 'right', 'local_relevel_steps': 40}}
    main = input_spec(row, 'outcome', 'stage_late')
    start = input_spec(row, 'start', 'stage_late')
    original_sampling = outcome_updates([row], 8, 20261108)
    row['support_time'] = {'label': 'PRIVATE_TIME_LABEL', 'first_loss_seconds_after_observation': 'PRIVATE_FUTURE_TRUTH'}
    assert input_spec(row, 'outcome', 'stage_late') == main
    assert input_spec(row, 'start', 'stage_late') == start
    assert outcome_updates([row], 8, 20261108) == original_sampling
    paths, prompt = input_spec(row, 'support_time', 'stage_late')
    assert paths == ['h1.npz', 'h2.npz']
    assert 'PRIVATE_' not in prompt and 'WHEN' in prompt
    assert 'task completed, process guards passed' not in prompt
    assert task_label(row, 'support_time') == 'PRIVATE_TIME_LABEL'
    assert task_label(row, 'outcome') == 'B'
