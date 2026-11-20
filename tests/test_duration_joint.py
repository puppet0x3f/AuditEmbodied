import copy
import math

import numpy as np
import pytest

from auditembodied.vlm_duration_lora import conditional_process_loss, family_pair_loss, joint_marginal_metrics, task_prompt_for
from auditembodied.vlm_duration_policy import routed_input
from auditembodied.vlm_duration_temporal import family_updates, input_spec, joint_action_features


def example(n=40, wait=66):
    previous = {'positions': [[0., 0., .9], [0., .2, 1.]],
        'fingers': [[.01, -.01], [.02, -.02]],
        'reference_positions': [[0., 0., 1.], [0., .2, 1.]]}
    current = copy.deepcopy(previous)
    current['positions'][0][0] = .02
    return {'image': 'h1.npz', 'after_prefix_image': 'h2.npz', 'before_image': 'h0.npz',
        'robot': previous, 'after_prefix_robot': current, 'before_robot': copy.deepcopy(previous),
        'late_history': [], 'early_history': [], 'label': 'A', 'seed': 'PRIVATE_SEED',
        'plan': {'kind': 'local', 'repair_arm': 'left', 'keeper_arm': 'right',
            'local_relevel_steps': n, 'post_prefix_wait_steps': wait}}


def test_process_marginal_preserves_completion_independent_semantics():
    torch = pytest.importorskip('torch')
    # A currently dominates: it is correct for process-safe C, wrong for completed-unsafe B.
    v = torch.tensor([10., -5., 0., -5., -9.], requires_grad=True)
    ids = list(range(5))
    assert conditional_process_loss(v, ids, 'C').item() < 1e-4
    assert conditional_process_loss(v, ids, 'B').item() > 10
    assert conditional_process_loss(v, ids, 'B').item() == conditional_process_loss(v, ids, 'D').item()
    conditional_process_loss(v, ids, 'B').backward()
    assert v.grad[0] > 0 and v.grad[1] < 0 and v.grad[3] < 0
    with pytest.raises(ValueError):
        conditional_process_loss(v, ids, 'E')


def test_family_ranking_uses_actual_flips_and_does_not_order_safe_controls():
    torch = pytest.importorskip('torch')
    ids = list(range(5))
    logits = lambda letter: torch.tensor([6. if c == letter else -3. for c in 'ABCDE'], requires_grad=True)
    loss, counts = family_pair_loss([logits('A') for _ in range(3)], ids, ['A'] * 3)
    assert loss.item() == 0 and counts == {'joint': 0, 'process': 0}
    good, counts = family_pair_loss([logits('B'), logits('A'), logits('A')], ids, ['B', 'A', 'A'])
    wrong, _ = family_pair_loss([logits('A'), logits('B'), logits('B')], ids, ['B', 'A', 'A'])
    assert counts == {'joint': 2, 'process': 2} and wrong.item() > good.item()
    _, counts = family_pair_loss([logits('D'), logits('D'), logits('C')], ids, ['D', 'D', 'C'])
    assert counts == {'joint': 0, 'process': 2}


def test_nominal_features_are_role_shared_and_use_current_pose_only():
    row = example()
    f = joint_action_features(row)
    expected_angle = math.atan2(.1, np.linalg.norm([.02, -.2]))
    assert f['planned_relevel_seconds'] == 2.
    assert f['planned_post_prefix_hold_seconds'] == 3.3
    assert f['nominal_current_relevel_angle_rad'] == pytest.approx(expected_angle)
    slower = copy.deepcopy(row)
    slower['plan']['local_relevel_steps'] = 60
    assert f['nominal_angular_speed_rad_per_second'] / joint_action_features(slower)['nominal_angular_speed_rad_per_second'] == pytest.approx(1.5)
    mirrored = copy.deepcopy(row)
    for field in ('robot', 'after_prefix_robot'):
        for key in ('positions', 'fingers', 'reference_positions'):
            mirrored[field][key].reverse()
    mirrored['plan'].update(repair_arm='right', keeper_arm='left')
    assert joint_action_features(mirrored) == f
    row.update(future_trace='PRIVATE_FUTURE', support_time={'label': 'PRIVATE_TIME'}, label='PRIVATE_LABEL')
    assert joint_action_features(row) == f
    prompt = input_spec(row, 'outcome', 'stage_joint')[1]
    assert 'PRIVATE_' not in prompt and 'not future measured coordinates' in prompt


def test_family_budget_keeps_wait_and_observations_fixed():
    rows = [example(n, wait) for wait in (0, 66) for n in (30, 40, 60)]
    updates, report = family_updates(rows, 128, 20261108)
    assert report['duration_families'] == 2 and report['presentations'] == 256
    assert report['update_counts'] == {'duration_family': 64, 'uniform_single': 64}
    assert report['all_valid_examples_presented']
    for update in updates:
        if update['kind'] == 'duration_family':
            group = [rows[i] for i in update['indices']]
            assert [r['plan']['local_relevel_steps'] for r in group] == [30, 40, 60]
            assert len({r['plan']['post_prefix_wait_steps'] for r in group}) == 1
    with pytest.raises(ValueError):
        family_updates(rows[:-1], 128, 20261108)


def test_old_new_replay_keeps_equal_presentations_and_complete_families():
    rows=[]
    for source,states in (('old',2),('new',3)):
        for state in range(states):
            for n in (30,40,60):
                row=example(n);row.update(image=f'{source}-{state}.npz',replay_source=source)
                rows.append(row)
    updates,report=family_updates(rows,512,20261108,balanced_replay=True)
    assert report['source_presentations']=={'old':512,'new':512}
    assert report['all_valid_examples_presented'] and report['presentations']==1024
    for u in updates:
        assert len({rows[i]['replay_source'] for i in u['indices']})==1


def test_prerequisite_cache_identity_includes_observation_and_roles():
    from auditembodied.vlm_duration_lora import start_key
    row=example();key=start_key(row)
    changed=copy.deepcopy(row);changed['plan']['local_relevel_steps']=60
    assert start_key(changed)==key
    changed['image']='h0.npz'
    assert start_key(changed)!=key
    changed=copy.deepcopy(row);changed['robot']['positions'][0][0]+=.001
    assert start_key(changed)!=key
    changed=copy.deepcopy(row);changed['plan'].update(repair_arm='right',keeper_arm='left')
    assert start_key(changed)!=key


def test_worker_and_trainer_share_joint_input_while_start_is_unchanged():
    row = example()
    context = {**row, 'post_wait': True}
    adapter, paths, prompt = routed_input({'robot': row['robot'], 'plan': row['plan']}, 'outcome', context, 'stage_joint')
    assert adapter == 'post_wait'
    assert (paths, prompt) == input_spec(row, 'outcome', 'stage_joint')
    assert input_spec(row, 'start', 'stage_joint') == ([row['image']], task_prompt_for(row, 'start'))
    assert 'nominal_current_relevel_angle_rad' not in input_spec(row, 'outcome', 'stage_late')[1]


def test_joint_marginals_distinguish_safe_incomplete_from_unsafe_complete():
    rows = [{'label': 'A', 'conditional_probabilities': [.99, .004, .003, .003]},
        {'label': 'B', 'conditional_probabilities': [.003, .99, .002, .005]},
        {'label': 'C', 'conditional_probabilities': [.001, .002, .99, .007]}]
    report = joint_marginal_metrics(rows)
    assert report['process_correct'] == report['completion_correct'] == 3
    assert report['process_missed_violations'] == report['process_false_alarms'] == 0
