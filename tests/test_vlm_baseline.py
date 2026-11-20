import json

import numpy as np

from auditembodied.vlm_baseline import input_hash, parse_scores, robot_description, routes, routing_metrics


def test_probability_parser_accepts_extracted_json_and_rejects_nonfinite_or_missing_heads():
    values, valid = parse_scores('```json\n{"giver_supports":0.9,"receiver_supports":0.2,"coordinated_support":0.1}\n```')
    assert valid and values.tolist() == [.9, .2, .1]
    for raw in ('{"giver_supports":NaN,"receiver_supports":0,"coordinated_support":0}',
                '{"giver_supports":1.1,"receiver_supports":0,"coordinated_support":0}', 'uncertain'):
        p, valid = parse_scores(raw)
        assert not valid and p.tolist() == [0., 0., 0.]


def test_robot_prompt_preserves_all_float32_measurements_without_extra_context():
    robot = np.linspace(-.123456789, .987654321, 46, dtype=np.float32)
    text = robot_description(robot)
    recovered = []
    for line in text.splitlines(): recovered.extend(json.loads(line.split('=', 1)[1]))
    np.testing.assert_array_equal(np.asarray(recovered, np.float32), robot)
    assert not any(word in text for word in ('fault', 'scenario', 'seed', 'stage', 'object_position'))


def test_scope_logic_keeps_receiver_only_completion_and_requires_coordination_when_both_support():
    states = np.array([[False, True, False], [True, True, False], [True, True, True], [False, False, False]])
    assert routes(states).tolist() == ['continue', 'R1_retarget', 'continue', 'R3']
    predicted = np.array([[False, True, False], [True, True, True], [True, True, True], [False, False, False]])
    metrics = routing_metrics(states, predicted)
    assert metrics['unsafe_continue_count'] == 1
    assert metrics['release_ineligible_examples'] == 2


def test_input_cache_key_depends_only_on_current_images_and_robot_measurements():
    image = np.zeros((4, 64, 64, 3), dtype=np.uint8); robot = np.zeros(46, dtype=np.float32)
    key = input_hash(image, robot)
    assert input_hash(image.copy(), robot.copy()) == key
    robot[20] = .01
    assert input_hash(image, robot) != key
