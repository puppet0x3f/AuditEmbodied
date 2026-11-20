from dataclasses import asdict

import pytest

from auditembodied.agentchord_style import AgentChordStyle, graph_spec
from auditembodied.l4_selector import ProprioRule


def port(left=True, right=True, height=.0):
    return {'positions': [[0., 0., 1. + height], [.5, 0., 1.]],
            'reference_positions': [[0., 0., 1.], [.5, 0., 1.]],
            'fingers': [[.012, -.012] if loaded else [.001, -.001] for loaded in (left, right)]}


def policy():
    return AgentChordStyle(graph_spec(asdict(ProprioRule())))


def test_same_ports_and_registered_load_bounds_without_contact_truth():
    graph = policy()
    assert graph(port(False, True))['candidate'] == 'R_left'
    assert graph(port(True, False))['candidate'] == 'R_right'
    assert graph(port(False, False))['candidate'] == 'R_global'
    assert graph(port(height=.05))['candidate'] == 'R_right'
    assert graph(port())['candidate'] == 'continue'
    with pytest.raises(ValueError):
        graph({**port(), 'contact_truth': [True, True]})


def test_repair_does_not_monitor_its_own_intentionally_absent_grasp():
    graph = policy()
    assert not graph.monitor('left_keeper_missing', port(False, True))
    assert graph.monitor('left_keeper_missing', port(True, False))
    assert not graph.monitor('right_keeper_missing', port(True, False))


def test_upstream_runtime_switches_edges_and_checks_completion():
    graph, observation, actions = policy(), port(False, True), []

    def execute(skill, monitor):
        nonlocal observation
        index = monitor(observation) if monitor else None
        if index is None and skill != 'continue':
            actions.append(skill);observation = port()
        return {'monitor_index': index, 'actions': []}

    result = graph.execute(execute)
    assert result['completed'] and result['transitions'] == 3
    assert actions == result['executed_skills'] == ['R_left']


def test_intra_skill_interruption_resumes_pending_completion_check():
    graph, observation, actions = policy(), port(False, True), []

    def execute(skill, monitor):
        nonlocal observation
        if skill == 'R_left':
            observation = port(False, False)
        index = monitor(observation) if monitor else None
        if skill != 'continue':
            actions.append(skill)
        if skill == 'R_global':
            observation = port()
        return {'monitor_index': index, 'actions': []}

    result = graph.execute(execute)
    assert result['completed']
    assert actions == ['R_left', 'R_global']
    assert result['events'][-1]['monitors'] == ['incomplete']


def test_repeated_failure_retries_are_bounded_by_upstream_transition_limit():
    graph = policy()

    def execute(skill, monitor):
        return {'monitor_index': monitor(port(False, False)) if monitor else None, 'actions': []}

    result = graph.execute(execute)
    assert not result['completed'] and result['transitions'] == graph.spec['max_transitions']
