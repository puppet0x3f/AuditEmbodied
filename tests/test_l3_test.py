from copy import deepcopy

import pytest

from auditembodied.l3_test import compare_prefixes, go_no_go, oracle_choice


def trial(candidate, success, violation=False, seconds=1.9):
    return {"candidate": candidate, "success": success, "protection_violation": violation,
            "recovery_seconds": seconds, "recovery_path_m": seconds / 10}


def test_oracle_excludes_the_rule_even_if_only_the_rule_succeeds():
    group = {"R1_retarget": trial("R1_retarget", False, True),
             "R3": trial("R3", False, True, 20.4), "rule": trial("rule", True)}
    oracle = oracle_choice(group)
    assert not oracle["success"]
    assert oracle["oracle_source"] == "R1_retarget"
    assert oracle["oracle_tier"] == "failed_local_fallback"
    assert oracle["recovery_seconds"] == 1.9


@pytest.mark.parametrize("local_unsafe,expected", [(False, "R1_retarget"), (True, "R3")])
def test_oracle_uses_actual_safe_candidate_cost(local_unsafe, expected):
    group = {"R1_retarget": trial("R1_retarget", True, local_unsafe),
             "R3": trial("R3", True, False, 20.4)}
    oracle = oracle_choice(group)
    assert oracle["oracle_source"] == expected
    assert oracle["recovery_seconds"] == group[expected]["recovery_seconds"]
    assert not oracle["protection_violation"]


def test_uniform_initialization_failures_remain_comparable_without_a_fault_prefix():
    state = {"qpos": [0.1, 0.2], "qvel": [0., 0.], "time": 0.5}
    row = {"initialization_state": state, "initialization_end_state": state,
           "failure_reason": "initialization_error"}
    group = {k: deepcopy(row) for k in ("R1_retarget", "R3", "rule")}
    result = compare_prefixes(group)
    assert result["comparable"] and result["uniform_initialization_failure"]
    assert result["fault_prefix_identical"] is None
    group["rule"]["initialization_end_state"]["time"] += .1
    assert not compare_prefixes(group)["comparable"]


def test_mismatched_fault_prefix_invalidates_comparison():
    state = {"qpos": [0.1], "time": .5}
    row = {"initialization_state": state, "initialization_end_state": state, "prefix_state": state}
    group = {k: deepcopy(row) for k in ("R1_retarget", "R3", "rule")}
    assert compare_prefixes(group)["comparable"]
    group["R3"]["prefix_state"]["qpos"][0] += 1e-8
    assert not compare_prefixes(group)["comparable"]


def test_go_requires_protection_as_well_as_success_and_cost():
    report = {"by_method": {"R1_retarget": {"safe_successes": 50, "episodes": 100},
        "R3": {"safe_successes": 90, "episodes": 100, "protection_violation_rate": .1},
        "rule": {"safe_successes": 95, "episodes": 100, "protection_violation_rate": .05}},
        "validity": {"all_cases_comparable": True, "all_sources_frozen": True,
            "initialized_seed_rate": 1., **{k: 0 for k in ("numerical_instabilities", "execution_errors",
            "contact_anchors", "manual_state_edits", "forbidden_calls", "wrench_leaks")}},
        "paired_costs_vs_R3": {"scenes": 90, "recovery_seconds": {"reduction_fraction": .3},
                                              "recovery_path_m": {"reduction_fraction": .4}}}
    criteria = {"minimum_initialization_seed_rate": .9, "global_success_margin": .05}
    assert go_no_go(report, criteria)["decision"] == "Go"
    report["by_method"]["rule"]["protection_violation_rate"] = .2
    assert go_no_go(report, criteria)["decision"] == "No-Go"
