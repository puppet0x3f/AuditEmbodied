import copy
import json

import pytest

from auditembodied.l4_test import CRITERIA, TEST_SEEDS, intervals, method_summary, paired_prefix, prepare, verdict


def test_pairing_detects_tiny_mismatch_and_retains_uniform_initialization_failure():
    row = {k: {"qpos": [0., 1.], "time": 1.} for k in
           ("initialization_state", "preclose_state", "initialization_end_state", "prefix_state")}
    row["nominal_ready"] = True
    group = {"R_left": copy.deepcopy(row), "proprio": copy.deepcopy(row)}
    assert paired_prefix(group)["comparable"]
    group["proprio"]["prefix_state"]["time"] += 1e-12
    assert not paired_prefix(group)["comparable"]
    for r in group.values():
        r.pop("prefix_state")
        r.update(nominal_ready=False, failure_reason="nominal_lift_or_hold_failed")
    result = paired_prefix(group)
    assert result["comparable"] and result["uniform_initial_failure"] and not result["prefix_reached"]


@pytest.mark.parametrize("policy_safe,global_safe,policy_guard,expected", [
    (60, 59, 0., "Go"), (59, 59, 0., "Go"), (58, 59, 0., "No-Go"), (60, 60, .02, "No-Go"),
])
def test_prefrozen_verdict_accepts_better_coverage_but_rejects_safety_regression(
        policy_safe, global_safe, policy_guard, expected):
    validity = {k: 0 for k in ("numerical_instabilities", "execution_errors", "contact_anchors",
                              "state_edits_after_initialization", "forbidden_calls")}
    validity.update(all_cases_comparable=True, all_sources_and_prior_results_frozen=True,
                    policy_input_valid=True, selected_candidate_execution_consistent=True,
                    initialized_seed_rate=1., valid_fault_rate=1.)
    report = {"validity": validity,
              "by_method": {"proprio": {"safe_successes": policy_safe, "safe_success_rate": policy_safe / 60,
                                        "protection_violation_rate": policy_guard},
                            "R_global": {"safe_successes": global_safe, "protection_violation_rate": 0.},
                            "R_left": {"safe_successes": 40}, "R_right": {"safe_successes": 39}},
              "paired_cost_vs_global": {"paired_safe_scenes": 58, "recovery_seconds": {"reduction_fraction": .5},
                                        "recovery_path_m": {"reduction_fraction": .5}},
              "healthy_control": {"safe_success_rate": 1., "all_initialized_safely_continue": True}}
    assert verdict(report, CRITERIA)["result"] == expected


def test_safe_summary_does_not_count_unsafe_final_completion():
    row = {"success": True, "protection_violation": True, "failure_reason": None,
           "recovery_seconds": 4.3, "recovery_path_m": .2, "blocked_calls": [],
           "protection": {"keeper_grasp_lost_steps": 7, "max_unsupported_seconds": 0., "deep_interarm_contact_steps": 0}}
    result = method_summary([row])
    assert result["successes"] == 1 and result["safe_successes"] == 0
    assert result["safe_failure_reasons"] == {"safety_condition_failed": 1}
    assert result["guard_causes_overlapping"] == {"keeper_grasp_interrupted_or_absent": 1}


def test_seed_cluster_bootstrap_preserves_constant_paired_reduction():
    groups = {(seed, scene): {method: {"success": True, "protection_violation": False,
                                     "recovery_seconds": 1. if method == "proprio" else 2.,
                                     "recovery_path_m": .1 if method == "proprio" else .2}
                             for method in ("proprio", "R_left", "R_right", "R_global")}
              for seed in (0, 1) for scene in ("left_loss", "right_loss", "coupled_height")}
    result = intervals(groups, {"draws": 100, "analysis_rng_seed": 0, "unit": "seed"})
    assert result["safe_success_delta_vs_R_global"]["ci95"] == [0., 0.]
    assert result["paired_recovery_seconds_reduction_vs_global"]["ci95"] == [.5, .5]
    assert result["paired_recovery_path_m_reduction_vs_global"]["ci95"] == [.5, .5]


def test_prepare_rejects_nonfresh_seeds_before_creating_test(tmp_path, monkeypatch):
    import auditembodied.l4_test as module
    reference = tmp_path / "reference"; reference.mkdir()
    (reference / "frozen.json").write_text("{}")
    monkeypatch.setattr(module, "verify_frozen", lambda manifest: None)
    local, remote = tmp_path / "local.json", tmp_path / "remote.json"
    local.write_text(json.dumps({"fresh": False, "proposed_test_seeds": TEST_SEEDS}))
    remote.write_text(json.dumps({"fresh": True, "proposed_test_seeds": TEST_SEEDS}))
    target = tmp_path / "test"
    with pytest.raises(ValueError, match="not verified fresh"):
        prepare(target, reference, local, remote)
    assert not target.exists()
