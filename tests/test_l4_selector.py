import copy
import json

import pytest

from auditembodied.l4_lift import run_trial
from auditembodied.l4_selector import INPUT_KEYS, ProprioRule, calibrate


def observation(fingers, height_difference=0.):
    return {"fingers": fingers, "positions": [[0., -.15, 1. + height_difference], [0., .15, 1.]],
            "reference_positions": [[0., -.15, 1.], [0., .15, 1.]]}


@pytest.mark.parametrize("fingers,height,expected", [
    ([[.01, -.01], [.01, -.01]], 0., "continue"),
    ([[.01, -.01], [.01, -.01]], .06, "R_right"),
    ([[.04, -.04], [.01, -.01]], .04, "R_left"),
    ([[.04, -.04], [.011, -.009]], .04, "R_global"),
    ([[.0005, -.0005], [.0005, -.0005]], 0., "R_global"),
])
def test_rule_uses_robot_measurements_without_fault_or_truth_labels(fingers, height, expected):
    class Forbidden:
        def __array__(self, *args, **kwargs):
            raise AssertionError("hidden context was read")
        def __bool__(self):
            raise AssertionError("hidden context was read")
    measured = observation(fingers, height)
    measured.update({k: Forbidden() for k in ("seed", "scenario", "stage", "evaluator", "rgb")})
    assert ProprioRule()(measured)["candidate"] == expected


def test_calibration_treats_unsafe_completion_as_failure():
    rows = []
    for seed, q in enumerate((.0101, .0106)):
        current = observation([[.04, -.04], [q, -(0.02-q)]], .04)
        for candidate in ("continue", "R_left", "R_right", "R_global"):
            rows.append({"seed": seed, "scenario": "left_loss", "candidate": candidate,
                         "success": candidate in ("R_left", "R_global"),
                         "protection_violation": seed == 1 and candidate == "R_left",
                         "recovery_seconds": 11.6 if candidate == "R_global" else 4.3,
                         "recovery_path_m": 1. if candidate == "R_global" else .2,
                         "trace": [{"stage": "nominal_stable_hold", "phase": "initialization",
                                    "robot": {"positions": current["reference_positions"]}},
                                   {"stage": "fault_withdraw", "phase": "fault", "robot": current}]})
    rule, report = calibrate(rows)
    assert rule.asymmetry_limit == .02
    assert report["grid"][-1]["safe_successes"] == 1
    assert report["grid"][0]["safe_successes"] == 2


def test_online_selector_receives_only_robot_port_and_matches_fixed_physics():
    seen = []
    def selector(port):
        assert set(port) == INPUT_KEYS
        seen.append(copy.deepcopy(port))
        return ProprioRule()(port)
    selected = run_trial(0, "coupled_height", selector=selector)
    fixed = run_trial(0, "coupled_height", "R_right")
    assert len(seen) == 1 and selected["selection"]["input"] == seen[0]
    assert selected["selection"]["rgb_frames"] == 0
    assert selected["candidate"] == "R_right"
    assert selected["success"] and not selected["protection_violation"]
    for name in ("initialization_state", "preclose_state", "initialization_end_state", "prefix_state"):
        assert selected[name] == fixed[name]
    assert selected["trace"] == fixed["trace"]


def test_healthy_physical_hold_selects_continue_and_keeps_one_second():
    row = run_trial(0, "sham", selector=ProprioRule())
    assert row["candidate"] == "continue" and row["fault_class_valid"]
    assert row["success"] and not row["protection_violation"]
    assert row["hold_seconds"] >= 1. - 1e-9
    assert row["recovery_seconds"] == row["recovery_path_m"] == 0.
    assert row["contact_anchor_count"] == row["state_edit_count_after_initialization"] == 0


@pytest.mark.parametrize("global_failure,policy_failure,expected", [
    (True, False, "Go"), (False, False, "Go"), (False, True, "No-Go"),
])
def test_assessment_accepts_better_safe_coverage_and_rejects_worse(
        tmp_path, monkeypatch, global_failure, policy_failure, expected):
    import auditembodied.l4_selector as module
    (tmp_path / "dev").mkdir()
    (tmp_path / "frozen.json").write_text(json.dumps({"dev_seeds": [0], "rule": {}}))
    rows = []
    for scenario in (*module.SCENARIOS, "sham"):
        port = observation([[.01, -.01], [.01, -.01]], 0. if scenario == "sham" else .06)
        choice = ProprioRule()(port)
        for method in (("proprio",) if scenario == "sham" else module.METHODS):
            failed = scenario == "left_loss" and (
                method == "R_global" and global_failure or method == "proprio" and policy_failure)
            row = {"seed": 0, "scenario": scenario, "method": method,
                   "candidate": choice["candidate"] if method == "proprio" else method,
                   "success": not failed, "protection_violation": False,
                   "failure_reason": "table_regrasp_not_verified" if failed else None,
                   "recovery_seconds": 11.6 if method == "R_global" else 1.9,
                   "recovery_path_m": 1. if method == "R_global" else .1,
                   "fault_class_valid": True, "contact_anchor_count": 0,
                   "state_edit_count_after_initialization": 0, "blocked_calls": []}
            row.update({k: {"time": 1.} for k in module.STATE_KEYS})
            if method == "proprio":
                row["selection"] = {"input": port, "decision": choice, "rgb_frames": 0, "robot_observations": 1}
            rows.append(row)
    monkeypatch.setattr(module, "verify_frozen", lambda manifest: None)
    monkeypatch.setattr(module, "read_rows", lambda path: rows)
    report = module.analyze(tmp_path, "dev")
    assert report["decision"]["result"] == expected
    assert report["decision"]["checks"]["safe_coverage_not_lower_than_global"] is (not policy_failure)
