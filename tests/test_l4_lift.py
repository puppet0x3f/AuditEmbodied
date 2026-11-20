import copy

import pytest

from auditembodied.l4_lift import LiftConfig, TwoArmLiftEnv, analyze_matrix, run_trial


def test_lift_requires_both_grasps_balance_and_a_full_second():
    row = run_trial(0)
    assert row["success"] and row["nominal_ready"]
    assert row["nominal_hold_seconds"] >= 1.0 - 1e-9
    held = [f["evaluator"] for f in row["trace"] if f["stage"] == "nominal_stable_hold"]
    assert len(held) == 20
    assert all(all(f["supports"]) and f["clearance"] >= .10 for f in held)
    assert all(f["tilt_deg"] <= 15 and f["handle_height_difference"] <= .03 for f in held)
    assert row["contact_anchor_count"] == row["state_edit_count_after_initialization"] == 0
    assert not row["protection_violation"]


def test_healthy_pose_with_short_hold_is_not_success():
    row = run_trial(0, config=LiftConfig(hold_steps=19))
    assert all(f["evaluator"]["ready"] for f in row["trace"] if f["stage"] == "nominal_stable_hold")
    assert not row["success"]
    assert row["failure_reason"] == "nominal_lift_or_hold_failed"


def test_lift_blocks_initialization_edits_before_physical_closing(monkeypatch):
    def edited_action(self, *args, **kwargs):
        self.sync_gripper_controller(0, closed=True)
    monkeypatch.setattr(TwoArmLiftEnv, "action", edited_action)
    row = run_trial(0)
    assert not row["success"] and row["failure_reason"] == "forbidden_state_edit"
    assert [c["method"] for c in row["blocked_calls"]] == ["sync_gripper_controller"]
    assert row["state_edit_count_after_initialization"] == 0


def test_local_balance_correction_preserves_existing_grasps_and_paired_prefix():
    direct = run_trial(0, "coupled_height", "continue")
    local = run_trial(0, "coupled_height", "R_left")
    assert not direct["success"] and direct["failure_reason"] == "imbalance"
    assert local["success"] and not local["protection_violation"]
    assert local["fault_class_valid"]
    for key in ("initialization_state", "preclose_state", "initialization_end_state", "prefix_state"):
        assert direct[key] == local[key]
    for frame in local["trace"]:
        if frame["phase"] == "recovery":
            assert frame["robot"]["action"][6] == frame["robot"]["action"][13] == 1.


def test_oracle_excludes_unsafe_completions_and_detects_a_different_prefix():
    rows = []
    for candidate, success, violation, seconds in (
        ("continue", False, False, 0.),
        ("R_left", True, True, 1.),
        ("R_right", False, True, 1.),
        ("R_global", True, False, 11.6),
    ):
        row = dict(seed=0, scenario="left_loss", candidate=candidate, success=success,
                   protection_violation=violation, recovery_seconds=seconds, recovery_path_m=seconds / 10,
                   fault_class_valid=True)
        for name in ("initialization_state", "preclose_state", "initialization_end_state", "prefix_state"):
            row[name] = {"qpos": [0., 1.], "time": 1.}
        rows.append(row)
    report = analyze_matrix(rows)
    assert report["oracle"]["selected_candidates"] == {"R_global": 1}
    assert report["oracle_vs_global_paired_cost"]["recovery_seconds"]["reduction_fraction"] == 0.
    assert report["prefix_pairing"]["all_equal"]
    changed = copy.deepcopy(rows)
    changed[0]["prefix_state"]["qpos"][0] = .1
    assert not analyze_matrix(changed)["prefix_pairing"]["all_equal"]
