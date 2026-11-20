import numpy as np
import pytest

from auditembodied.l3_matrix import MatrixConfig, _grasp_relative_change, run_trial
from auditembodied.recovery_skills import ContinuousR3Recovery


def test_receiver_support_allows_direct_completion_without_r2():
    direct = run_trial(0, "giver_loss", "continue")
    r2 = run_trial(0, "giver_loss", "R2")
    assert direct["success"] and r2["success"]
    assert direct["recovery_seconds"] == 0
    assert r2["skill_goal_achieved"]
    assert r2["receiver_support_continuous_during_recovery"]
    assert r2["recovery_seconds"] >= 1.9 - 1e-9
    for key in direct["prefix_state"]:
        np.testing.assert_array_equal(direct["prefix_state"][key], r2["prefix_state"][key])
    assert len(r2["trace"][0]["robot"]["proprio20"]) == 20
    assert len(r2["trace"][0]["robot"]["actual_finger_qpos4"]) == 4


@pytest.mark.parametrize("seed", [0, 1])
def test_r3_physically_puts_down_picks_up_and_completes_handover(seed):
    # Seed 1 previously picked up successfully but grasped empty space on
    # returning to the original receiver height.
    row = run_trial(seed, "nominal", "R3")
    assert row["success"] and row["handover_completed"]
    assert row["table_setdown_verified"] and row["table_regrasp_and_lift_verified"]
    assert row["giver_fully_released"] and row["receiver_stable_for_1s"]
    assert row["contact_anchor_count"] == row["state_edit_count_after_initialization"] == 0
    assert not row["protection_violation"]


def test_table_setdown_alone_is_not_r3_success():
    row = run_trial(0, "nominal", "R3", MatrixConfig(r3_search_offsets=()))
    assert row["table_setdown_verified"]
    assert not row["success"] and not row["handover_completed"]
    assert row["failure_reason"] == "table_pickup_not_verified"


def test_r3_rejects_kinematic_table_placement(monkeypatch):
    def edited_recovery(self, env, *args, **kwargs):
        env.place_object_on_table()
    monkeypatch.setattr(ContinuousR3Recovery, "run", edited_recovery)
    row = run_trial(0, "nominal", "R3")
    assert not row["success"]
    assert row["failure_reason"] == "forbidden_state_edit"
    assert row["blocked_calls"][0]["method"] == "place_object_on_table"


def test_displaced_shared_carry_is_repaired_by_retargeted_local_r1():
    cfg = MatrixConfig(coupled_mode="shared_shift_receiver_loss", coupled_shared_shift_m=0.12)
    rows = [run_trial(0, "coupled_offset", skill, cfg) for skill in ("continue", "R1", "R1_retarget", "R3")]
    assert not rows[0]["success"] and not rows[1]["success"]
    assert rows[2]["success"] and not rows[2]["protection_violation"]
    assert rows[3]["success"] and not rows[3]["protection_violation"]
    assert rows[2]["giver_hold_continuous_during_recovery"]
    assert rows[2]["recovery_seconds"] < rows[3]["recovery_seconds"]
    # Isolate target relocation: keep the frozen R1 wrist actions and timings.
    local_steps = [x for x in rows[2]["trace"] if x["phase"] == "recovery"]
    assert len(local_steps) == 38
    for point in local_steps:
        action = np.asarray(point["robot"]["action14"])
        np.testing.assert_array_equal(action[[3, 4, 5, 10, 11, 12]], np.zeros(6))
    for row in rows:
        assert row["post_fault_predicate"]["giver_holds"]
        assert not row["post_fault_predicate"]["receiver_supports"]
        assert row["contact_anchor_count"] == row["state_edit_count_after_initialization"] == 0
        for key in row["prefix_state"]:
            np.testing.assert_array_equal(rows[0]["prefix_state"][key], row["prefix_state"][key])


def test_common_rigid_motion_is_not_object_slip_in_the_gripper():
    positions = np.array([[0.0, -0.06, 1.0], [0.0, 0.06, 1.0]])
    object_position = np.array([0.02, 0.0, 1.0])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    offset = np.array([0.12, -0.1, 0.2])
    before = {"robot": {"proprio20": positions.ravel().tolist(),
                         "actual_eef_rotations2x3x3": [np.eye(3).tolist()] * 2},
              "evaluator": {"object_position": object_position.tolist(), "object_quaternion": [1, 0, 0, 0]}}
    after = {"robot": {"proprio20": (positions @ rotation.T + offset).ravel().tolist(),
                        "actual_eef_rotations2x3x3": [rotation.tolist()] * 2},
             "evaluator": {"object_position": (rotation @ object_position + offset).tolist(),
                           "object_quaternion": [2**-.5, 0, 0, 2**-.5]}}
    for change in _grasp_relative_change(before, after):
        assert change["translation_norm_m"] < 1e-6
        assert change["rotation_deg"] < 0.1


def test_physical_object_rotation_needs_more_than_pose_retargeting():
    cfg = MatrixConfig(object_torque_nm=(-2.0, 0.0, 0.0))
    local, global_ = [run_trial(0, "object_rotation", k, cfg) for k in ("R1_retarget", "R3")]
    assert not local["success"]
    assert global_["success"] and not global_["protection_violation"]
    assert global_["table_cycle_verified"]
    for row in (local, global_):
        assert row["post_fault_predicate"]["giver_holds"]
        assert row["grasp_relative_change"][0]["rotation_deg"] > 8
        assert row["external_wrench_at_recovery_start"] == row["external_wrench_leak_steps"] == 0
        assert row["contact_anchor_count"] == row["state_edit_count_after_initialization"] == 0
        for key in row["prefix_state"]:
            np.testing.assert_array_equal(local["prefix_state"][key], row["prefix_state"][key])


@pytest.mark.parametrize("seed", [8, 64, 65, 67])
def test_smooth_pulse_stays_stable_on_formerly_unstable_seeds(seed):
    cfg = MatrixConfig(object_torque_nm=(-1.0, 0.0, 0.0), wrench_steps=13, wrench_profile="half_sine")
    row = run_trial(seed, "object_rotation", "R1_retarget", cfg)
    assert not any(row["bad_state_warnings"].values())
    assert row.get("numerical_failure_stage") is None
    assert row["post_fault_predicate"]["giver_holds"]
    assert 8.0 < row["grasp_relative_change"][0]["rotation_deg"] < 90.0
    assert np.all(np.diff([p["time"] for p in row["trace"]]) > 0)
    assert row["external_wrench_at_recovery_start"] == row["external_wrench_leak_steps"] == 0
    assert row["contact_anchor_count"] == row["state_edit_count_after_initialization"] == 0


def test_numerical_reset_aborts_before_recovery_instead_of_becoming_a_new_prefix():
    row = run_trial(8, "object_rotation", "R1_retarget", MatrixConfig(object_torque_nm=(-2, 0, 0)))
    assert row["failure_reason"] == "numerical_instability"
    assert row["numerical_failure_stage"] == "fault_wrench"
    assert row["time_rollback_detected"]
    assert "prefix_state" not in row and "recovery" not in row
    assert not any(point["phase"] == "recovery" for point in row["trace"])
