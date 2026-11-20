from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from auditembodied.l3_matrix import MatrixConfig, run_trial
from auditembodied.recovery_rules import LocalThenTableRule


@pytest.mark.parametrize("measured_grasp", [True, False])
def test_scope_decision_ignores_oracle_diagnostics(monkeypatch, measured_grasp):
    import auditembodied.recovery_rules as rules
    # Deliberately contradict the measured robot joints with oracle metadata.
    monkeypatch.setattr(rules.PoseRetargetedR1Recovery, "run", Mock(return_value={
        "success": not measured_grasp, "predicate": {"receiver_supports": not measured_grasp}}))
    width = 0.018 if measured_grasp else 0.001
    monkeypatch.setattr(rules, "actual_gripper_positions", lambda env: np.array([[.018, -.018], [width, -width]]))
    global_recovery = SimpleNamespace(run=Mock())
    result = LocalThenTableRule().run(object(), object(), object(), None,
        np.zeros((2, 3)), np.tile(np.eye(3), (2, 1, 1)), [.018, .018], [0, -.45, .8], global_recovery)
    assert result["used_global_recovery"] is not measured_grasp
    assert global_recovery.run.call_count == int(not measured_grasp)


def test_rule_physically_escalates_from_local_failure_to_table_recovery():
    row = run_trial(0, "object_rotation", "rule", MatrixConfig(object_torque_nm=(-2, 0, 0)))
    assert row["recovery"]["sequence"] == ["R1_retarget", "R3"]
    assert row["success"] and row["table_cycle_verified"]
    assert row["giver_fully_released"] and row["receiver_stable_for_1s"]
    assert not row["protection_violation"]
    assert row["recovery_seconds"] >= 22.3 - 1e-8
    assert row["contact_anchor_count"] == row["state_edit_count_after_initialization"] == 0
    assert row["external_wrench_leak_steps"] == 0
