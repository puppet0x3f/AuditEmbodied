import json

import numpy as np
import pytest

from auditembodied.l4_selector import CANDIDATES, ProprioRule
from auditembodied.l4_offline_selection import features, fit_tree, predict, summarize


def outcome(action, success=True, seconds=2.7, path=.1):
    return {"candidate": action, "success": success, "protection_violation": False,
            "recovery_seconds": seconds, "recovery_path_m": path, "failure_reason": None}


def test_direct_features_equal_original_observable_front_end():
    port = {"positions": [[0, -.15, 1.04], [0, .15, 1]],
            "reference_positions": [[0, -.15, 1], [0, .15, 1]],
            "fingers": [[.04, -.04], [.011, -.009]]}
    rule = ProprioRule(); f = rule(port)["features"]
    np.testing.assert_array_equal(features(port, rule),
        np.r_[f["mean_finger_qpos"], f["loaded"], f["finger_asymmetry"], f["height_error_m"]])


def test_direct_tree_is_portable_and_learns_action_boundary():
    x = np.array([[0], [.1], [.9], [1.]])
    y = np.array([1, 1, 3, 3])
    tree = json.loads(json.dumps(fit_tree(x, y, max_depth=1, min_leaf=1)))
    assert predict(tree, [[.2], [.8]]) == ["R_left", "R_global"]


def test_unexecuted_healthy_recovery_is_unknown_not_failure_or_zero_cost():
    case = {"outcomes": {"continue": outcome("continue", seconds=0, path=0)}}
    result = summarize([case], ["R_global"])
    assert result["unobserved_outcomes"] == 1
    assert result["safe_success_rate"] is None
    assert result["protection_violation_rate"] is None
    assert result["mean_recovery_seconds"] is None
    assert result["safe_success_deficit_to_oracle"] is None


def test_safe_alternative_is_not_a_failure_for_disagreeing_with_oracle_label():
    case = {"outcomes": {"R_left": outcome("R_left", path=.12),
                         "R_right": outcome("R_right", path=.1)}}
    result = summarize([case], ["R_left"])
    assert result["safe_successes"] == result["joint_safe_oracle_scenes"] == 1
    assert result["safe_success_deficit_to_oracle"] == 0
    assert result["mean_time_gap_to_oracle_on_joint_safe"] == 0
    assert result["mean_path_gap_to_oracle_on_joint_safe"] == pytest.approx(.02)


def test_tree_complexity_selection_holds_out_whole_training_seeds(monkeypatch):
    import auditembodied.l4_offline_selection as module
    fitted = []

    def fit(x, y, depth, leaf):
        fitted.append(set(x[:, 0])); return {"action": "continue"}

    def predict_held(tree, x):
        assert not (set(x[:, 0]) & fitted[-1])
        return ["continue" if row[1] else "R_left" for row in x]

    cases = []
    for seed in range(3):
        for healthy in (False, True):
            action = "continue" if healthy else "R_left"
            cases.append({"seed": seed, "healthy": healthy, "x": np.array([seed, int(healthy)]),
                          "outcomes": {action: outcome(action, seconds=0 if healthy else 2.7)}})
    monkeypatch.setattr(module, "fit_tree", fit)
    monkeypatch.setattr(module, "predict", predict_held)
    _, report = module.calibrate_tree(cases)
    assert report["selected"]["fault"]["safe_successes"] == 3
    assert fitted[-1] == {0, 1, 2}
