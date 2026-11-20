from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from auditembodied.l3_visual import RGBRecorder, support_targets
from auditembodied.visual_recovery import (
    MultiSupportVerifier, VisualRecoverySelector, binary_metrics, calibrate_heads, fit_heads, visual_features,
)


@pytest.mark.parametrize("giver_speed,collision,expected", [(.01, False, [1, 1, 1]),
    (.4, False, [0, 1, 0]), (.01, True, [1, 1, 0])])
def test_coordination_requires_two_stable_supports_without_collision(giver_speed, collision, expected):
    predicate = dict(giver_holds=True, receiver_supports=True, clearance=.2,
                     height_threshold=.1, relative_speed_threshold=.2)
    assert support_targets(predicate, giver_speed, collision).tolist() == expected


def test_robot_feature_boundary_rejects_extra_privileged_coordinates():
    rgb = np.zeros((2, 4, 64, 64, 3), np.uint8)
    robot = np.zeros((2, 46), np.float32)
    assert visual_features(rgb, robot).shape == (2, 1582)
    for size in (20, 49):
        with pytest.raises(ValueError, match="46 robot-only"):
            visual_features(rgb, np.zeros((2, size)))


def test_saved_model_uses_images_at_fixed_robot_state(tmp_path):
    rgb = np.zeros((2, 4, 64, 64, 3), np.uint8); rgb[1] = 255
    robot = np.zeros((2, 46), np.float32)
    features = visual_features(rgb, robot)
    weights = np.zeros((features.shape[1], 3)); weights[0] = 4
    model = MultiSupportVerifier(np.zeros(features.shape[1]), np.ones(features.shape[1]),
                                weights, np.full(3, -2.), np.full(3, .5))
    model.save(tmp_path / "support.npz")
    loaded = MultiSupportVerifier.load(tmp_path / "support.npz")
    probability = loaded.predict_proba(rgb, robot)
    assert np.all(probability[0] < .2) and np.all(probability[1] > .8)
    np.testing.assert_array_equal(probability, model.predict_proba(rgb, robot))


def test_head_fit_and_dev_thresholds_keep_false_positive_constraint():
    x = np.linspace(-2, 2, 40)[:, None]
    y = np.repeat((x > 0).astype(int), 3, axis=1)
    model = fit_heads(x, y, epochs=150)
    probabilities = model.probabilities_from_features(x)
    thresholds, metrics = calibrate_heads(probabilities, y, false_positive_budget=0)
    assert np.all(thresholds >= .5)
    assert all(m["recall"] >= .9 and m["fp"] == 0 for m in metrics.values())
    assert binary_metrics([1, 0], [0, 0])["precision"] == 0


def test_carrier_calibration_preserves_recognition_with_one_high_confidence_negative():
    y = np.array([[1,1,1]]*6 + [[0,0,0]]*4)
    probabilities = np.array([[.8,.8,.8]]*6 + [[.999,.999,.999]] + [[.01,.01,.01]]*3)
    thresholds, metrics = calibrate_heads(probabilities, y, false_positive_budget=0)
    assert thresholds[0] < .8 and metrics["giver_supports"]["recall"] == 1
    assert metrics["giver_supports"]["fp"] == 1
    assert metrics["receiver_supports"]["fp"] == metrics["coordinated_support"]["fp"] == 0


def test_explicit_rendering_changes_only_camera_observables():
    env = SimpleNamespace(raw=SimpleNamespace(modify_observable=Mock()))
    recorder = RGBRecorder()
    recorder.after_reset(env)
    assert env.raw.modify_observable.call_count == 4
    assert all(c.args[1:] == ("enabled", False) for c in env.raw.modify_observable.call_args_list)


@pytest.mark.parametrize("predictions,expected", [
    ([[0, 1, 0]], ["continue"]),
    ([[1, 1, 1]], ["continue"]),
    ([[1, 0, 0], [1, 1, 1]], ["R1_retarget"]),
    ([[1, 0, 0], [1, 0, 0]], ["R1_retarget", "R3"]),
    ([[0, 0, 0]], ["R3"]),
])
def test_visual_scope_uses_sensor_predictions_not_skill_oracle_metadata(monkeypatch, predictions, expected):
    import auditembodied.recovery_skills as skills
    local = Mock(return_value={"success": True, "predicate": {"receiver_supports": True}})
    monkeypatch.setattr(skills.PoseRetargetedR1Recovery, "run", local)
    verifier = SimpleNamespace(thresholds=np.full(3, .5),
        predict_proba=Mock(side_effect=[np.array([p]) for p in predictions]))
    global_recovery = SimpleNamespace(run=Mock(return_value={"success": True}))
    observe = Mock(return_value={"rgb": np.zeros((4,64,64,3), np.uint8), "robot": np.zeros(46)})
    result = VisualRecoverySelector(verifier).run(object(), object(), object(), None,
        np.zeros((2,3)), np.tile(np.eye(3), (2,1,1)), [.018,.018], [0,-.45,.8],
        global_recovery, observe=observe)
    assert result["sequence"] == expected
    assert result["used_global_recovery"] == ("R3" in expected)
    assert observe.call_count == len(predictions)
    assert local.call_count == int("R1_retarget" in expected)
    assert global_recovery.run.call_count == int("R3" in expected)
