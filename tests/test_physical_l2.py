import numpy as np
import pytest

from auditembodied.physical_l2 import ChangeHead, _grids, _rows, _select, predict


def test_change_predictions_do_not_use_future_observations():
    rng = np.random.default_rng(7)
    probabilities = rng.uniform(size=7)
    features = rng.normal(size=(7, 4))
    model = ChangeHead(np.linspace(-0.15, 0.15, 23), 0.1, np.zeros(23), np.ones(23))
    prefix = model.predict(probabilities[:4], features[:4])
    full = model.predict(probabilities, features)
    np.testing.assert_allclose(prefix, full[:4])


def test_change_signal_invalidates_an_otherwise_accepted_ttl():
    cache = {"p": np.array([0.99, 0.1, 0.1]), "change": np.array([0.0, 0.95, 0.1])}
    params = {"threshold": 0.8, "ttl": 3, "change_threshold": 0.5}
    assert predict(cache, 2, "ttl", params)
    assert not predict(cache, 2, "learned_change", params)
    assert not predict(cache, 2, "ttl", params, allow_stale=False)


def test_failed_initializations_remain_in_evaluation():
    episode = {"seed": 14, "perturbation": "nominal", "stages": ["receiver_grasp"],
               "evaluator_metadata": {"failure_reason": "initialization_not_supported"}}
    rows = _rows([{"episode": episode}], "frame", {"threshold": 0.9})
    assert len(rows) == 1
    assert rows[0]["failure_reason"] == "initialization_not_supported"
    assert not rows[0]["success"]


def test_threshold_plateau_prefers_fixed_l1_in_both_grid_orders():
    def point(threshold):
        return {"params": {"threshold": threshold}, "success_rate": 1.0,
                "false_pass_rate": 0.0, "mean_extra_observations": 26.0,
                "supported_coverage": 1.0, "risk": 0.0}
    points = [point(1e-5), point(0.25), point(0.5)]
    for candidates in (points, points[::-1]):
        assert _select(candidates, 0.02, reference_threshold=0.25)["params"]["threshold"] == 0.25


def test_threshold_grid_includes_recovery_certificate_and_excludes_subfloor():
    cache = {"p": np.array([1e-5, 0.25]),
             "episode": {"stages": ["gate_decision", "recovery_decision"]}}
    grid = _grids([cache], reference_threshold=0.25)
    assert {p["threshold"] for p in grid["frame"]} >= {0.25, 0.5}
    assert all(p["threshold"] >= 0.25 for points in grid.values() for p in points)


def test_unsafe_dev_grid_is_not_silently_selected():
    with pytest.raises(ValueError, match="false-pass constraint"):
        _select([{"false_pass_rate": 1.0}], 0.02)


def test_new_support_supersedes_old_alarm_including_after_r1():
    cache = {"p": np.array([0.99, 0.1, 0.95]), "change": np.array([0.0, 0.99, 0.9])}
    params = {"threshold": 0.8, "ttl": 3, "change_threshold": 0.5, "fresh_confirmation": True}
    assert predict(cache, 2, "learned_change", params, allow_stale=False)
    assert not predict(cache, 2, "learned_change", {**params, "fresh_confirmation": False})


def test_memory_invalidation_resets_only_after_fresh_confirmation():
    params = {"threshold": 0.8, "ttl": 3, "change_threshold": 0.5, "fresh_confirmation": True}
    cache = {"p": np.array([0.99, 0.1, 0.1, 0.99, 0.1]),
             "change": np.array([0.0, 0.99, 0.0, 0.0, 0.0])}
    assert not predict(cache, 2, "learned_change", params)
    assert predict(cache, 4, "learned_change", params)
    assert not predict(cache, 4, "learned_change", params, allow_stale=False)
