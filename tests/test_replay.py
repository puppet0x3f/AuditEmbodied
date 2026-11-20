from __future__ import annotations

import numpy as np
import pytest

from auditembodied import l1, temporal, l2_ablation


@pytest.mark.parametrize("entry", ["l1", "frame", "temporal8", "ablation"])
@pytest.mark.parametrize(
    "scores, labels, success, recovered, cost, unsafe",
    [
        # Accept immediately: the stored R1 branch must not consume observations.
        ([0.9, 0.9, 0.9, 0.9, 0.9], [1, 1, 0, 0, 0], True, False, 0, False),
        # A false release in recovery ends the branch before a later true frame.
        ([0.9, 0.1, 0.9, 0.9, 0.9], [1, 0, 0, 1, 1], False, False, 1, True),
        # Block, observe approach, then accept a genuinely supported recovery.
        ([0.9, 0.1, 0.1, 0.9, 0.9], [1, 0, 0, 1, 1], True, True, 2, False),
        # Recorded R1 success cannot override a gate that never releases.
        ([0.9, 0.1, 0.1, 0.1, 0.1], [1, 0, 0, 1, 1], False, False, 3, False),
    ],
)
def test_replay_truth_and_actual_observation_cost(
    monkeypatch, entry, scores, labels, success, recovered, cost, unsafe
):
    scores = np.asarray(scores)
    features = np.zeros((5, 1))
    episode = {
        "seed": 0, "perturbation": "late",
        "stages": ["receiver_grasp", "disturbance_late", "r1_recovery_approach",
                   "r1_recovery_action", "r1_recovery_action"],
        "labels": np.asarray(labels), "rgb": None, "proprio": None,
        "evaluator_metadata": {"r1": {"success": True}},
    }

    class Verifier:
        threshold = 0.5

        def predict_proba(self, rgb, proprio):
            return scores

    class TemporalModel:
        def predict_sequence(self, probabilities, raw_features):
            return probabilities, np.zeros(len(probabilities))

    verifier = Verifier()
    monkeypatch.setattr(temporal, "_probs_and_features", lambda *args: (scores, features))
    if entry == "l1":
        row = l1.visual_closed_loop_report(verifier, [episode])["rows"][0]
    elif entry == "frame":
        row = temporal._episode_policy_row(verifier, episode, "frame", {})
    elif entry == "temporal8":
        row = temporal._temporal_episode_policy_row(
            verifier, TemporalModel(), episode, {"state_threshold": 0.5, "change_threshold": 0.5}
        )
    else:
        row = l2_ablation._rows(
            [{"episode": episode, "probabilities": scores, "features": features}],
            "frame", {"threshold": 0.5},
        )[0]
        metrics = l2_ablation._metrics([row])
        assert metrics["unsafe_release_rate"] == float(unsafe)
        if unsafe:
            assert metrics["false_pass_rate"] > 0
            assert metrics["risk"] == 1.0
    assert row["success"] is success
    assert row["recovered"] is recovered
    assert row["r1_success"] is recovered
    assert row["extra_observations"] == cost
    assert row["observations"] == 1 + cost
    assert row["unsupported_release"] is unsafe
    assert row["success_scope"] == "support_at_replayed_release"


def test_replay_does_not_reuse_recovery_before_the_decision():
    episode = {
        "labels": np.asarray([0, 1, 1]),
        "stages": ["r1_recovery_action", "r1_recovery_action", "continuous_release"],
        "evaluator_metadata": {"r1": {"success": True}},
    }
    row = l1._replay_release_decision(episode, 2, lambda *args: False)
    assert row["success"] is False
    assert row["recovery_attempted"] is False
    assert row["observations"] == 1


def test_physical_replay_consumes_monitoring_prefix_and_checks_skill_boundary():
    episode = {
        "labels": np.asarray([1, 0, 0, 0, 1, 1]),
        "stages": ["receiver_grasp", "gate_decision", "r1_recovery_approach",
                   "r1_recovery_action", "recovery_decision", "giver_open"],
    }
    checked = []
    def predict(index, stale):
        checked.append(index)
        return index >= 2
    row = l1._replay_release_decision(episode, 1, predict)
    assert checked == [1, 4]
    assert row["release_index"] == 4
    assert row["success"]
    assert row["observations"] == 5
    assert row["extra_observations"] == 3
