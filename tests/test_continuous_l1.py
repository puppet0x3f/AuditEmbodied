from __future__ import annotations

import numpy as np
import pytest

from auditembodied.continuous import TrajectoryEpisode, write_hdf5
from auditembodied.l1 import _features, load_hdf5, train_and_evaluate, LogisticVisualVerifier
from auditembodied.temporal import evaluate_policy


def test_invalidated_camera_data_cannot_enter_training(tmp_path):
    import h5py
    path = tmp_path / "invalid.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["invalid_reason"] = "corrupted EGL camera buffers"
    with pytest.raises(ValueError, match="corrupted EGL"):
        load_hdf5(path)


def _episode(seed: int, perturbation: str, offset: int) -> TrajectoryEpisode:
    episode = TrajectoryEpisode(seed=seed, perturbation=perturbation)
    for index in range(4):
        label = int(index >= 2)
        # Give the two classes a visible RGB separation so the tiny logistic
        # head is tested on the same data path as the real collector.
        value = 220 if label else 20
        episode.rgb.append(np.full((4, 16, 16, 3), value + offset, dtype=np.uint8))
        episode.actions.append(np.zeros(14, dtype=np.float32))
        episode.proprio.append(np.zeros(23, dtype=np.float32))
        episode.labels.append(label)
        episode.stages.append("continuous_release" if index >= 2 else "reset")
        episode.predicates.append({"receiver_supports": bool(label)})
    return episode


def test_seed_filtered_training_does_not_read_held_out_arrays(tmp_path):
    import h5py
    path = tmp_path / "split.h5"
    write_hdf5([_episode(0, "nominal", 0), _episode(60, "nominal", 1),
                _episode(200, "nominal", 2)], path)
    with h5py.File(path, "r+") as handle:
        # A held-out stream that would fail immediately if the loader read it.
        del handle["episodes/episode_00002/rgb"]
    selected = load_hdf5(path, seeds=[0, 60])
    assert [ep["seed"] for ep in selected] == [0, 60]
    with pytest.raises(ValueError, match="missing robot stream"):
        load_hdf5(path)


def test_trajectory_hdf5_and_l1_roundtrip(tmp_path) -> None:
    dataset = tmp_path / "trajectory.h5"
    write_hdf5([_episode(0, "nominal", 0), _episode(1, "late", 1), _episode(2, "early", 2)], dataset)
    assert len(load_hdf5(dataset)) == 3
    report = train_and_evaluate(dataset, tmp_path / "model.npz")
    assert report["schema"] == "auditembodied.l1.single_frame.v1"
    assert report["test"]["frames"] > 0
    assert (tmp_path / "model.npz").exists()


def test_robot_only_features_and_temporal_policy(tmp_path) -> None:
    dataset = tmp_path / "trajectory.h5"
    write_hdf5([_episode(0, "nominal", 0), _episode(1, "late", 1), _episode(2, "early", 2)], dataset)
    report = train_and_evaluate(dataset, tmp_path / "model.npz")
    verifier = LogisticVisualVerifier.load(tmp_path / "model.npz")
    episodes = load_hdf5(dataset)
    # The legacy trajectory shape may contain three object-state values, but
    # the verifier feature path retains only robot proprioception.
    assert _features(episodes[0]["rgb"], episodes[0]["proprio"]).shape[1] == 4 * 8 * 8 * 6 + 20
    temporal = evaluate_policy(verifier, episodes, "change", {"delta": 0.5})
    assert temporal["episodes"] > 0
    assert temporal["false_pass_rate"] >= 0.0
