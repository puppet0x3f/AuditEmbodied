"""A lightweight single-frame visual verifier for the L1 handover loop.

This is intentionally a small, dependency-free baseline: four RGB views are
pooled to a fixed grid, concatenated with the recorded proprioceptive vector,
and fitted with a deterministic logistic head.  It is an actual image-based
verifier and closed-loop gate input, while leaving a DINOv2/MLP replacement as
an isolated upgrade rather than making the first visual experiment depend on
large model downloads.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .continuous import (
    ACTION_DIM,
    EVALUATOR_SCHEMA,
    ROBOT_PROPRIO_DIM,
    TRAJECTORY_FIELDS,
    TRAJECTORY_SCHEMA,
    evaluator_path,
)


def _features(rgb: np.ndarray, proprio: np.ndarray, grid: int = 8) -> np.ndarray:
    """Extract fixed-size per-frame image + proprio features."""

    images = np.asarray(rgb, dtype=np.float32) / 255.0
    if images.ndim != 5:
        raise ValueError(f"expected [T,V,H,W,3] RGB array, got {images.shape}")
    t, views, height, width, channels = images.shape
    if channels != 3:
        raise ValueError(f"expected RGB channels, got {channels}")
    # Average and standard deviation pooling retain coarse object / gripper
    # appearance while keeping the head small enough for CPU training.
    gh = max(1, min(grid, height))
    gw = max(1, min(grid, width))
    h_trim = (height // gh) * gh
    w_trim = (width // gw) * gw
    pooled = images[:, :, :h_trim, :w_trim].reshape(t, views, gh, h_trim // gh, gw, w_trim // gw, channels)
    mean = pooled.mean(axis=(3, 5))
    std = pooled.std(axis=(3, 5))
    image_features = np.concatenate([mean, std], axis=-1).reshape(t, -1)
    proprio_array = np.asarray(proprio, dtype=np.float32)
    if proprio_array.ndim != 2 or proprio_array.shape[0] != t:
        raise ValueError(f"proprio shape {proprio_array.shape} does not match frames {t}")
    # Legacy 23-D files appended the simulator's hammer xyz after the 20
    # robot EEF / joint coordinates. Those three privileged values must never
    # enter a visual verifier. Live observations expose only the robot values.
    proprio_array = proprio_array[:, :20]
    return np.concatenate([image_features, proprio_array], axis=1).astype(np.float32)


@dataclass
class LogisticVisualVerifier:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    threshold: float = 0.5
    # Scores below this bound are treated as ordinary HOLD; scores in
    # [abstain_low, threshold) are explicitly marked ABSTAIN by the calibrated
    # policy.  Both outcomes keep the giver closed, but the distinction lets
    # the experiment report where the visual head is uncertain.
    abstain_low: float = 0.5
    feature_grid: int = 8

    def predict_proba(self, rgb: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        x = _features(rgb, proprio, self.feature_grid)
        # Keep old checkpoints loadable: the first prototype appended three
        # privileged hammer coordinates. New live observations do not expose
        # them, so legacy inference pads those slots with zeros while all new
        # training and reported models use the robot-only feature dimension.
        if x.shape[1] < self.mean.shape[0]:
            x = np.pad(x, ((0, 0), (0, self.mean.shape[0] - x.shape[1])))
        elif x.shape[1] > self.mean.shape[0]:
            x = x[:, : self.mean.shape[0]]
        x = (x - self.mean) / self.scale
        logits = x @ self.weights + float(self.bias)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))

    def predict(self, rgb: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        return (self.predict_proba(rgb, proprio) >= self.threshold).astype(np.int8)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            mean=self.mean,
            scale=self.scale,
            weights=self.weights,
            bias=np.asarray([self.bias], dtype=np.float32),
            threshold=np.asarray([self.threshold], dtype=np.float32),
            abstain_low=np.asarray([self.abstain_low], dtype=np.float32),
            feature_grid=np.asarray([self.feature_grid], dtype=np.int64),
            proprio_dim=np.asarray([ROBOT_PROPRIO_DIM], dtype=np.int64),
            action_dim=np.asarray([ACTION_DIM], dtype=np.int64),
            protocol_schema=np.asarray([TRAJECTORY_SCHEMA]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LogisticVisualVerifier":
        values = np.load(path)
        if "proprio_dim" in values and int(values["proprio_dim"][0]) != ROBOT_PROPRIO_DIM:
            raise ValueError("model was trained with a non-robot-only proprioception stream")
        return cls(
            mean=np.asarray(values["mean"], dtype=np.float32),
            scale=np.asarray(values["scale"], dtype=np.float32),
            weights=np.asarray(values["weights"], dtype=np.float32),
            bias=float(values["bias"][0]),
            threshold=float(values["threshold"][0]),
            abstain_low=float(values["abstain_low"][0]) if "abstain_low" in values else 0.5,
            feature_grid=int(values["feature_grid"][0]),
        )


def _fit_logistic(x: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 0.15, l2: float = 1e-3) -> LogisticVisualVerifier:
    y = np.asarray(y, dtype=np.float32)
    if np.unique(y).size < 2:
        raise ValueError("L1 training split contains only one class")
    mean = x.mean(axis=0).astype(np.float32)
    scale = x.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    z = ((x - mean) / scale).astype(np.float64)
    weights = np.zeros(z.shape[1], dtype=np.float64)
    bias = 0.0
    n = max(1, z.shape[0])
    for _ in range(max(1, int(epochs))):
        logits = np.clip(z @ weights + bias, -40.0, 40.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        error = probs - y
        weights -= float(lr) * ((z.T @ error) / n + float(l2) * weights)
        bias -= float(lr) * float(error.mean())
    return LogisticVisualVerifier(
        mean=mean,
        scale=scale,
        weights=weights.astype(np.float32),
        bias=float(bias),
    )


def _decode_json_attr(value: Any) -> Dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    try:
        return json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}


def _decode_strings(dataset: Any) -> List[str]:
    values = dataset.asstr()[:] if hasattr(dataset, "asstr") else dataset[:]
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def load_hdf5(
    path: str | Path,
    *,
    include_evaluator: bool = True,
    allow_legacy: bool = False,
    seeds: Optional[Iterable[int]] = None,
) -> List[Dict[str, Any]]:
    """Load a robot-only trajectory, optionally joining its evaluator sidecar.

    The main file is checked before any arrays are returned.  In particular,
    an old inline ``labels``/``predicates`` file or a 23-D proprio stream is
    rejected unless ``allow_legacy=True`` is explicitly requested.  Normal L1
    training always uses the v2 robot stream plus the separate evaluator file.
    """

    import h5py

    source = Path(path)
    selected_seeds = None if seeds is None else {int(seed) for seed in seeds}
    episodes: List[Dict[str, Any]] = []
    with h5py.File(source, "r") as handle:
        if handle.attrs.get("invalid_reason"):
            raise ValueError(f"invalidated trajectory: {handle.attrs['invalid_reason']}")
        schema = str(handle.attrs.get("schema", ""))
        if schema != TRAJECTORY_SCHEMA and not allow_legacy:
            raise ValueError(
                f"unsupported trajectory schema {schema!r}; expected {TRAJECTORY_SCHEMA}. "
                "Convert the complete data set instead of mixing old and new files."
            )
        if "episodes" not in handle:
            raise ValueError("trajectory file has no /episodes group")
        declared_fields = handle.attrs.get("trajectory_fields", "")
        if schema == TRAJECTORY_SCHEMA:
            if isinstance(declared_fields, bytes):
                declared_fields = declared_fields.decode("utf-8")
            try:
                fields = tuple(json.loads(str(declared_fields)))
            except (TypeError, ValueError, json.JSONDecodeError):
                fields = ()
            if fields != TRAJECTORY_FIELDS:
                raise ValueError(f"robot trajectory fields are not frozen: {fields!r}")
            if int(handle.attrs.get("proprio_dim", -1)) != ROBOT_PROPRIO_DIM:
                raise ValueError("robot-only trajectory must declare proprio_dim=20")
            if int(handle.attrs.get("action_dim", -1)) != ACTION_DIM:
                raise ValueError("robot-only trajectory must declare action_dim=14")
        for name in sorted(handle["episodes"]):
            group = handle["episodes"][name]
            if selected_seeds is not None and int(group.attrs["seed"]) not in selected_seeds:
                continue
            keys = set(group.keys())
            if schema == TRAJECTORY_SCHEMA and ("labels" in keys or "predicates" in keys):
                raise ValueError(f"privileged evaluator data is embedded in main episode {name}")
            required = {"rgb", "proprio", "actions", "stages"}
            if not required.issubset(keys):
                raise ValueError(f"episode {name} is missing robot stream fields: {sorted(required - keys)}")
            proprio = group["proprio"][:]
            if schema == TRAJECTORY_SCHEMA and (proprio.ndim != 2 or proprio.shape[1] != ROBOT_PROPRIO_DIM):
                raise ValueError(f"episode {name} has non-robot proprio shape {proprio.shape}")
            episodes.append(
                {
                    "name": name,
                    "seed": int(group.attrs["seed"]),
                    "perturbation": str(group.attrs["perturbation"]),
                    "rgb": group["rgb"][:],
                    "proprio": proprio,
                    "actions": group["actions"][:],
                    "stages": _decode_strings(group["stages"]),
                    "metadata": _decode_json_attr(group.attrs.get("metadata", "{}")),
                }
            )

    sidecar = evaluator_path(source)
    if include_evaluator and sidecar.exists():
        with h5py.File(sidecar, "r") as handle:
            side_schema = str(handle.attrs.get("schema", ""))
            if side_schema != EVALUATOR_SCHEMA:
                raise ValueError(f"unsupported evaluator sidecar schema {side_schema!r}")
            for episode in episodes:
                if episode["name"] not in handle.get("episodes", {}):
                    raise ValueError(f"evaluator sidecar is missing {episode['name']}")
                group = handle["episodes"][episode["name"]]
                labels = np.asarray(group["labels"][:], dtype=np.int8)
                if len(labels) != len(episode["stages"]):
                    raise ValueError(f"label/frame length mismatch in {episode['name']}")
                episode["labels"] = labels
                episode["timestamps"] = np.asarray(group.get("timestamps", np.arange(len(labels)))[:], dtype=np.float64)
                episode["predicates"] = [_decode_json_attr(value) for value in _decode_strings(group["predicates"])] if "predicates" in group else []
                episode["failure_events"] = [_decode_json_attr(value) for value in _decode_strings(group["failure_events"])] if "failure_events" in group else []
                episode["failure_event_times"] = np.asarray(
                    group["failure_event_times"][:], dtype=np.float64
                ) if "failure_event_times" in group else np.asarray(
                    [float(event.get("sim_time", np.nan)) for event in episode["failure_events"]],
                    dtype=np.float64,
                )
                episode["evaluator_metadata"] = _decode_json_attr(group.attrs.get("metadata", "{}"))
    elif include_evaluator and allow_legacy:
        # Explicit legacy import path only. New files must use the sidecar.
        with h5py.File(source, "r") as handle:
            for episode in episodes:
                group = handle["episodes"][episode["name"]]
                if "labels" in group:
                    episode["labels"] = group["labels"][:].astype(np.int8)
                if "predicates" in group:
                    episode["predicates"] = [_decode_json_attr(value) for value in _decode_strings(group["predicates"])]
    if include_evaluator:
        missing = [episode["name"] for episode in episodes if "labels" not in episode]
        if missing:
            raise ValueError(
                "evaluator-only labels are required for L1 evaluation; missing sidecar entries: "
                + ", ".join(missing[:3])
            )
    return episodes


def split_by_seed(episodes: Sequence[Mapping[str, Any]], fractions: Tuple[float, float, float] = (0.6, 0.15, 0.25)) -> Dict[str, List[Mapping[str, Any]]]:
    seeds = sorted({int(ep["seed"]) for ep in episodes})
    if not seeds:
        return {"train": [], "dev": [], "test": []}
    # A one-seed smoke dataset cannot provide an independent source-group
    # split.  Keep it usable for a pipeline check and mark the limitation in
    # the report; the full run uses the seed-level split below.
    if len(seeds) == 1:
        return {"train": list(episodes), "dev": [], "test": list(episodes)}
    n = len(seeds)
    n_train = max(1, int(round(n * fractions[0])))
    n_dev = max(0, int(round(n * fractions[1])))
    if n >= 3 and n_train + n_dev >= n:
        n_dev = max(0, n - n_train - 1)
    train_seeds = set(seeds[:n_train])
    dev_seeds = set(seeds[n_train : n_train + n_dev])
    test_seeds = set(seeds[n_train + n_dev :])
    if not test_seeds:
        test_seeds = {seeds[-1]}
        train_seeds.discard(seeds[-1])
    return {
        "train": [ep for ep in episodes if int(ep["seed"]) in train_seeds],
        "dev": [ep for ep in episodes if int(ep["seed"]) in dev_seeds],
        "test": [ep for ep in episodes if int(ep["seed"]) in test_seeds],
    }


def _stack_split(episodes: Sequence[Mapping[str, Any]], grid: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for ep in episodes:
        xs.append(_features(ep["rgb"], ep["proprio"], grid))
        ys.append(np.asarray(ep["labels"], dtype=np.int8))
    if not xs:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int8)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "frames": float(len(y_true)),
        "accuracy": (tp + tn) / max(1.0, len(y_true)),
        "precision_supported": precision,
        "recall_supported": recall,
        "f1_supported": f1,
        "false_pass_rate": fp / max(1.0, fp + tn),
        "false_reject_rate": fn / max(1.0, fn + tp),
    }


def _coverage_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """Report calibrated coverage and risk at episode decision points."""

    if not rows:
        return {
            "decision_points": 0.0,
            "supported_points": 0.0,
            "coverage": 0.0,
            "risk": 0.0,
            "false_pass_rate": 0.0,
            "false_reject_rate": 0.0,
        }
    true_support = np.asarray([int(bool(row["true_support"])) for row in rows], dtype=np.int8)
    release = np.asarray([int(bool(row.get("predicted_support", row.get("predicted_support_at_decision", False)))) for row in rows], dtype=np.int8)
    positives = true_support == 1
    released_positives = positives & (release == 1)
    unsafe = (~positives) & (release == 1)
    coverage = float(np.sum(released_positives) / max(1, np.sum(positives)))
    return {
        "decision_points": float(len(rows)),
        "supported_points": float(np.sum(positives)),
        "coverage": coverage,
        "risk": float(np.sum(unsafe) / max(1, np.sum(release))),
        "false_pass_rate": float(np.sum(unsafe) / max(1, np.sum(~positives))),
        "false_reject_rate": float(np.sum(positives & (release == 0)) / max(1, np.sum(positives))),
    }


def _decision_rows(verifier: LogisticVisualVerifier, episodes: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Return the frames at which a visual gate would make a decision."""

    rows: List[Dict[str, Any]] = []
    for ep in episodes:
        probabilities = verifier.predict_proba(ep["rgb"], ep["proprio"])
        physical = "gate_decision" in ep["stages"]
        for index, stage in enumerate(ep["stages"]):
            if (stage in ("gate_decision", "recovery_decision")) if physical else ("release" in stage or "disturbance_late" in stage or stage in ("r1_recovery_action", "r1_recovery_anchor")):
                row = {
                        "seed": int(ep["seed"]),
                        "perturbation": str(ep["perturbation"]),
                        "stage": stage,
                        "probability": float(probabilities[index]),
                        "true_support": bool(ep["labels"][index]),
                        "frame_index": int(index),
                    }
                if "timestamps" in ep and index < len(ep["timestamps"]):
                    row["sim_time"] = float(ep["timestamps"][index])
                rows.append(row)
    return rows


def _policy_metrics(rows: Sequence[Mapping[str, Any]], release_threshold: float, abstain_low: float = 0.5) -> Dict[str, float]:
    """Measure release, abstention, and hold decisions at gate points."""

    if not rows:
        return {"decision_points": 0.0}
    probabilities = np.asarray([float(row["probability"]) for row in rows], dtype=np.float64)
    labels = np.asarray([int(bool(row["true_support"])) for row in rows], dtype=np.int8)
    release = probabilities >= float(release_threshold)
    abstain = (probabilities >= float(abstain_low)) & ~release
    hold = ~release & ~abstain
    unsafe_release = release & (labels == 0)
    safe_release = release & (labels == 1)
    metrics = {
        "decision_points": float(len(rows)),
        "release_rate": float(np.mean(release)),
        "abstain_rate": float(np.mean(abstain)),
        "hold_rate": float(np.mean(hold)),
        "release_precision": float(np.sum(safe_release) / max(1, np.sum(release))),
        "false_pass_rate": float(np.sum(unsafe_release) / max(1, np.sum(labels == 0))),
        "false_reject_rate": float(np.sum(~release & (labels == 1)) / max(1, np.sum(labels == 1))),
        "unsafe_release_count": float(np.sum(unsafe_release)),
        "abstain_count": float(np.sum(abstain)),
    }
    metrics.update(_coverage_metrics([
        {
            "true_support": int(label),
            "predicted_support": bool(flag),
        }
        for label, flag in zip(labels, release)
    ]))
    return metrics


def calibrate_threshold(
    verifier: LogisticVisualVerifier,
    episodes: Sequence[Mapping[str, Any]],
    target_false_pass: float = 0.02,
    abstain_low: float = 0.5,
) -> Dict[str, Any]:
    """Choose the lowest-risk release threshold on a held-out dev split.

    The operating constraint is a maximum false-pass rate at release-labelled
    decision points.  Among thresholds satisfying that constraint, retain the
    one with the fewest false rejects (equivalently, the most releases).  The
    test split is never used for this choice.
    """

    rows = _decision_rows(verifier, episodes)
    if not rows:
        raise ValueError("cannot calibrate L1 without decision-point rows")
    probabilities = np.asarray([float(row["probability"]) for row in rows], dtype=np.float64)
    # Include observed probabilities so the selected operating point is stable
    # under reruns, plus a small regular grid for a readable threshold.
    candidates = np.unique(
        np.concatenate(
            [
                np.asarray([float(abstain_low), 0.5, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 1.0]),
                probabilities,
                np.linspace(0.5, 1.0, 1001),
            ]
        )
    )
    evaluated: List[Dict[str, Any]] = []
    for threshold in candidates:
        metrics = _policy_metrics(rows, float(threshold), abstain_low=abstain_low)
        evaluated.append({"threshold": float(threshold), **metrics})
    safe = [row for row in evaluated if row["false_pass_rate"] <= float(target_false_pass) + 1e-12]
    if not safe:
        # A value above one is an explicit no-release policy.  It is preferable
        # to silently violating the requested safety constraint.
        selected = {"threshold": 1.000001, **_policy_metrics(rows, 1.000001, abstain_low=abstain_low)}
    else:
        selected = min(safe, key=lambda row: (row["false_reject_rate"], row["threshold"]))
    return {
        "schema": "auditembodied.l1.calibration.v1",
        "target_false_pass": float(target_false_pass),
        "abstain_low": float(abstain_low),
        "selected": selected,
        "default_threshold": float(verifier.threshold),
        "default_metrics": _policy_metrics(rows, verifier.threshold, abstain_low=abstain_low),
        "candidate_count": len(evaluated),
        "candidates": evaluated,
        "rows": rows,
    }


def evaluate_verifier(verifier: LogisticVisualVerifier, episodes: Sequence[Mapping[str, Any]], grid: int = 8) -> Dict[str, Any]:
    x, y = _stack_split(episodes, grid)
    if not len(y):
        return {"frames": 0}
    # Use the already-fitted verifier's public API to keep train and inference
    # feature extraction identical.
    probabilities: List[np.ndarray] = []
    per_perturbation_rows: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for ep in episodes:
        p = verifier.predict_proba(ep["rgb"], ep["proprio"])
        probabilities.append(p)
        pred = (p >= verifier.threshold).astype(np.int8)
        per_perturbation_rows.setdefault(str(ep["perturbation"]), []).append(
            (np.asarray(ep["labels"], dtype=np.int8), pred)
        )
    pred_all = (np.concatenate(probabilities) >= verifier.threshold).astype(np.int8)
    result = _classification_metrics(y, pred_all)
    result["per_perturbation"] = {
        perturbation: _classification_metrics(
            np.concatenate([row[0] for row in rows]), np.concatenate([row[1] for row in rows])
        )
        for perturbation, rows in sorted(per_perturbation_rows.items())
    }
    return result


def visual_gate_report(verifier: LogisticVisualVerifier, episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Measure what a single-frame gate would do at release-labelled frames."""

    rows: List[Dict[str, Any]] = []
    for ep in episodes:
        probabilities = verifier.predict_proba(ep["rgb"], ep["proprio"])
        for index, stage in enumerate(ep["stages"]):
            if "release" in stage or "disturbance_late" in stage or stage in ("r1_recovery_action", "r1_recovery_anchor"):
                predicted_support = bool(probabilities[index] >= verifier.threshold)
                true_support = bool(ep["labels"][index])
                rows.append(
                    {
                        "seed": int(ep["seed"]),
                        "perturbation": str(ep["perturbation"]),
                        "stage": stage,
                        "predicted_support": predicted_support,
                        "true_support": true_support,
                        "probability": float(probabilities[index]),
                    }
                )
    metrics = _classification_metrics(
        np.asarray([int(row["true_support"]) for row in rows], dtype=np.int8),
        np.asarray([int(row["predicted_support"]) for row in rows], dtype=np.int8),
    ) if rows else {"frames": 0}
    metrics["coverage"] = _coverage_metrics(rows)["coverage"] if rows else 0.0
    return {"decision_points": len(rows), "metrics": metrics, "coverage": _coverage_metrics(rows), "rows": rows}


def _replay_release_decision(
    ep: Mapping[str, Any],
    decision_index: int,
    predict_support: Callable[[int, bool], bool],
) -> Dict[str, Any]:
    """Replay gate checks, verifying truth at the first accepted release.

    Recovery observations are consumed only if the initial gate holds. A
    false positive during recovery terminates the branch as an unsafe
    release; a later correct prediction cannot undo that action. This checks
    support at release, not a counterfactual physical handover or a durable
    post-release recovery, neither of which offline frames can establish.
    """

    labels = np.asarray(ep["labels"], dtype=np.int8)
    predicted = bool(predict_support(decision_index, True))
    physical = "gate_decision" in ep["stages"]
    recovery_indices = [
        i for i, stage in enumerate(ep["stages"])
        if i > decision_index and stage in (("recovery_decision",) if physical else (
            "r1_recovery_approach", "r1_recovery_action",
            "r1_recovery_anchor", "r1_recovery_finalize",
        ))
    ]
    observations = 0
    checked_indices = [int(decision_index)]
    release_index = decision_index if predicted else None
    attempted = bool(not predicted and recovery_indices)
    if attempted:
        for index in recovery_indices:
            observations += 1
            checked_indices.append(int(index))
            if predict_support(index, False):
                release_index = index
                break
    if physical:
        # The gate monitors every frame in the consumed 20 Hz prefix, even
        # though release is checked only at the two fixed skill boundaries.
        observations = (checked_indices[-1] - decision_index)
    true_at_release = bool(labels[release_index]) if release_index is not None else None
    recovered = bool(attempted and release_index is not None and true_at_release)
    evaluator_metadata = ep.get("evaluator_metadata", {})
    r1_metadata = evaluator_metadata.get("r1", {}) if isinstance(evaluator_metadata, Mapping) else {}
    return {
        "predicted_support_at_decision": predicted,
        "true_support_at_decision": bool(labels[decision_index]),
        "recovery_requested": bool(not predicted),
        "recovery_attempted": attempted,
        "r1_available": bool(recovery_indices),
        "recorded_r1_success": bool(r1_metadata.get("success", False)) if isinstance(r1_metadata, Mapping) else False,
        "r1_success": recovered,
        "recovered": recovered,
        "release_index": release_index,
        "true_support_at_release": true_at_release,
        "unsupported_release": bool(release_index is not None and not true_at_release),
        "success": bool(release_index is not None and true_at_release),
        "success_scope": "support_at_replayed_release",
        "checked_indices": checked_indices,
        "unsupported_checks": int(sum(not bool(labels[index]) for index in checked_indices)),
        "observations": float((decision_index + 1 if physical else 1) + observations),
        "extra_observations": float(observations),
    }


def visual_closed_loop_report(verifier: LogisticVisualVerifier, episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Replay the recorded trajectory decision points with the learned gate.

    This is an offline closed-loop check: actions and RGB frames come from the
    frozen trajectory, while release / hold / R1 decisions are recomputed from
    the single-frame verifier.  It isolates visual gate errors before running a
    more expensive online environment matrix.
    """

    rows: List[Dict[str, Any]] = []
    for ep in episodes:
        probabilities = verifier.predict_proba(ep["rgb"], ep["proprio"])
        stages = list(ep["stages"])
        perturbation = str(ep["perturbation"])
        if "gate_decision" in stages:
            candidates = [i for i, stage in enumerate(stages) if stage == "gate_decision"]
        elif perturbation == "late":
            candidates = [
                i for i, stage in enumerate(stages) if stage in ("disturbance_late", "disturbance_late_fallback")
            ]
        else:
            candidates = [i for i, stage in enumerate(stages) if stage == "continuous_release"]
        if not candidates:
            continue
        # Use the post-perturbation frame; when a seed needs the explicit
        # safety fallback this is ``disturbance_late_fallback``.
        decision_index = candidates[-1]
        replay = _replay_release_decision(
            ep, decision_index,
            lambda index, allow_stale: bool(probabilities[index] >= verifier.threshold),
        )
        predicted_support = replay["predicted_support_at_decision"]
        true_support = replay["true_support_at_decision"]
        final_support = bool(ep.get("labels", [0])[-1]) if len(ep.get("labels", [])) else False
        # The visual decision and the physical outcome are kept separate. This
        # lets the report say whether a failure was caused by the gate or by R1
        # execution, instead of charging both to the verifier.
        visual_error = bool(predicted_support != true_support)
        execution_failure = bool((not visual_error) and not final_support)
        rows.append(
            {
                "seed": int(ep["seed"]),
                "perturbation": perturbation,
                **replay,
                "r1_attempted": replay["recovery_attempted"],
                "final_support": final_support,
                "visual_error": visual_error,
                "execution_failure": execution_failure,
                "probability": float(probabilities[decision_index]),
                "decision_index": int(decision_index),
                "decision_time": (
                    float(ep["timestamps"][decision_index])
                    if "timestamps" in ep and len(ep["timestamps"]) > decision_index
                    else None
                ),
            }
        )
    by_perturbation: Dict[str, Dict[str, float]] = {}
    for perturbation in sorted({row["perturbation"] for row in rows}):
        subset = [row for row in rows if row["perturbation"] == perturbation]
        by_perturbation[perturbation] = {
            "episodes": float(len(subset)),
            "success_rate": float(np.mean([row["success"] for row in subset])) if subset else 0.0,
            "false_pass_rate": float(
                np.mean([row["unsupported_release"] for row in subset])
            ) if subset else 0.0,
            "recovery_rate": float(np.mean([row["recovered"] for row in subset])) if subset else 0.0,
            "visual_error_rate": float(np.mean([row["visual_error"] for row in subset])) if subset else 0.0,
            "execution_failure_rate": float(np.mean([row["execution_failure"] for row in subset])) if subset else 0.0,
        }
    coverage = _coverage_metrics(
        [
            {
                "true_support": row["true_support_at_decision"],
                "predicted_support": row["predicted_support_at_decision"],
            }
            for row in rows
        ]
    )
    return {
        "evaluation_mode": "offline_gate_replay",
        "success_scope": "support_at_replayed_release",
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])) if rows else 0.0,
        "false_pass_rate": float(
            np.mean([row["unsupported_release"] for row in rows])
        ) if rows else 0.0,
        "visual_error_rate": float(np.mean([row["visual_error"] for row in rows])) if rows else 0.0,
        "execution_failure_rate": float(np.mean([row["execution_failure"] for row in rows])) if rows else 0.0,
        "coverage": coverage,
        "by_perturbation": by_perturbation,
        "rows": rows,
    }


def online_visual_benchmark(
    verifier: LogisticVisualVerifier,
    seeds: Iterable[int] = range(3),
    output: str | Path = "outputs/l1-online-pilot.json",
) -> Dict[str, Any]:
    """Evaluate L1 with physical R1, giver release and a one-second hold."""
    from .temporal import online_temporal_benchmark
    return online_temporal_benchmark(
        verifier, "frame", {"threshold": verifier.threshold}, seeds=seeds, output=output,
    )


def online_offline_gap(
    offline: Mapping[str, Any],
    online: Mapping[str, Any],
) -> Dict[str, float]:
    """Compare the same gate metrics in replayed and live observations."""

    offline_success = float(offline.get("success_rate", 0.0))
    offline_false_pass = float(offline.get("false_pass_rate", 0.0))
    online_success = float(online.get("success_rate", 0.0))
    online_false_pass = float(online.get("false_pass_rate", 0.0))
    return {
        "success_rate_gap_online_minus_offline": online_success - offline_success,
        "false_pass_rate_gap_online_minus_offline": online_false_pass - offline_false_pass,
        "absolute_success_rate_gap": abs(online_success - offline_success),
        "absolute_false_pass_rate_gap": abs(online_false_pass - offline_false_pass),
    }


def train_and_evaluate(dataset: str | Path, model_output: str | Path) -> Dict[str, Any]:
    episodes = load_hdf5(dataset)
    splits = split_by_seed(episodes)
    x_train, y_train = _stack_split(splits["train"])
    verifier = _fit_logistic(x_train, y_train)
    verifier.save(model_output)
    report = {
        # Keep the report name stable for downstream readers; the nested
        # protocol block below records that this run used the v2 robot stream.
        "schema": "auditembodied.l1.single_frame.v1",
        "report_version": 2,
        "dataset": str(dataset),
        "model": str(model_output),
        "protocol": {
            "trajectory_schema": TRAJECTORY_SCHEMA,
            "evaluator_schema": EVALUATOR_SCHEMA,
            "trajectory_fields": list(TRAJECTORY_FIELDS),
            "proprio_dim": ROBOT_PROPRIO_DIM,
            "action_dim": ACTION_DIM,
            "labels_source": "evaluator-only sidecar",
        },
        "feature_dim": int(x_train.shape[1]) if x_train.ndim == 2 else 0,
        "split_seeds": {
            name: sorted({int(ep["seed"]) for ep in value}) for name, value in splits.items()
        },
        "train": evaluate_verifier(verifier, splits["train"]),
        "dev": evaluate_verifier(verifier, splits["dev"]),
        "test": evaluate_verifier(verifier, splits["test"]),
        "visual_gate": visual_gate_report(verifier, splits["test"]),
        "visual_closed_loop": visual_closed_loop_report(verifier, splits["test"]),
    }
    report_path = Path(model_output).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path)
    return report


def calibrate_and_evaluate(
    dataset: str | Path,
    model_input: str | Path,
    model_output: str | Path,
    target_false_pass: float = 0.02,
    abstain_low: float = 0.5,
) -> Dict[str, Any]:
    """Calibrate a frozen L1 head on dev seeds and evaluate on test seeds."""

    episodes = load_hdf5(dataset)
    splits = split_by_seed(episodes)
    verifier = LogisticVisualVerifier.load(model_input)
    calibration = calibrate_threshold(
        verifier,
        splits["dev"],
        target_false_pass=target_false_pass,
        abstain_low=abstain_low,
    )
    verifier.threshold = float(calibration["selected"]["threshold"])
    verifier.abstain_low = float(abstain_low)
    verifier.save(model_output)
    report = {
        "schema": "auditembodied.l1.calibrated.v1",
        "report_version": 2,
        "dataset": str(dataset),
        "model_input": str(model_input),
        "model": str(model_output),
        "split_seeds": {
            name: sorted({int(ep["seed"]) for ep in value}) for name, value in splits.items()
        },
        "protocol": {
            "trajectory_schema": TRAJECTORY_SCHEMA,
            "evaluator_schema": EVALUATOR_SCHEMA,
            "proprio_dim": ROBOT_PROPRIO_DIM,
            "labels_source": "evaluator-only sidecar",
        },
        "calibration": {
            key: value for key, value in calibration.items() if key not in {"candidates", "rows"}
        },
        "dev": evaluate_verifier(verifier, splits["dev"]),
        "test": evaluate_verifier(verifier, splits["test"]),
        "test_visual_gate": visual_gate_report(verifier, splits["test"]),
        "test_visual_closed_loop": visual_closed_loop_report(verifier, splits["test"]),
        "test_policy": _policy_metrics(_decision_rows(verifier, splits["test"]), verifier.threshold, abstain_low),
    }
    closed_loop = report["test_visual_closed_loop"]
    report["error_decomposition"] = {
        "visual_error_rate": float(closed_loop.get("visual_error_rate", 0.0)),
        "execution_failure_rate": float(closed_loop.get("execution_failure_rate", 0.0)),
        "visual_error_count": int(sum(bool(row.get("visual_error", False)) for row in closed_loop.get("rows", []))),
        "execution_failure_count": int(sum(bool(row.get("execution_failure", False)) for row in closed_loop.get("rows", []))),
    }
    report_path = Path(model_output).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path)
    return report


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AuditEmbodied L1 single-frame visual verifier")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-output", default="outputs/l1_visual_verifier.npz")
    return parser.parse_args(argv)


def calibration_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate an AuditEmbodied L1 release threshold")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-input", required=True)
    parser.add_argument("--model-output", default="outputs/l1_visual_verifier_calibrated.npz")
    parser.add_argument("--target-false-pass", type=float, default=0.02)
    parser.add_argument("--abstain-low", type=float, default=0.5)
    args = parser.parse_args(argv)
    report = calibrate_and_evaluate(
        args.dataset,
        args.model_input,
        args.model_output,
        target_false_pass=float(args.target_false_pass),
        abstain_low=float(args.abstain_low),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    report = train_and_evaluate(args.dataset, args.model_output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
