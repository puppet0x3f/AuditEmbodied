"""Small L2 temporal baselines over the frozen L1 feature stream.

The first L2 pass deliberately keeps the same RGB+robot-proprio feature
front-end as L1.  It compares a single-frame gate with two causal controls:
fixed evidence TTL and a short-history change detector.  This is enough to
measure whether temporal freshness helps before introducing a larger
Transformer or a new visual backbone.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .l1 import (
    LogisticVisualVerifier,
    _features,
    _replay_release_decision,
    load_hdf5,
    split_by_seed,
)
from .continuous import EVALUATOR_SCHEMA, ROBOT_PROPRIO_DIM, TRAJECTORY_SCHEMA


BASELINES = ("frame", "ttl", "change", "temporal8", "temporal_no_change")
CAUSAL_WINDOW = 8


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def _temporal_vector(
    probabilities: np.ndarray,
    features: np.ndarray,
    index: int,
    window: int = CAUSAL_WINDOW,
) -> np.ndarray:
    """Build a causal, fixed-size sequence vector for one time index.

    The vector contains the last ``window`` single-frame scores and feature
    change magnitudes, left-padded with the first observation.  Summary terms
    make the small head sensitive to persistence and recent motion without
    giving it any future frame or evaluator value.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    index = int(index)
    window = max(1, int(window))
    start = max(0, index - window + 1)
    indices = list(range(start, index + 1))
    if not indices:
        indices = [0]
    pad = [indices[0]] * max(0, window - len(indices))
    indices = pad + indices[-window:]
    prob_hist = np.asarray([probabilities[i] for i in indices], dtype=np.float64)
    deltas: List[float] = []
    for i in indices:
        if i <= 0:
            deltas.append(0.0)
        else:
            deltas.append(float(np.mean(np.abs(features[i] - features[i - 1]))))
    delta_hist = np.asarray(deltas, dtype=np.float64)
    recent = prob_hist[-min(3, window) :]
    summary = np.asarray(
        [
            float(prob_hist[-1]),
            float(prob_hist.mean()),
            float(prob_hist.max()),
            float(prob_hist.min()),
            float(recent[-1] - recent[0]) if len(recent) > 1 else 0.0,
            float(delta_hist[-1]),
            float(delta_hist.mean()),
        ],
        dtype=np.float64,
    )
    return np.concatenate([prob_hist, delta_hist, summary], axis=0)


def _fit_binary_head(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = 250,
    lr: float = 0.10,
    l2: float = 1e-3,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Fit a deterministic logistic head and return normalization too."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or not len(x):
        raise ValueError("temporal head received no examples")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = (x - mean) / scale
    weights = np.zeros(z.shape[1], dtype=np.float64)
    bias = 0.0
    n = float(max(1, len(z)))
    for _ in range(max(1, int(epochs))):
        probabilities = _sigmoid(z @ weights + bias)
        error = probabilities - y
        weights -= float(lr) * ((z.T @ error) / n + float(l2) * weights)
        bias -= float(lr) * float(error.mean())
    return weights.astype(np.float32), float(bias), mean.astype(np.float32), scale.astype(np.float32)


def _fit_head_on_scaled(
    scaled: np.ndarray,
    y: np.ndarray,
    epochs: int = 250,
    lr: float = 0.10,
    l2: float = 1e-3,
) -> Tuple[np.ndarray, float]:
    """Fit a head when the caller already fixed feature normalization."""

    scaled = np.asarray(scaled, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.zeros(scaled.shape[1], dtype=np.float64)
    bias = 0.0
    n = float(max(1, len(scaled)))
    for _ in range(max(1, int(epochs))):
        probabilities = _sigmoid(scaled @ weights + bias)
        error = probabilities - y
        weights -= float(lr) * ((scaled.T @ error) / n + float(l2) * weights)
        bias -= float(lr) * float(error.mean())
    return weights.astype(np.float32), float(bias)


@dataclass
class CausalTemporalVerifier:
    """Lightweight causal state/change model used for the first L2 pass."""

    window: int
    state_weights: np.ndarray
    state_bias: float
    change_weights: np.ndarray
    change_bias: float
    vector_mean: np.ndarray
    vector_scale: np.ndarray
    state_threshold: float = 0.5
    change_threshold: float = 0.5

    @property
    def feature_dim(self) -> int:
        return int(self.state_weights.shape[0])

    def _scaled_vectors(self, probabilities: np.ndarray, features: np.ndarray) -> np.ndarray:
        return np.stack(
            [_temporal_vector(probabilities, features, i, self.window) for i in range(len(probabilities))],
            axis=0,
        )

    def predict_sequence(
        self,
        probabilities: np.ndarray,
        features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        vectors = self._scaled_vectors(probabilities, features)
        scaled = (vectors - self.vector_mean) / self.vector_scale
        state = _sigmoid(scaled @ self.state_weights + float(self.state_bias))
        change = _sigmoid(scaled @ self.change_weights + float(self.change_bias))
        return state.astype(np.float64), change.astype(np.float64)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            schema=np.asarray(["auditembodied.l2.causal_temporal.v1"]),
            window=np.asarray([self.window], dtype=np.int64),
            state_weights=self.state_weights,
            state_bias=np.asarray([self.state_bias], dtype=np.float32),
            change_weights=self.change_weights,
            change_bias=np.asarray([self.change_bias], dtype=np.float32),
            vector_mean=self.vector_mean,
            vector_scale=self.vector_scale,
            state_threshold=np.asarray([self.state_threshold], dtype=np.float32),
            change_threshold=np.asarray([self.change_threshold], dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CausalTemporalVerifier":
        values = np.load(path)
        if "schema" in values and str(values["schema"][0]) != "auditembodied.l2.causal_temporal.v1":
            raise ValueError("unsupported causal temporal model schema")
        return cls(
            window=int(values["window"][0]),
            state_weights=np.asarray(values["state_weights"], dtype=np.float32),
            state_bias=float(values["state_bias"][0]),
            change_weights=np.asarray(values["change_weights"], dtype=np.float32),
            change_bias=float(values["change_bias"][0]),
            vector_mean=np.asarray(values["vector_mean"], dtype=np.float32),
            vector_scale=np.asarray(values["vector_scale"], dtype=np.float32),
            state_threshold=float(values["state_threshold"][0]) if "state_threshold" in values else 0.5,
            change_threshold=float(values["change_threshold"][0]) if "change_threshold" in values else 0.5,
        )


def _temporal_examples(
    verifier: LogisticVisualVerifier,
    episodes: Sequence[Mapping[str, Any]],
    window: int = CAUSAL_WINDOW,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors: List[np.ndarray] = []
    state_targets: List[int] = []
    change_targets: List[int] = []
    for ep in episodes:
        probabilities, features = _probs_and_features(verifier, ep)
        labels = np.asarray(ep["labels"], dtype=np.int8)
        for index in range(len(labels)):
            vectors.append(_temporal_vector(probabilities, features, index, window))
            state_targets.append(int(labels[index]))
            change_targets.append(int(index > 0 and labels[index] != labels[index - 1]))
    if not vectors:
        raise ValueError("cannot fit an L2 model without trajectory frames")
    return np.asarray(vectors, dtype=np.float64), np.asarray(state_targets, dtype=np.int8), np.asarray(change_targets, dtype=np.int8)


def fit_causal_temporal(
    verifier: LogisticVisualVerifier,
    episodes: Sequence[Mapping[str, Any]],
    window: int = CAUSAL_WINDOW,
) -> CausalTemporalVerifier:
    """Train state and change heads using only train-seed observations."""

    vectors, state_targets, change_targets = _temporal_examples(verifier, episodes, window)
    state_weights, state_bias, mean, scale = _fit_binary_head(vectors, state_targets)
    # Both heads use the same fixed normalization, so predictions remain
    # causal and cheap to run online.
    scaled = (vectors - mean) / scale
    change_weights, change_bias = _fit_head_on_scaled(scaled, change_targets)
    return CausalTemporalVerifier(
        window=int(window),
        state_weights=state_weights,
        state_bias=state_bias,
        change_weights=change_weights,
        change_bias=change_bias,
        vector_mean=mean,
        vector_scale=scale,
    )


def _binary_summary(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    target = np.asarray(target, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=np.int8)
    tp = float(np.sum((target == 1) & (prediction == 1)))
    tn = float(np.sum((target == 0) & (prediction == 0)))
    fp = float(np.sum((target == 0) & (prediction == 1)))
    fn = float(np.sum((target == 1) & (prediction == 0)))
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    return {
        "frames": float(len(target)),
        "accuracy": float((tp + tn) / max(1.0, len(target))),
        "precision": precision,
        "recall": recall,
        "f1": float(2.0 * precision * recall / max(1e-12, precision + recall)),
        "false_positive_rate": float(fp / max(1.0, fp + tn)),
    }


def temporal_head_diagnostics(
    verifier: LogisticVisualVerifier,
    temporal_model: CausalTemporalVerifier,
    episodes: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    state_targets: List[int] = []
    state_predictions: List[int] = []
    change_targets: List[int] = []
    change_predictions: List[int] = []
    for ep in episodes:
        probabilities, features = _probs_and_features(verifier, ep)
        state, change = temporal_model.predict_sequence(probabilities, features)
        labels = np.asarray(ep["labels"], dtype=np.int8)
        state_targets.extend(labels.tolist())
        state_predictions.extend((state >= temporal_model.state_threshold).astype(np.int8).tolist())
        change_targets.extend([int(i > 0 and labels[i] != labels[i - 1]) for i in range(len(labels))])
        change_predictions.extend((change >= temporal_model.change_threshold).astype(np.int8).tolist())
    return {
        "state_head": _binary_summary(np.asarray(state_targets), np.asarray(state_predictions)),
        "change_head": _binary_summary(np.asarray(change_targets), np.asarray(change_predictions)),
    }


def _decision_index(ep: Mapping[str, Any]) -> Optional[int]:
    stages = list(ep["stages"])
    if "gate_decision" in stages:
        candidates = [i for i, stage in enumerate(stages) if stage == "gate_decision"]
    elif str(ep["perturbation"]) == "late":
        candidates = [
            i for i, stage in enumerate(stages)
            if stage in ("disturbance_late", "disturbance_late_fallback")
        ]
    else:
        candidates = [i for i, stage in enumerate(stages) if stage == "continuous_release"]
    return candidates[-1] if candidates else None


def _probs_and_features(
    verifier: LogisticVisualVerifier, ep: Mapping[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    probabilities = verifier.predict_proba(ep["rgb"], ep["proprio"])
    raw = _features(ep["rgb"], ep["proprio"], verifier.feature_grid)
    scaled = (raw - verifier.mean) / verifier.scale
    return probabilities.astype(np.float64), scaled.astype(np.float64)


def _confirmation_index(
    probabilities: np.ndarray,
    stages: Sequence[str],
    before: int,
    threshold: float,
) -> Optional[int]:
    preferred = {
        "receiver_grasp",
        "continuous_handover",
        "continuous_release",
        "r1_recovery_action",
        "r1_recovery_anchor",
    }
    candidates = [
        i for i in range(max(0, before))
        if probabilities[i] >= threshold and stages[i] in preferred
    ]
    if candidates:
        return candidates[-1]
    candidates = [i for i in range(max(0, before)) if probabilities[i] >= threshold]
    return candidates[-1] if candidates else None


def _change_score(features: np.ndarray, current: int, reference: int) -> float:
    if current == reference:
        return 0.0
    return float(np.mean(np.abs(features[current] - features[reference])))


def _predict_index(
    probabilities: np.ndarray,
    features: np.ndarray,
    stages: Sequence[str],
    index: int,
    verifier: LogisticVisualVerifier,
    policy: str,
    params: Mapping[str, Any],
    allow_stale_ttl: bool = True,
) -> bool:
    threshold = float(verifier.threshold)
    current_probability = float(probabilities[index])
    if policy == "frame":
        return current_probability >= threshold
    if policy == "temporal_no_change":
        window = max(1, int(params.get("window", 3)))
        start = max(0, index - window + 1)
        return float(np.mean(probabilities[start : index + 1])) >= threshold
    if policy == "ttl":
        if current_probability >= threshold:
            return True
        if not allow_stale_ttl:
            return False
        confirmation = _confirmation_index(probabilities, stages, index, threshold)
        ttl = int(params.get("ttl", 0))
        return confirmation is not None and (index - confirmation) <= ttl
    if policy == "change":
        if current_probability < threshold:
            return False
        confirmation = _confirmation_index(probabilities, stages, index, threshold)
        if confirmation is None:
            return True
        delta = float(params.get("delta", 0.5))
        return _change_score(features, index, confirmation) <= delta
    raise ValueError(f"unknown L2 policy: {policy}")


def _temporal_confirmation_index(
    state_probabilities: np.ndarray,
    change_probabilities: np.ndarray,
    before: int,
    state_threshold: float,
    change_threshold: float,
) -> Optional[int]:
    candidates = [
        i
        for i in range(max(0, int(before)))
        if state_probabilities[i] >= state_threshold
        and change_probabilities[i] < change_threshold
    ]
    return candidates[-1] if candidates else None


def _temporal_predict_index(
    state_probabilities: np.ndarray,
    change_probabilities: np.ndarray,
    index: int,
    params: Mapping[str, Any],
    allow_stale: bool = True,
) -> bool:
    state_threshold = float(params.get("state_threshold", 0.5))
    change_threshold = float(params.get("change_threshold", 0.5))
    index = int(index)
    if state_probabilities[index] < state_threshold:
        return False
    confirmation = _temporal_confirmation_index(
        state_probabilities,
        change_probabilities,
        index,
        state_threshold,
        change_threshold,
    )
    if confirmation is None:
        return not (change_probabilities[index] >= change_threshold)
    if not allow_stale and index > confirmation and change_probabilities[index] >= change_threshold:
        return False
    # A change event invalidates the old confirmation until a new stable state
    # is observed.  This is the causal version-currentness rule.
    if index > confirmation and np.any(change_probabilities[confirmation + 1 : index + 1] >= change_threshold):
        return False
    return True


def _temporal_episode_policy_row(
    verifier: LogisticVisualVerifier,
    temporal_model: CausalTemporalVerifier,
    ep: Mapping[str, Any],
    params: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    decision = _decision_index(ep)
    if decision is None:
        return None
    probabilities, features = _probs_and_features(verifier, ep)
    state_probabilities, change_probabilities = temporal_model.predict_sequence(probabilities, features)
    replay = _replay_release_decision(
        ep, decision,
        lambda index, allow_stale: _temporal_predict_index(
            state_probabilities, change_probabilities, index, params, allow_stale=allow_stale
        ),
    )
    return {
        "seed": int(ep["seed"]),
        "perturbation": str(ep["perturbation"]),
        "decision_index": int(decision),
        "probability": float(state_probabilities[decision]),
        "change_probability": float(change_probabilities[decision]),
        **replay,
    }


def _episode_policy_row(
    verifier: LogisticVisualVerifier,
    ep: Mapping[str, Any],
    policy: str,
    params: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    decision = _decision_index(ep)
    if decision is None:
        return None
    probabilities, features = _probs_and_features(verifier, ep)
    stages = list(ep["stages"])
    replay = _replay_release_decision(
        ep, decision,
        lambda index, allow_stale: _predict_index(
            probabilities, features, stages, index, verifier, policy, params,
            allow_stale_ttl=allow_stale,
        ),
    )
    return {
        "seed": int(ep["seed"]),
        "perturbation": str(ep["perturbation"]),
        "decision_index": int(decision),
        "probability": float(probabilities[decision]),
        **replay,
    }


def _metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "false_pass_rate": 0.0,
            "false_reject_rate": 0.0,
            "recovery_rate": 0.0,
            "mean_observations": 0.0,
            "mean_extra_observations": 0.0,
            "r1_success_rate": 0.0,
            "late_false_pass_rate": 0.0,
        }
    late_rows = [row for row in rows if row.get("perturbation") == "late"]
    attempted = [row for row in rows if row.get("r1_available", False) and row.get("recovery_attempted", False)]
    return {
        "episodes": len(rows),
        "success_rate": float(np.mean([bool(row["success"]) for row in rows])),
        "false_pass_rate": float(
            np.mean([
                bool(row["unsupported_release"])
                for row in rows
            ])
        ),
        "false_reject_rate": float(
            np.mean([
                not bool(row["predicted_support_at_decision"])
                and bool(row["true_support_at_decision"])
                for row in rows
            ])
        ),
        "recovery_rate": float(np.mean([bool(row["recovered"]) for row in rows])),
        "mean_observations": float(np.mean([float(row["observations"]) for row in rows])),
        "mean_extra_observations": float(np.mean([max(0.0, float(row["observations"]) - 1.0) for row in rows])),
        "r1_success_rate": float(np.mean([bool(row.get("r1_success", False)) for row in attempted])) if attempted else 0.0,
        "late_false_pass_rate": float(
            np.mean([
                bool(row["unsupported_release"])
                for row in late_rows
            ])
        ) if late_rows else 0.0,
    }


def _true_failure_index(ep: Mapping[str, Any]) -> Optional[int]:
    events = ep.get("failure_events", [])
    if events:
        indices = [int(event["frame_index"]) for event in events if "frame_index" in event]
        if indices:
            return min(indices)
    labels = np.asarray(ep.get("labels", []), dtype=np.int8)
    transitions = np.flatnonzero((labels[:-1] == 1) & (labels[1:] == 0)) + 1 if len(labels) > 1 else []
    return int(transitions[0]) if len(transitions) else None


def _policy_sequence(
    verifier: LogisticVisualVerifier,
    ep: Mapping[str, Any],
    policy: str,
    params: Mapping[str, Any],
    temporal_model: Optional[CausalTemporalVerifier] = None,
) -> np.ndarray:
    probabilities, features = _probs_and_features(verifier, ep)
    if policy == "temporal8":
        if temporal_model is None:
            raise ValueError("temporal8 evaluation requires a fitted temporal model")
        state, change = temporal_model.predict_sequence(probabilities, features)
        return np.asarray([
            _temporal_predict_index(state, change, i, params)
            for i in range(len(probabilities))
        ], dtype=bool)
    stages = list(ep["stages"])
    return np.asarray([
        _predict_index(probabilities, features, stages, i, verifier, policy, params)
        for i in range(len(probabilities))
    ], dtype=bool)


def _detection_metrics(
    verifier: LogisticVisualVerifier,
    episodes: Sequence[Mapping[str, Any]],
    policy: str,
    params: Mapping[str, Any],
    temporal_model: Optional[CausalTemporalVerifier] = None,
) -> Dict[str, Any]:
    delays: List[float] = []
    time_delays: List[float] = []
    missed = 0
    for ep in episodes:
        failure = _true_failure_index(ep)
        if failure is None:
            continue
        sequence = _policy_sequence(verifier, ep, policy, params, temporal_model)
        detected = np.flatnonzero(~sequence[max(0, failure) :])
        if not len(detected):
            missed += 1
            continue
        detected_index = int(failure + detected[0])
        delays.append(float(detected_index - failure))
        timestamps = np.asarray(ep.get("timestamps", []), dtype=np.float64)
        if len(timestamps) > detected_index and len(timestamps) > failure:
            time_delays.append(float(timestamps[detected_index] - timestamps[failure]))
    total = len(delays) + int(missed)
    return {
        "events": total,
        "detected_events": len(delays),
        "miss_rate": float(missed / max(1, total)),
        "mean_delay_frames": float(np.mean(delays)) if delays else None,
        "mean_delay_seconds": float(np.mean(time_delays)) if time_delays else None,
    }


def _change_head_detection_metrics(
    verifier: LogisticVisualVerifier,
    temporal_model: CausalTemporalVerifier,
    episodes: Sequence[Mapping[str, Any]],
    change_threshold: float,
) -> Dict[str, Any]:
    delays: List[float] = []
    misses = 0
    total = 0
    for ep in episodes:
        failure = _true_failure_index(ep)
        if failure is None:
            continue
        total += 1
        probabilities, features = _probs_and_features(verifier, ep)
        _, change = temporal_model.predict_sequence(probabilities, features)
        detected = np.flatnonzero(change[max(0, failure) :] >= float(change_threshold))
        if not len(detected):
            misses += 1
        else:
            delays.append(float(detected[0]))
    return {
        "events": total,
        "detected_events": len(delays),
        "miss_rate": float(misses / max(1, total)),
        "mean_delay_frames": float(np.mean(delays)) if delays else None,
    }


def _seed_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    n_bootstrap: int = 2000,
    seed: int = 20261015,
) -> Dict[str, Any]:
    """Bootstrap one metric with the seed, rather than frame, as the unit."""

    grouped: Dict[int, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["seed"]), []).append(row)
    seeds = sorted(grouped)
    if not seeds:
        return {"estimate": 0.0, "ci95": [0.0, 0.0], "seeds": 0}

    def value(selected: Sequence[int]) -> float:
        values: List[float] = []
        for index in selected:
            values.extend(float(row.get(value_key, 0.0)) for row in grouped[seeds[index]])
        return float(np.mean(values)) if values else 0.0

    estimate = value(list(range(len(seeds))))
    rng = np.random.default_rng(int(seed))
    samples = np.asarray(
        [value(rng.integers(0, len(seeds), size=len(seeds)).tolist()) for _ in range(max(100, int(n_bootstrap)))],
        dtype=np.float64,
    )
    return {
        "estimate": estimate,
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "seeds": len(seeds),
    }


def _paired_seed_difference(
    rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    value_key: str,
    n_bootstrap: int = 2000,
    seed: int = 20261016,
) -> Dict[str, Any]:
    """Paired seed bootstrap for a policy minus the frame baseline."""

    def by_seed(items: Sequence[Mapping[str, Any]]) -> Dict[int, float]:
        grouped: Dict[int, List[float]] = {}
        for row in items:
            grouped.setdefault(int(row["seed"]), []).append(float(row.get(value_key, 0.0)))
        return {key: float(np.mean(values)) for key, values in grouped.items()}

    left = by_seed(rows)
    right = by_seed(reference_rows)
    seeds = sorted(set(left).intersection(right))
    if not seeds:
        return {"estimate": 0.0, "ci95": [0.0, 0.0], "seeds": 0}
    differences = np.asarray([left[key] - right[key] for key in seeds], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    samples = np.asarray(
        [float(np.mean(differences[rng.integers(0, len(differences), size=len(differences))])) for _ in range(max(100, int(n_bootstrap)))],
        dtype=np.float64,
    )
    return {
        "estimate": float(np.mean(differences)),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "seeds": len(seeds),
    }


def evaluate_policy(
    verifier: LogisticVisualVerifier,
    episodes: Sequence[Mapping[str, Any]],
    policy: str,
    params: Optional[Mapping[str, Any]] = None,
    temporal_model: Optional[CausalTemporalVerifier] = None,
) -> Dict[str, Any]:
    params = params or {}
    if policy == "temporal8":
        if temporal_model is None:
            raise ValueError("temporal8 evaluation requires a fitted temporal model")
        rows = [
            row for ep in episodes
            if (row := _temporal_episode_policy_row(verifier, temporal_model, ep, params)) is not None
        ]
    else:
        rows = [
            row for ep in episodes
            if (row := _episode_policy_row(verifier, ep, policy, params)) is not None
        ]
    by_perturbation = {
        perturbation: _metrics([row for row in rows if row["perturbation"] == perturbation])
        for perturbation in sorted({row["perturbation"] for row in rows})
    }
    result = {
        "policy": policy, "params": dict(params), **_metrics(rows),
        "by_perturbation": by_perturbation, "rows": rows,
        "evaluation_mode": "offline_gate_replay",
        "success_scope": "support_at_replayed_release",
    }
    result["detection"] = _detection_metrics(verifier, episodes, policy, params, temporal_model)
    if policy == "temporal8" and temporal_model is not None:
        result["change_head_detection"] = _change_head_detection_metrics(
            verifier,
            temporal_model,
            episodes,
            float(params.get("change_threshold", 0.5)),
        )
    return result


def _tune_ttl(verifier: LogisticVisualVerifier, episodes: Sequence[Mapping[str, Any]], target_fp: float) -> Dict[str, Any]:
    candidates = []
    for ttl in (0, 1, 2, 3, 4, 6, 8, 12, 20):
        result = evaluate_policy(verifier, episodes, "ttl", {"ttl": ttl})
        if result["false_pass_rate"] <= target_fp + 1e-12:
            candidates.append(result)
    if not candidates:
        return evaluate_policy(verifier, episodes, "ttl", {"ttl": 0})
    return max(candidates, key=lambda result: (result["success_rate"], -result["mean_observations"], -result["params"]["ttl"]))


def _change_candidates(verifier: LogisticVisualVerifier, episodes: Sequence[Mapping[str, Any]]) -> List[float]:
    scores: List[float] = []
    for ep in episodes:
        decision = _decision_index(ep)
        if decision is None:
            continue
        probabilities, features = _probs_and_features(verifier, ep)
        confirmation = _confirmation_index(probabilities, ep["stages"], decision, verifier.threshold)
        if confirmation is not None:
            scores.append(_change_score(features, decision, confirmation))
    if not scores:
        return [0.5]
    return sorted(set([0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0] + [float(x) for x in np.quantile(scores, np.linspace(0.05, 0.95, 19))]))


def _tune_change(verifier: LogisticVisualVerifier, episodes: Sequence[Mapping[str, Any]], target_fp: float) -> Dict[str, Any]:
    candidates = []
    for delta in _change_candidates(verifier, episodes):
        result = evaluate_policy(verifier, episodes, "change", {"delta": delta})
        if result["false_pass_rate"] <= target_fp + 1e-12:
            candidates.append(result)
    if not candidates:
        return evaluate_policy(verifier, episodes, "change", {"delta": 0.5})
    return max(candidates, key=lambda result: (result["success_rate"], -result["mean_observations"], -result["params"]["delta"]))


def _tune_temporal(
    verifier: LogisticVisualVerifier,
    temporal_model: CausalTemporalVerifier,
    episodes: Sequence[Mapping[str, Any]],
    target_fp: float,
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    state_thresholds = (0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99, 0.995)
    change_thresholds = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
    for state_threshold in state_thresholds:
        for change_threshold in change_thresholds:
            params = {
                "state_threshold": float(state_threshold),
                "change_threshold": float(change_threshold),
            }
            result = evaluate_policy(
                verifier,
                episodes,
                "temporal8",
                params,
                temporal_model=temporal_model,
            )
            if result["false_pass_rate"] <= float(target_fp) + 1e-12:
                candidates.append(result)
    if not candidates:
        params = {"state_threshold": 0.995, "change_threshold": 0.5}
        return evaluate_policy(verifier, episodes, "temporal8", params, temporal_model=temporal_model)
    return max(
        candidates,
        key=lambda result: (
            result["success_rate"],
            -result["mean_observations"],
            -result["params"]["state_threshold"],
            -result["params"]["change_threshold"],
        ),
    )


def _bootstrap_report(
    results: Mapping[str, Mapping[str, Any]],
    frame_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    metrics = ("success", "predicted_support_at_decision", "recovered")
    report: Dict[str, Any] = {}
    for policy, result in results.items():
        rows = result.get("rows", [])
        report[policy] = {
            "success_rate": _seed_bootstrap(rows, "success"),
            "false_pass_rate": _seed_bootstrap(
                [
                    {
                        **row,
                        "false_pass": bool(row["unsupported_release"]),
                    }
                    for row in rows
                ],
                "false_pass",
            ),
            "late_false_pass_rate": _seed_bootstrap(
                [
                    {
                        **row,
                        "late_false_pass": bool(row["unsupported_release"]),
                    }
                    for row in rows
                    if row.get("perturbation") == "late"
                ],
                "late_false_pass",
            ),
        }
    return report


def run_l2(dataset: str | Path, model: str | Path, target_false_pass: float = 0.02) -> Dict[str, Any]:
    episodes = load_hdf5(dataset)
    splits = split_by_seed(episodes)
    verifier = LogisticVisualVerifier.load(model)
    temporal_model = fit_causal_temporal(verifier, splits["train"], window=CAUSAL_WINDOW)
    temporal_model_path = Path(dataset).with_name("l2-temporal-8.npz")
    temporal_model.save(temporal_model_path)
    ttl = _tune_ttl(verifier, splits["dev"], target_false_pass)
    change = _tune_change(verifier, splits["dev"], target_false_pass)
    temporal = _tune_temporal(verifier, temporal_model, splits["dev"], target_false_pass)
    temporal_model.state_threshold = float(temporal["params"].get("state_threshold", 0.5))
    temporal_model.change_threshold = float(temporal["params"].get("change_threshold", 0.5))
    temporal_model.save(temporal_model_path)
    selected = {
        "frame": {},
        "ttl": ttl["params"],
        "change": change["params"],
        "temporal8": temporal["params"],
        # Kept as a diagnostic opponent: history without a learned change head.
        "temporal_no_change": {"window": 3},
    }
    dev_results = {
        policy: evaluate_policy(
            verifier,
            splits["dev"],
            policy,
            params,
            temporal_model=temporal_model if policy == "temporal8" else None,
        )
        for policy, params in selected.items()
    }
    test_results = {
        policy: evaluate_policy(
            verifier,
            splits["test"],
            policy,
            params,
            temporal_model=temporal_model if policy == "temporal8" else None,
        )
        for policy, params in selected.items()
    }
    report: Dict[str, Any] = {
        "schema": "auditembodied.l2.temporal_baselines.v2",
        "dataset": str(dataset),
        "model": str(model),
        "protocol": {
            "trajectory_schema": TRAJECTORY_SCHEMA,
            "evaluator_schema": EVALUATOR_SCHEMA,
            "proprio_dim": ROBOT_PROPRIO_DIM,
            "labels_source": "evaluator-only sidecar",
        },
        "temporal_model": str(temporal_model_path),
        "window": CAUSAL_WINDOW,
        "causal": True,
        "primary_policies": ["frame", "ttl", "change", "temporal8"],
        "threshold": float(verifier.threshold),
        "target_false_pass": float(target_false_pass),
        "split_seeds": {name: sorted({int(ep["seed"]) for ep in value}) for name, value in splits.items()},
        "tuning": {"ttl_dev": ttl, "change_dev": change, "temporal8_dev": temporal},
        "temporal_heads": {
            "train": temporal_head_diagnostics(verifier, temporal_model, splits["train"]),
            "dev": temporal_head_diagnostics(verifier, temporal_model, splits["dev"]),
            "test": temporal_head_diagnostics(verifier, temporal_model, splits["test"]),
        },
        "selected": selected,
        "dev": dev_results,
        "test": test_results,
        "bootstrap_by_seed": _bootstrap_report(test_results, test_results["frame"]["rows"]),
        "paired_bootstrap_vs_frame": {
            policy: {
                "success_rate": _paired_seed_difference(
                    result["rows"], test_results["frame"]["rows"], "success"
                ),
                "false_pass_rate": _paired_seed_difference(
                    [
                        {
                            **row,
                            "false_pass": bool(row["unsupported_release"]),
                        }
                        for row in result["rows"]
                    ],
                    [
                        {
                            **row,
                            "false_pass": bool(row["unsupported_release"]),
                        }
                        for row in test_results["frame"]["rows"]
                    ],
                    "false_pass",
                ),
            }
            for policy, result in test_results.items()
            if policy != "frame"
        },
    }
    output = Path(dataset).with_name("l2-temporal-report.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["output"] = str(output)
    return report


def _online_policy(
    history: Sequence[Mapping[str, Any]],
    verifier: LogisticVisualVerifier,
    policy: str,
    params: Mapping[str, Any],
    temporal_model: Optional[CausalTemporalVerifier] = None,
) -> bool:
    if not history:
        return False
    current = history[-1]
    threshold = float(verifier.threshold)
    if policy == "frame":
        return current["probability"] >= threshold
    if policy == "temporal_no_change":
        window = max(1, int(params.get("window", 3)))
        return float(np.mean([row["probability"] for row in history[-window:]])) >= threshold
    if policy == "ttl":
        if current["probability"] >= threshold:
            return True
        positive = [i for i, row in enumerate(history[:-1]) if row["probability"] >= threshold]
        return bool(positive) and (len(history) - 1 - positive[-1]) <= int(params.get("ttl", 0))
    if policy == "change":
        if current["probability"] < threshold:
            return False
        positive = [row for row in history[:-1] if row["probability"] >= threshold]
        if not positive:
            return True
        return float(np.mean(np.abs(current["feature"] - positive[-1]["feature"]))) <= float(params.get("delta", 0.5))
    if policy == "temporal8":
        if temporal_model is None:
            raise ValueError("online temporal8 policy requires a fitted model")
        probabilities = np.asarray([float(row["probability"]) for row in history], dtype=np.float64)
        features = np.stack([np.asarray(row["feature"], dtype=np.float64) for row in history], axis=0)
        state, change = temporal_model.predict_sequence(probabilities, features)
        return _temporal_predict_index(state, change, len(history) - 1, params)
    raise ValueError(f"unknown L2 policy: {policy}")


def online_temporal_benchmark(
    verifier: LogisticVisualVerifier,
    policy: str,
    params: Mapping[str, Any],
    seeds: Iterable[int] = range(3),
    output: str | Path = "outputs/l2-online-pilot.json",
    continuous_recovery: bool = True,
    perturbations: Sequence[str] = ("nominal", "early", "late"),
    temporal_model: Optional[CausalTemporalVerifier] = None,
) -> Dict[str, Any]:
    """Run the shared no-anchor physical protocol with a robot-only gate."""
    from .l2_ablation import _predict
    from .r1_validation import PHYSICAL_PROTOCOL, run_physical_trial

    rows: List[Dict[str, Any]] = []
    selected = {"threshold": verifier.threshold, **params}
    def gate(stream, index, allow_stale):
        probabilities, features = _probs_and_features(verifier, stream)
        cache = {"episode": stream, "probabilities": probabilities, "features": features}
        if temporal_model is not None:
            cache["state"], cache["change"] = temporal_model.predict_sequence(probabilities, features)
        return _predict(cache, index, policy, selected, allow_stale=allow_stale)

    for seed in seeds:
        for perturbation in perturbations:
            row, _ = run_physical_trial(int(seed), perturbation, render=True, gate=gate)
            rows.append({key: value for key, value in row.items() if key != "trace"})
    # continuous_recovery is retained for caller compatibility; the current
    # protocol always uses the validated actuator-only recovery skill.
    result = {"schema": "auditembodied.l2.online_physical.v1", "protocol": PHYSICAL_PROTOCOL,
              "policy": policy, "params": selected, "episodes": len(rows),
              "success_rate": float(np.mean([row["success"] for row in rows])) if rows else 0.0,
              "false_pass_rate": sum(row["unsupported_release"] for row in rows) / max(1, sum(row["unsupported_checks"] for row in rows)),
              "mean_observations": float(np.mean([row["observations"] for row in rows])) if rows else 0.0,
              "success_scope": "physical_release_and_1s_hold", "rows": rows}
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["output"] = str(output_path)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate AuditEmbodied L2 temporal baselines")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-false-pass", type=float, default=0.02)
    args = parser.parse_args(argv)
    print(json.dumps(run_l2(args.dataset, args.model, args.target_false_pass), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
