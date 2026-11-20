"""Threshold and change-detection ablations for the frozen L2 protocol.

This module deliberately evaluates the existing L2 policies from cached
single-frame scores and temporal-head scores.  Thresholds and policy
parameters are selected on the dev seeds only; the test split is evaluated
with the selected dev operating points and is also written as a full
risk-coverage curve for auditability.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .continuous import EVALUATOR_SCHEMA, ROBOT_PROPRIO_DIM, TRAJECTORY_SCHEMA
from .l1 import LogisticVisualVerifier, _replay_release_decision, load_hdf5, split_by_seed
from .temporal import (
    CAUSAL_WINDOW,
    CausalTemporalVerifier,
    _change_candidates,
    _decision_index,
    _probs_and_features,
    _temporal_predict_index,
    fit_causal_temporal,
    temporal_head_diagnostics,
)


POLICIES = ("frame", "ttl", "change", "temporal_no_change", "temporal8")
# `change` is a hand-coded feature-distance control, not a learned detector.
LEARNED_POLICIES = ("temporal8",)
TTL_VALUES = (0, 1, 2, 3, 4, 6, 8, 12, 20)
HISTORY_WINDOWS = (1, 2, 3, 5, 8)
STATE_THRESHOLDS = (0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0, 1.000001)
CHANGE_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def _round_unique(values: Iterable[float], digits: int = 8) -> List[float]:
    return sorted({float(round(float(value), digits)) for value in values})


def _decision_probabilities(caches: Sequence[Mapping[str, Any]]) -> np.ndarray:
    values: List[float] = []
    for cache in caches:
        decision = _decision_index(cache["episode"])
        if decision is not None:
            values.append(float(cache["probabilities"][decision]))
    return np.asarray(values, dtype=np.float64)


def _threshold_grid(caches: Sequence[Mapping[str, Any]]) -> List[float]:
    scores = _decision_probabilities(caches)
    fixed = (0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999, 1.0, 1.000001)
    if not len(scores):
        return list(fixed)
    quantiles = np.quantile(scores, np.linspace(0.02, 0.98, 49))
    return _round_unique([*fixed, *quantiles])


def _change_delta_grid(verifier: LogisticVisualVerifier, dev: Sequence[Mapping[str, Any]]) -> List[float]:
    # Use only dev observations to construct the candidate grid.  The helper
    # includes fixed readable values and observed dev quantiles.
    values = _change_candidates(verifier, dev)
    return _round_unique([*values, 1.5, 2.0, 3.0])


def _cache_episodes(
    verifier: LogisticVisualVerifier,
    episodes: Sequence[Mapping[str, Any]],
    temporal_model: CausalTemporalVerifier,
) -> List[Dict[str, Any]]:
    cached: List[Dict[str, Any]] = []
    for episode in episodes:
        probabilities, features = _probs_and_features(verifier, episode)
        state, change = temporal_model.predict_sequence(probabilities, features)
        cached.append(
            {
                "episode": episode,
                "probabilities": probabilities,
                "features": features,
                "state": state,
                "change": change,
            }
        )
    return cached


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
        i for i in range(max(0, int(before)))
        if probabilities[i] >= threshold and stages[i] in preferred
    ]
    if candidates:
        return candidates[-1]
    candidates = [i for i in range(max(0, int(before))) if probabilities[i] >= threshold]
    return candidates[-1] if candidates else None


def _change_score(features: np.ndarray, current: int, reference: int) -> float:
    if current == reference:
        return 0.0
    return float(np.mean(np.abs(features[current] - features[reference])))


def _predict(
    cache: Mapping[str, Any],
    index: int,
    policy: str,
    params: Mapping[str, Any],
    *,
    allow_stale: bool = True,
) -> bool:
    probabilities = np.asarray(cache["probabilities"], dtype=np.float64)
    features = np.asarray(cache["features"], dtype=np.float64)
    episode = cache["episode"]
    stages = list(episode["stages"])
    threshold = float(params.get("threshold", 0.5))
    index = int(index)

    if policy == "frame":
        return bool(probabilities[index] >= threshold)
    if policy == "temporal_no_change":
        window = max(1, int(params.get("window", 3)))
        start = max(0, index - window + 1)
        return bool(np.mean(probabilities[start : index + 1]) >= threshold)
    if policy == "ttl":
        if probabilities[index] >= threshold:
            return True
        if not allow_stale:
            return False
        confirmation = _confirmation_index(probabilities, stages, index, threshold)
        ttl = int(params.get("ttl", 0))
        return bool(confirmation is not None and index - confirmation <= ttl)
    if policy == "change":
        if probabilities[index] < threshold:
            return False
        confirmation = _confirmation_index(probabilities, stages, index, threshold)
        if confirmation is None:
            return True
        delta = float(params.get("delta", 0.5))
        return bool(_change_score(features, index, confirmation) <= delta)
    if policy == "temporal8":
        return bool(
            _temporal_predict_index(
                np.asarray(cache["state"], dtype=np.float64),
                np.asarray(cache["change"], dtype=np.float64),
                index,
                params,
                allow_stale=allow_stale,
            )
        )
    raise ValueError(f"unknown L2 ablation policy: {policy}")


def _rows(
    caches: Sequence[Mapping[str, Any]],
    policy: str,
    params: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cache in caches:
        episode = cache["episode"]
        decision = _decision_index(episode)
        if decision is None:
            continue
        replay = _replay_release_decision(
            episode, decision,
            lambda index, allow_stale: _predict(cache, index, policy, params, allow_stale=allow_stale),
        )
        rows.append(
            {
                "seed": int(episode["seed"]),
                "perturbation": str(episode["perturbation"]),
                "decision_index": int(decision),
                "probability": float(
                    cache["state"][decision] if policy == "temporal8" else cache["probabilities"][decision]
                ),
                "change_probability": float(cache["change"][decision]) if policy == "temporal8" else None,
                **replay,
            }
        )
    return rows


def _metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "episodes": 0.0,
            "success_rate": 0.0,
            "false_pass_rate": 0.0,
            "false_reject_rate": 0.0,
            "coverage": 0.0,
            "risk": 0.0,
            "release_rate": 0.0,
            "mean_observations": 0.0,
            "mean_extra_observations": 0.0,
            "late_false_pass_rate": 0.0,
            "unsafe_release_rate": 0.0,
        }
    release = np.asarray([bool(row["predicted_support_at_decision"]) for row in rows], dtype=bool)
    labels = np.asarray([bool(row["true_support_at_decision"]) for row in rows], dtype=bool)
    unsafe = np.asarray([bool(row["unsupported_release"]) for row in rows], dtype=bool)
    actual_release = np.asarray([row["release_index"] is not None for row in rows], dtype=bool)
    safe = release & labels
    positives = int(np.sum(labels))
    negatives = sum(int(row["unsupported_checks"]) for row in rows)
    releases = int(np.sum(actual_release))
    late_rows = [row for row in rows if row["perturbation"] == "late"]
    late_unsafe = [
        bool(row["unsupported_release"])
        for row in late_rows
    ]
    return {
        "episodes": float(len(rows)),
        "success_rate": float(np.mean([bool(row["success"]) for row in rows])),
        "false_pass_rate": float(np.sum(unsafe) / max(1, negatives)),
        "false_reject_rate": float(np.sum(labels & ~release) / max(1, positives)),
        "coverage": float(np.sum(safe) / max(1, positives)),
        "risk": float(np.sum(unsafe) / max(1, releases)),
        "release_rate": float(np.mean(actual_release)),
        "unsafe_release_rate": float(np.mean(unsafe)),
        "mean_observations": float(np.mean([float(row["observations"]) for row in rows])),
        "mean_extra_observations": float(np.mean([float(row["extra_observations"]) for row in rows])),
        "late_false_pass_rate": float(np.mean(late_unsafe)) if late_unsafe else 0.0,
    }


def _evaluate(
    caches: Sequence[Mapping[str, Any]],
    policy: str,
    params: Mapping[str, Any],
    *,
    include_rows: bool = False,
) -> Dict[str, Any]:
    rows = _rows(caches, policy, params)
    metrics = _metrics(rows)
    by_perturbation = {
        perturbation: _metrics([row for row in rows if row["perturbation"] == perturbation])
        for perturbation in sorted({row["perturbation"] for row in rows})
    }
    result: Dict[str, Any] = {
        "policy": policy,
        "params": {str(key): float(value) if isinstance(value, (np.floating, float)) else int(value) if isinstance(value, (np.integer, int)) else value for key, value in params.items()},
        **metrics,
        "by_perturbation": by_perturbation,
    }
    if include_rows:
        result["rows"] = rows
    return result


def _candidate_grid(
    verifier: LogisticVisualVerifier,
    dev_caches: Sequence[Mapping[str, Any]],
    dev_episodes: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    thresholds = _threshold_grid(dev_caches)
    deltas = _change_delta_grid(verifier, dev_episodes)
    return {
        "frame": [{"threshold": threshold} for threshold in thresholds],
        "ttl": [
            {"threshold": threshold, "ttl": ttl}
            for threshold in thresholds
            for ttl in TTL_VALUES
        ],
        "change": [
            {"threshold": threshold, "delta": delta}
            for threshold in thresholds
            for delta in deltas
        ],
        "temporal_no_change": [
            {"threshold": threshold, "window": window}
            for threshold in thresholds
            for window in HISTORY_WINDOWS
        ],
        "temporal8": [
            {"state_threshold": state_threshold, "change_threshold": change_threshold}
            for state_threshold in STATE_THRESHOLDS
            for change_threshold in CHANGE_THRESHOLDS
        ],
    }


def _select_dev(candidates: Sequence[Mapping[str, Any]], target_false_pass: float) -> Dict[str, Any]:
    safe = [row for row in candidates if float(row["false_pass_rate"]) <= float(target_false_pass) + 1e-12]
    pool = safe if safe else list(candidates)
    if not pool:
        raise ValueError("empty L2 ablation candidate set")
    selected = max(
        pool,
        key=lambda row: (
            float(row["success_rate"]),
            float(row["coverage"]),
            -float(row["mean_extra_observations"]),
            -float(row["risk"]),
        ),
    )
    return dict(selected)


def _paired_bootstrap_value(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    value_key: str,
    n_bootstrap: int = 2000,
    seed: int = 20261017,
) -> Dict[str, Any]:
    def by_seed(rows: Sequence[Mapping[str, Any]]) -> Dict[int, float]:
        grouped: Dict[int, List[float]] = {}
        for row in rows:
            grouped.setdefault(int(row["seed"]), []).append(float(row.get(value_key, 0.0)))
        return {key: float(np.mean(values)) for key, values in grouped.items()}

    left = by_seed(left_rows)
    right = by_seed(right_rows)
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


def _curve_csv(path: Path, curves: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "policy",
        "candidate_id",
        "params",
        "false_pass_rate",
        "risk",
        "coverage",
        "release_rate",
        "success_rate",
        "false_reject_rate",
        "mean_extra_observations",
        "late_false_pass_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split, policy_curves in curves.items():
            for policy, points in policy_curves.items():
                for candidate_id, point in enumerate(points):
                    writer.writerow(
                        {
                            "split": split,
                            "policy": policy,
                            "candidate_id": candidate_id,
                            "params": json.dumps(point["params"], sort_keys=True),
                            **{field: point.get(field, "") for field in fields[4:]},
                        }
                    )


def _markdown_report(path: Path, report: Mapping[str, Any]) -> None:
    go = report["go_no_go"]
    lines = [
        "# L2 阈值与变化检测消融",
        "",
        f"- 数据：`{report['dataset']}`",
        f"- 目标错误放行率：`{report['target_false_pass']:.3f}`",
        f"- Go/No-Go：**{go['decision']}**",
        f"- 判据：{go['criterion']}",
        "",
        "## 测试集选定工作点",
        "",
        "| 策略 | 成功率 | 错误放行率 | 风险 | coverage | 额外观察 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy, point in report["selected_test"].items():
        lines.append(
            f"| {policy} | {point['success_rate']:.3f} | {point['false_pass_rate']:.3f} | "
            f"{point['risk']:.3f} | {point['coverage']:.3f} | {point['mean_extra_observations']:.2f} |"
        )
    lines.extend(["", "## 与最佳非学习基线的配对比较", ""])
    for policy, comparison in report["go_no_go"]["comparisons"].items():
        lines.append(
            f"- `{policy}` vs `{comparison['baseline_policy']}`：成功率差 "
            f"`{comparison['success_gain']['estimate']:.3f}` "
            f"（95% CI `{comparison['success_gain']['ci95'][0]:.3f}, {comparison['success_gain']['ci95'][1]:.3f}`），"
            f"额外观察差 `{comparison['extra_observation_gain']['estimate']:.2f}`。"
        )
    lines.extend(["", "完整风险—coverage 点见 `l2-ablation-report.json` 与 `l2-risk-coverage.csv`。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_l2_ablation(
    dataset: str | Path,
    model: str | Path,
    output_dir: str | Path,
    target_false_pass: float = 0.02,
) -> Dict[str, Any]:
    dataset = Path(dataset)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = load_hdf5(dataset)
    splits = split_by_seed(episodes)
    verifier = LogisticVisualVerifier.load(model)
    temporal_model = fit_causal_temporal(verifier, splits["train"], window=CAUSAL_WINDOW)
    dev_caches = _cache_episodes(verifier, splits["dev"], temporal_model)
    test_caches = _cache_episodes(verifier, splits["test"], temporal_model)
    grids = _candidate_grid(verifier, dev_caches, splits["dev"])

    dev_curves: Dict[str, List[Dict[str, Any]]] = {}
    test_curves: Dict[str, List[Dict[str, Any]]] = {}
    selected_dev: Dict[str, Dict[str, Any]] = {}
    selected_test: Dict[str, Dict[str, Any]] = {}
    selected_test_rows: Dict[str, List[Dict[str, Any]]] = {}
    for policy in POLICIES:
        dev_points: List[Dict[str, Any]] = []
        test_points: List[Dict[str, Any]] = []
        for params in grids[policy]:
            dev_point = _evaluate(dev_caches, policy, params, include_rows=False)
            test_point = _evaluate(test_caches, policy, params, include_rows=False)
            dev_points.append(dev_point)
            test_points.append(test_point)
        dev_curves[policy] = dev_points
        test_curves[policy] = test_points
        selected = _select_dev(dev_points, target_false_pass)
        selected_dev[policy] = selected
        selected_params = selected["params"]
        selected_test[policy] = _evaluate(test_caches, policy, selected_params, include_rows=False)
        selected_test_rows[policy] = _rows(test_caches, policy, selected_params)

    baseline_policy = max(
        ("frame", "ttl", "temporal_no_change"),
        key=lambda policy: (
            float(selected_dev[policy]["success_rate"]),
            float(selected_dev[policy]["coverage"]),
            -float(selected_dev[policy]["mean_extra_observations"]),
        ),
    )
    comparisons: Dict[str, Any] = {}
    for policy in LEARNED_POLICIES:
        comparisons[policy] = {
            "baseline_policy": baseline_policy,
            "success_gain": _paired_bootstrap_value(
                selected_test_rows[policy], selected_test_rows[baseline_policy], "success", seed=20261017
            ),
            "extra_observation_gain": _paired_bootstrap_value(
                selected_test_rows[policy], selected_test_rows[baseline_policy], "extra_observations", seed=20261018
            ),
            "false_pass_safe": bool(selected_test[policy]["false_pass_rate"] <= target_false_pass + 1e-12),
        }

    go_reasons: List[str] = []
    go_candidates: List[str] = []
    for policy, comparison in comparisons.items():
        test_point = selected_test[policy]
        success_gain = comparison["success_gain"]
        extra_gain = comparison["extra_observation_gain"]
        success_go = (
            comparison["false_pass_safe"]
            and float(success_gain["ci95"][0]) > 0.0
        )
        cost_go = (
            comparison["false_pass_safe"]
            and float(success_gain["estimate"]) >= -0.02
            and float(extra_gain["estimate"]) <= -1.0
        )
        if success_go or cost_go:
            go_candidates.append(policy)
        else:
            go_reasons.append(
                f"{policy} 未在相同风险约束下显著提高成功率，也未达到至少 1 个额外观察的成本节省"
            )
    decision = "GO" if go_candidates else "NO-GO"
    criterion = (
        "任一学习型变化策略在 test 上满足错误放行率≤目标且成功率相对最佳非学习基线的 seed-bootstrap 95% CI 下界>0；"
        "或成功率不低于 2 个百分点且平均额外观察至少减少 1 步。"
    )
    report: Dict[str, Any] = {
        "schema": "auditembodied.l2.threshold_change_ablation.v1",
        "dataset": str(dataset),
        "model": str(model),
        "target_false_pass": float(target_false_pass),
        "causal_window": CAUSAL_WINDOW,
        "protocol": {
            "trajectory_schema": TRAJECTORY_SCHEMA,
            "evaluator_schema": EVALUATOR_SCHEMA,
            "proprio_dim": ROBOT_PROPRIO_DIM,
            "labels_source": "evaluator-only sidecar",
            "selection_split": "dev",
            "evaluation_split": "test",
            "evaluation_mode": "offline_gate_replay",
            "success_scope": "support_at_replayed_release",
            "false_pass_denominator": "unsupported checks actually consumed, including recovery",
            "coverage_denominator": "supported initial episode decisions",
        },
        "split_seeds": {name: sorted({int(ep["seed"]) for ep in value}) for name, value in splits.items()},
        "candidate_counts": {policy: len(points) for policy, points in grids.items()},
        "temporal_heads": {
            "train_at_default_threshold": temporal_head_diagnostics(verifier, temporal_model, splits["train"]),
            "dev_at_default_threshold": temporal_head_diagnostics(verifier, temporal_model, splits["dev"]),
            "test_at_default_threshold": temporal_head_diagnostics(verifier, temporal_model, splits["test"]),
        },
        "selected_dev": selected_dev,
        "selected_test": selected_test,
        "curves": {"dev": dev_curves, "test": test_curves},
        "go_no_go": {
            "decision": decision,
            "criterion": criterion,
            "candidates": go_candidates,
            "reasons": go_reasons,
            "baseline_policy": baseline_policy,
            "comparisons": comparisons,
        },
    }
    report_path = output_dir / "l2-ablation-report.json"
    csv_path = output_dir / "l2-risk-coverage.csv"
    markdown_path = output_dir / "l2-go-no-go.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _curve_csv(csv_path, {"dev": dev_curves, "test": test_curves})
    _markdown_report(markdown_path, report)
    report["outputs"] = {
        "report": str(report_path),
        "risk_coverage_csv": str(csv_path),
        "go_no_go": str(markdown_path),
    }
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run L2 threshold/change-detection ablations")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="outputs/l2-ablation")
    parser.add_argument("--target-false-pass", type=float, default=0.02)
    args = parser.parse_args(argv)
    report = run_l2_ablation(args.dataset, args.model, args.output_dir, args.target_false_pass)
    print(json.dumps({"decision": report["go_no_go"]["decision"], "outputs": report["outputs"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
