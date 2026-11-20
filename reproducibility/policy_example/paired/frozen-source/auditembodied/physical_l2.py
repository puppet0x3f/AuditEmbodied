"""L1 calibration and L2 ablations on the frozen, actuator-only protocol."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import multiprocessing
from pathlib import Path
import socket
from typing import Any, Mapping

import numpy as np

from .l1 import (
    LogisticVisualVerifier, _classification_metrics, _fit_logistic,
    _replay_release_decision, _stack_split, calibrate_threshold,
    evaluate_verifier, load_hdf5, split_by_seed,
)
from .l2_ablation import _paired_bootstrap_value
from .r1_validation import PHYSICAL_PROTOCOL, run_physical_trial
from .temporal import _fit_binary_head, _probs_and_features, _sigmoid, _temporal_vector


POLICIES = ("frame", "ttl", "history", "learned_change", "oracle_change")
DEPLOYABLE = POLICIES[:-1]
TRAIN_SEEDS = list(range(30))
DEV_SEEDS = list(range(60, 68))
TEST_SEEDS = list(range(200, 212))


def _loss_targets(labels):
    """A support loss remains an event target for three 20 Hz samples."""
    labels = np.asarray(labels, dtype=np.int8)
    losses = np.r_[False, (labels[:-1] == 1) & (labels[1:] == 0)]
    return np.asarray([np.any(losses[max(0, i - 2):i + 1]) for i in range(len(labels))], dtype=np.int8)


@dataclass
class ChangeHead:
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray
    window: int = 8

    def predict(self, probabilities, features):
        vectors = np.stack([_temporal_vector(probabilities, features, i, self.window) for i in range(len(probabilities))])
        return _sigmoid(((vectors - self.mean) / self.scale) @ self.weights + self.bias)

    def save(self, path):
        np.savez_compressed(path, weights=self.weights, bias=self.bias, mean=self.mean, scale=self.scale, window=self.window)

    @classmethod
    def load(cls, path):
        values = np.load(path)
        return cls(values["weights"], float(values["bias"]), values["mean"], values["scale"], int(values["window"]))


def _fit_change(verifier, train):
    vectors, targets = [], []
    for ep in train:
        probabilities, features = _probs_and_features(verifier, ep)
        vectors.extend(_temporal_vector(probabilities, features, i) for i in range(len(probabilities)))
        targets.extend(_loss_targets(ep["labels"]))
    x, y = np.asarray(vectors), np.asarray(targets)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(positive) or not len(negative):
        raise ValueError("change head training needs both losses and unchanged support")
    # Rare events get equal class weight; no dev/test examples enter fitting.
    repeat = max(1, int(round(len(negative) / len(positive))))
    balanced = np.r_[negative, np.tile(positive, repeat)]
    weights, bias, mean, scale = _fit_binary_head(x[balanced], y[balanced])
    return ChangeHead(weights, bias, mean, scale)


def _cache(verifier, head, episodes):
    result = []
    for ep in episodes:
        p, f = _probs_and_features(verifier, ep)
        result.append({"episode": ep, "p": p, "change": head.predict(p, f),
                       "oracle_change": _loss_targets(ep["labels"])})
    return result


def predict(cache, index, policy, params, allow_stale=True):
    p = cache["p"]
    threshold = params["threshold"]
    if policy == "frame":
        return bool(p[index] >= threshold)
    if policy == "history":
        return bool(np.mean(p[max(0, index - params["window"] + 1):index + 1]) >= threshold)
    if policy in ("learned_change", "oracle_change") and params.get("fresh_confirmation", False):
        # A current support certificate supersedes earlier invalidations.
        # Change detection governs reuse of old evidence, not newly acquired
        # support, including the mandatory fresh check after physical R1.
        if p[index] >= threshold:
            return True
        if not allow_stale:
            return False
        ttl = params.get("ttl", 0)
        start = max(0, index - ttl)
        previous = np.flatnonzero(p[start:index] >= threshold)
        if not len(previous):
            return False
        reference = start + int(previous[-1])
        signal = cache["change"] if policy == "learned_change" else cache["oracle_change"]
        return bool(np.max(signal[reference + 1:index + 1]) < params.get("change_threshold", 0.5))
    ttl = params.get("ttl", 0) if allow_stale else 0
    recent = np.flatnonzero(p[max(0, index - ttl):index + 1] >= threshold)
    if not len(recent):
        return False
    if policy == "ttl":
        return True
    reference = max(0, index - ttl) + int(recent[-1])
    # The same evidence rule is used for learned and perfect change signals.
    # The oracle replaces only the loss detector, not L1 support predictions.
    signal = cache["change"] if policy == "learned_change" else cache["oracle_change"]
    cutoff = params.get("change_threshold", 0.5)
    return bool(np.max(signal[max(0, reference - 2):index + 1]) < cutoff)


def _rows(caches, policy, params):
    result = []
    for cache in caches:
        ep = cache["episode"]
        if "gate_decision" not in ep["stages"]:
            # Initialization/execution failures stay in the task denominator.
            result.append({"seed": int(ep["seed"]), "perturbation": ep["perturbation"],
                           "predicted_support_at_decision": False, "true_support_at_decision": False,
                           "unsupported_release": False, "unsupported_checks": 0, "release_index": None,
                           "success": False, "observations": 0, "extra_observations": 0,
                           "failure_reason": ep["evaluator_metadata"].get("failure_reason", "missing_decision")})
            continue
        decision = list(ep["stages"]).index("gate_decision")
        row = _replay_release_decision(ep, decision, lambda i, stale: predict(cache, i, policy, params, stale))
        result.append({"seed": int(ep["seed"]), "perturbation": ep["perturbation"], **row})
    return result


def metrics(rows):
    n = len(rows)
    if not n:
        return {}
    accepted = sum(bool(r.get("predicted_support_at_decision")) for r in rows)
    initial_unsafe = sum(bool(r.get("predicted_support_at_decision")) and not r["true_support_at_decision"] for r in rows)
    unsafe = sum(r["unsupported_release"] for r in rows)
    releases = sum(r["release_index"] is not None for r in rows)
    positives = sum(bool(r.get("true_support_at_decision")) for r in rows)
    supported_accepted = sum(bool(r.get("predicted_support_at_decision")) and r["true_support_at_decision"] for r in rows)
    return {
        "episodes": n, "success_rate": float(np.mean([r["success"] for r in rows])),
        "false_pass_rate": unsafe / max(1, sum(r["unsupported_checks"] for r in rows)),
        "unsafe_release_count": unsafe, "unsafe_release_rate": unsafe / n,
        "risk": unsafe / max(1, releases),
        "initial_risk": initial_unsafe / max(1, accepted),
        "coverage": accepted / n, "supported_coverage": supported_accepted / max(1, positives),
        "mean_observations": float(np.mean([r["observations"] for r in rows])),
        "mean_extra_observations": float(np.mean([r["extra_observations"] for r in rows])),
        "failure_reasons": dict(Counter(r.get("failure_reason") for r in rows if r.get("failure_reason"))),
    }


def _grids(dev_caches, reference_threshold=0.5):
    scores = sorted(set(float(c["p"][i]) for c in dev_caches
                        for i, stage in enumerate(c["episode"]["stages"])
                        if stage in ("gate_decision", "recovery_decision")))
    midpoints = [(a + b) / 2 for a, b in zip(scores[:-1], scores[1:])]
    # Use gaps between observed scores instead of an epsilon above a single
    # negative. The fixed calibrated L1 threshold is the safety floor and is
    # always included, including when it is the minimum positive R1 score.
    thresholds = sorted(set(t for t in [float(reference_threshold), 0.5, 0.8, 0.95, 0.99,
                                        0.999, 1.000001, *midpoints] if t >= reference_threshold))
    ttls = (0, 3, 8, 20)
    return {
        "frame": [{"threshold": t} for t in thresholds],
        "ttl": [{"threshold": t, "ttl": age} for t in thresholds for age in ttls],
        "history": [{"threshold": t, "window": w} for t in thresholds for w in (1, 3, 5, 8)],
        "learned_change": [{"threshold": t, "ttl": age, "change_threshold": c}
                           for t in thresholds for age in ttls for c in (0.1, 0.3, 0.5, 0.7, 0.9)],
        "oracle_change": [{"threshold": t, "ttl": age} for t in thresholds for age in ttls],
    }


def _point(caches, policy, params):
    return {"params": params, **metrics(_rows(caches, policy, params))}


def _select(points, target, reference_threshold=0.5):
    safe = [p for p in points if p["false_pass_rate"] <= target]
    if not safe:
        raise ValueError("no dev operating point satisfies the false-pass constraint")
    return max(safe, key=lambda p: (
        p["success_rate"], -p["mean_extra_observations"], p["supported_coverage"], -p["risk"],
        -abs(p["params"]["threshold"] - reference_threshold),
        -p["params"].get("ttl", 0), -p["params"].get("window", 1),
    ))


def _online_job(job):
    seed, perturbation, policy, params, model_path, head_path = job
    verifier, head = LogisticVisualVerifier.load(model_path), ChangeHead.load(head_path)
    def gate(stream, index, allow_stale):
        # The runner deliberately omits labels, predicates and evaluator metadata.
        p, f = _probs_and_features(verifier, stream)
        return predict({"p": p, "change": head.predict(p, f)}, index, policy, params, allow_stale)
    row, _ = run_physical_trial(seed, perturbation, render=True, gate=gate)
    return policy, {k: v for k, v in row.items() if k != "trace"}


def _plot(curves, selected, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    colors = dict(zip(POLICIES, ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2")))
    for ax, split in zip(axes, ("dev", "test")):
        for policy, points in curves[split].items():
            x = [p["coverage"] for p in points]
            y = [p["initial_risk"] for p in points]
            ax.scatter(x, y, s=12, alpha=0.25, color=colors[policy])
            # Lower envelope summarizes the fixed grid; points remain in CSV.
            envelope = sorted({(p["coverage"], min(q["initial_risk"] for q in points if q["coverage"] == p["coverage"])) for p in points})
            ax.plot([p[0] for p in envelope], [p[1] for p in envelope], color=colors[policy], label=policy, linewidth=1.5)
            point = selected[split][policy]
            ax.scatter(point["coverage"], point["initial_risk"], marker="*", s=95, color=colors[policy], edgecolor="black", linewidth=0.4)
        ax.set(title=f"{split} (dev-selected stars)", xlabel="Initial release coverage", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Unsafe / accepted initial releases")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(output / "l2-risk-coverage.png", dpi=180)
    fig.savefig(output / "l2-risk-coverage.pdf")
    plt.close(fig)


def run_study(dataset, output_dir, *, online=True, workers=4, target=0.02):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    episodes = load_hdf5(dataset)
    if any(ep["metadata"].get("protocol") != PHYSICAL_PROTOCOL for ep in episodes):
        raise ValueError("this study requires the new physical dataset")
    splits = split_by_seed(episodes)
    seed_split = {name: sorted({int(ep["seed"]) for ep in group}) for name, group in splits.items()}
    if seed_split != {"train": TRAIN_SEEDS, "dev": DEV_SEEDS, "test": TEST_SEEDS}:
        raise ValueError(f"expected frozen 30/8/12 new-seed split, got {seed_split}")
    x, y = _stack_split(splits["train"])
    verifier = _fit_logistic(x, y)
    del x, y
    model_path, head_path = output / "l1-calibrated.npz", output / "change-head.npz"
    # Calibrate the exact serialized checkpoint used by live workers.
    verifier.save(model_path)
    verifier = LogisticVisualVerifier.load(model_path)
    calibration = calibrate_threshold(verifier, splits["dev"], target_false_pass=target)
    verifier.threshold = calibration["selected"]["threshold"]
    verifier.save(model_path)
    head = _fit_change(verifier, splits["train"])
    head.save(head_path)
    dev_caches = _cache(verifier, head, splits["dev"])
    grids = _grids(dev_caches, reference_threshold=verifier.threshold)
    curves = {"dev": {}, "test": {}}
    selected = {"dev": {}, "test": {}}
    for policy in POLICIES:
        points = [_point(dev_caches, policy, params) for params in grids[policy]]
        curves["dev"][policy] = points
        selected["dev"][policy] = _select(points, target, reference_threshold=verifier.threshold)
    baseline = max(("frame", "ttl", "history"), key=lambda p: (
        selected["dev"][p]["success_rate"], -selected["dev"][p]["mean_extra_observations"]))
    # Write operating points before any test scoring. They are never retuned.
    frozen = {"protocol": PHYSICAL_PROTOCOL, "seed_split": seed_split, "target_false_pass": target,
              "selection_split": "dev", "selected_dev": selected["dev"], "baseline_policy": baseline,
              "change_target": "support loss in the last three samples; causal window eight",
              "go_criterion": "online FPR <= 0.02 and positive success gain CI95 lower bound; or success drop <= 0.02 with >= 1 fewer observations",
              "observation_cost": "all robot frame bundles consumed at 20 Hz up to the selected release; release/hold audit frames excluded",
              "risk_coverage": "unsafe / accepted initial releases versus initial releases / episodes"}
    frozen["risk_constraint_scope"] = "empirical at evaluated checks; 12 test seed clusters do not certify population risk <= 2%"
    (output / "frozen-operating-points.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print(json.dumps({"stage": "dev_frozen", "baseline": baseline, "selected": selected["dev"]}), flush=True)
    test_caches = _cache(verifier, head, splits["test"])
    offline_rows = {}
    for policy in POLICIES:
        curves["test"][policy] = [_point(test_caches, policy, params) for params in grids[policy]]
        params = selected["dev"][policy]["params"]
        offline_rows[policy] = _rows(test_caches, policy, params)
        selected["test"][policy] = {"params": params, **metrics(offline_rows[policy])}
    import csv
    with (output / "l2-risk-coverage.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "policy", "params", "coverage", "risk", "supported_coverage", "replay_success", "mean_extra_observations"))
        for split, policies in curves.items():
            for policy, points in policies.items():
                writer.writerows((split, policy, json.dumps(p["params"]), p["coverage"], p["initial_risk"], p["supported_coverage"], p["success_rate"], p["mean_extra_observations"]) for p in points)
    _plot(curves, selected, output)
    l1_report = {
        "split_seeds": seed_split, "calibration": {k: v for k, v in calibration.items() if k not in ("rows", "candidates")},
        **{split: evaluate_verifier(verifier, group) for split, group in splits.items()},
    }
    (output / "l1-report.json").write_text(json.dumps(l1_report, indent=2) + "\n")
    chosen_threshold = selected["dev"]["learned_change"]["params"]["change_threshold"]
    head_diagnostics = {}
    for split, caches in (("dev", dev_caches), ("test", test_caches)):
        truth = np.concatenate([c["oracle_change"] for c in caches])
        prediction = np.concatenate([c["change"] >= chosen_threshold for c in caches])
        raw = _classification_metrics(truth, prediction)
        head_diagnostics[split] = {
            "frames": raw["frames"], "accuracy": raw["accuracy"],
            "loss_precision": raw["precision_supported"], "loss_recall": raw["recall_supported"],
            "loss_f1": raw["f1_supported"], "false_alarm_rate": raw["false_pass_rate"],
            "loss_miss_rate": raw["false_reject_rate"],
        }
    report = {"protocol": PHYSICAL_PROTOCOL, "dataset": str(dataset),
              "dataset_absolute_path": str(Path(dataset).resolve()), "execution_host": socket.gethostname(), **frozen,
              "selected_dev": selected["dev"], "offline_selected_test": selected["test"],
              "offline_success_scope": "support_at_replayed_release",
              "offline_rows": offline_rows, "change_head": head_diagnostics,
              "oracle_change_scope": "perfect support-loss event signal with the same L1 and TTL; evaluator-only upper bound for change detection",
              "online": {}, "go_no_go": {"decision": "PENDING_ONLINE"}}
    report_path = output / "l2-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if online:
        jobs = [(s, q, p, selected["dev"][p]["params"], str(model_path), str(head_path))
                for s in TEST_SEEDS for q in ("nominal", "early", "late") for p in DEPLOYABLE]
        online_rows = {p: [] for p in DEPLOYABLE}
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for policy, row in pool.map(_online_job, jobs):
                online_rows[policy].append(row)
                with (output / "online-rows.jsonl").open("a") as handle:
                    handle.write(json.dumps({"policy": policy, **row}) + "\n")
                print(json.dumps({"stage": "online", "policy": policy, "seed": row["seed"], "perturbation": row["perturbation"], "success": row["success"], "failure_reason": row["failure_reason"]}), flush=True)
        report["online"] = {p: {**metrics(rows), "rows": rows,
                                "by_perturbation": {q: metrics([r for r in rows if r["perturbation"] == q]) for q in ("nominal", "early", "late")}}
                            for p, rows in online_rows.items()}
        learned = report["online"]["learned_change"]
        gain = _paired_bootstrap_value(online_rows["learned_change"], online_rows[baseline], "success")
        cost = _paired_bootstrap_value(online_rows["learned_change"], online_rows[baseline], "extra_observations")
        safe = learned["false_pass_rate"] <= target
        go = safe and (gain["ci95"][0] > 0 or (gain["estimate"] >= -0.02 and cost["estimate"] <= -1))
        report["go_no_go"] = {"decision": "GO" if go else "NO-GO", "baseline_policy": baseline,
                              "success_gain": gain, "extra_observation_gain": cost,
                              "empirical_false_pass_constraint_met": safe,
                              "reason": "learned change must improve physical success or reduce observation cost at matched safety"}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = run_study(args.dataset, args.output_dir, online=not args.offline_only, workers=max(1, args.workers))
    print(json.dumps({"go_no_go": result["go_no_go"], "report": str(Path(args.output_dir) / "l2-report.json")}))


if __name__ == "__main__":
    main()
