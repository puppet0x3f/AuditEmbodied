"""Train/dev-only L2 corrections, then one frozen evaluation on fresh seeds."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing
from pathlib import Path
import shutil
import socket

import numpy as np

from .l1 import LogisticVisualVerifier, load_hdf5
from .l2_ablation import _paired_bootstrap_value
from .physical_l2 import (
    ChangeHead, POLICIES, _grids, _loss_targets, _plot, _point, _rows, _select,
    metrics, predict,
)
from .r1_validation import PHYSICAL_PROTOCOL, run_physical_trial
from .temporal import _fit_binary_head, _probs_and_features, _temporal_vector


TRAIN = list(range(30))
DEV = list(range(60, 68))
TEST = list(range(300, 312))
DEPLOYABLE = ("frame", "ttl", "history", "learned_change")
REFERENCE_CHANGE_CUTOFF = 0.7


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _feature_cache(verifier, episodes):
    result = []
    for episode in episodes:
        p, f = _probs_and_features(verifier, episode)
        result.append({"episode": episode, "p": p, "features": f,
                       "oracle_change": _loss_targets(episode["labels"])})
    return result


def _with_head(caches, head):
    return [{**c, "change": head.predict(c["p"], c["features"])} for c in caches]


def _fit_recovery_negatives(caches, weight):
    vectors, labels, recovery = [], [], []
    for c in caches:
        ep = c["episode"]
        vectors.extend(_temporal_vector(c["p"], c["features"], i) for i in range(len(c["p"])))
        labels.extend(c["oracle_change"])
        recovery.extend(bool(y) and (stage.startswith("r1_recovery") or stage == "recovery_decision")
                        for y, stage in zip(ep["labels"], ep["stages"]))
    x, y, recovery = np.asarray(vectors), np.asarray(labels), np.asarray(recovery)
    positive = np.flatnonzero(y == 1)
    negative = np.r_[np.flatnonzero(y == 0),
                     np.tile(np.flatnonzero((y == 0) & recovery), weight - 1)]
    repeat = max(1, int(round(len(negative) / len(positive))))
    indices = np.r_[negative, np.tile(positive, repeat)]
    weights, bias, mean, scale = _fit_binary_head(x[indices], y[indices])
    return ChangeHead(weights, bias, mean, scale)


def _head_metrics(caches, cutoff, support_threshold):
    truth, alarm, recovery_alarm = [], [], []
    recovery_vetoes = 0
    for c in caches:
        ep = c["episode"]
        truth.extend(c["oracle_change"])
        alarm.extend(c["change"] >= cutoff)
        for i, stage in enumerate(ep["stages"]):
            if ep["labels"][i] and (stage.startswith("r1_recovery") or stage == "recovery_decision"):
                recovery_alarm.append(bool(c["change"][i] >= cutoff))
            if stage == "recovery_decision" and ep["labels"][i] and c["p"][i] >= support_threshold:
                recovery_vetoes += int(np.max(c["change"][max(0, i - 2):i + 1]) >= cutoff)
    truth, alarm = np.asarray(truth, dtype=bool), np.asarray(alarm, dtype=bool)
    tp, fp = np.sum(truth & alarm), np.sum(~truth & alarm)
    recall = float(tp / max(1, np.sum(truth)))
    precision = float(tp / max(1, tp + fp))
    return {"loss_recall": recall, "loss_precision": precision,
            "loss_f1": 2 * recall * precision / max(1e-12, recall + precision),
            "false_alarm_rate": float(fp / max(1, np.sum(~truth))),
            "supported_recovery_frame_alarm_rate": float(np.mean(recovery_alarm)) if recovery_alarm else 0.0,
            "legacy_supported_recovery_vetoes": recovery_vetoes}


def _curve_grid(caches, anchor):
    original = _grids(caches, reference_threshold=0.0)
    thresholds = sorted({anchor, *(p["threshold"] for p in original["frame"])})
    return {
        "frame": [{"threshold": t} for t in thresholds],
        "ttl": [{"threshold": t, "ttl": age} for t in thresholds for age in (0, 3, 8, 20)],
        "history": [{"threshold": t, "window": w} for t in thresholds for w in (1, 3, 5, 8)],
    }


def tune(source_dataset, baseline_dir, output_dir, target=0.02):
    output, baseline_dir = Path(output_dir), Path(baseline_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Filtering happens before reading RGB or joining evaluator labels.
    episodes = load_hdf5(source_dataset, seeds=TRAIN + DEV)
    if {ep["seed"] for ep in episodes} != set(TRAIN + DEV):
        raise ValueError("missing train/dev seeds")
    if any(ep["metadata"].get("protocol") != PHYSICAL_PROTOCOL for ep in episodes):
        raise ValueError("requires the corrected physical v2 protocol")
    model_path = output / "l1-calibrated.npz"
    shutil.copy2(baseline_dir / "l1-calibrated.npz", model_path)
    verifier = LogisticVisualVerifier.load(model_path)
    anchor = float(verifier.threshold)
    original_head = ChangeHead.load(baseline_dir / "change-head.npz")
    print(json.dumps({"stage": "train_dev_only", "loaded_seeds": TRAIN + DEV,
                      "fixed_l1_threshold": anchor}), flush=True)
    dev_base = _feature_cache(verifier, [ep for ep in episodes if ep["seed"] in DEV])
    dev_original = _with_head(dev_base, original_head)
    grids = _curve_grid(dev_original, anchor)
    phase1 = {}
    for policy, grid in grids.items():
        points = [_point(dev_original, policy, p) for p in grid if p["threshold"] >= anchor]
        phase1[policy] = _select(points, target, reference_threshold=anchor)
    baseline = _point(dev_original, "frame", {"threshold": anchor})
    _write(output / "threshold-selection.json", {
        "data_seeds": DEV, "fixed_l1_threshold": anchor, "baseline": baseline,
        "selected_dev": phase1,
        "rule": "threshold >= fixed L1; safe dev first, utility then closest calibrated threshold and simplest history",
        "grid_rule": "score-gap midpoints from initial and recovery decisions; no nextafter-negative endpoints",
    })
    print(json.dumps({"stage": "thresholds_corrected", "selected": phase1}), flush=True)

    train_cache = _feature_cache(verifier, [ep for ep in episodes if ep["seed"] in TRAIN])
    candidates = {"original": original_head}
    for weight in (2, 4):
        candidates[f"recovery_negative_x{weight}"] = _fit_recovery_negatives(train_cache, weight)
    reference = _head_metrics(dev_original, REFERENCE_CHANGE_CUTOFF, anchor)
    head_points, head_caches = [], {}
    for name, head in candidates.items():
        head.save(output / f"{name}.npz")
        caches = _with_head(dev_base, head)
        head_caches[name] = caches
        for cutoff in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
            head_points.append({"model": name, "change_threshold": cutoff,
                                **_head_metrics(caches, cutoff, anchor)})
    feasible = [p for p in head_points if p["loss_recall"] >= reference["loss_recall"] - 1e-12]
    chosen = min(feasible, key=lambda p: (
        p["legacy_supported_recovery_vetoes"], p["supported_recovery_frame_alarm_rate"],
        p["false_alarm_rate"], -p["loss_recall"], abs(p["change_threshold"] - REFERENCE_CHANGE_CUTOFF),
    ))
    candidates[chosen["model"]].save(output / "change-head.npz")
    dev_caches = head_caches[chosen["model"]]
    _write(output / "head-calibration.json", {"train_seeds": TRAIN, "dev_seeds": DEV,
        "reference_cutoff": REFERENCE_CHANGE_CUTOFF, "reference": reference,
        "selection_rule": "reduce supported-recovery alarms without lowering dev loss recall",
        "selected": chosen, "candidates": head_points})
    cutoff = chosen["change_threshold"]
    thresholds = [p["threshold"] for p in grids["frame"]]
    grids["learned_change"] = [{"threshold": t, "ttl": age, "change_threshold": cutoff,
                                 "fresh_confirmation": True}
                                for t in thresholds for age in (0, 3, 8, 20)]
    grids["oracle_change"] = [{"threshold": t, "ttl": age, "fresh_confirmation": True}
                               for t in thresholds for age in (0, 3, 8, 20)]
    dev_curves = {p: [_point(dev_caches, p, values) for values in grids[p]] for p in POLICIES}
    selected = {"frame": baseline, "ttl": phase1["ttl"], "history": phase1["history"]}
    for policy in ("learned_change", "oracle_change"):
        # The state threshold remains the fixed L1 certificate for change policies.
        eligible = [point for point in dev_curves[policy] if point["params"]["threshold"] == anchor]
        selected[policy] = _select(eligible, target, reference_threshold=anchor)
    controls = {}
    for name, caches, fresh in (("original_strict", dev_original, False),
                                ("original_fresh", dev_original, True),
                                ("refined_strict", dev_caches, False),
                                ("refined_fresh", dev_caches, True)):
        params = {"threshold": anchor, "ttl": 0, "fresh_confirmation": fresh,
                  "change_threshold": REFERENCE_CHANGE_CUTOFF if name.startswith("original") else cutoff}
        controls[name] = _point(caches, "learned_change", params)
    frozen = {
        "protocol": PHYSICAL_PROTOCOL, "train_seeds": TRAIN, "dev_seeds": DEV, "test_seeds": TEST,
        "fixed_l1_threshold": anchor, "baseline_policy": "frame", "baseline_description": "fixed calibrated L1",
        "baseline_model_source": str(baseline_dir / "l1-calibrated.npz"), "l1_retrained": False,
        "target_false_pass": target, "selected_dev": selected, "head_selection": chosen,
        "dev_controls": controls, "curve_grids": grids, "dev_curves": dev_curves,
        "tuning_inputs": "only train 0-29 and dev 60-67 arrays were read; old test arrays excluded by loader",
        "fresh_certificate_rule": "current L1 pass resets invalidation; change head gates cached evidence only; no cached release after R1",
        "go_criterion": "test empirical FPR <= target and success-gain CI95 lower > 0; or success drop <= 0.02 with >= 1 fewer observations",
        "success_definition": "physical giver release with continuous receiver support and 1 s post-release hold",
        "risk_coverage": "initial accepted decisions / episodes; unsafe initial accepted decisions / initial accepted decisions",
        "observation_cost": "unique 20 Hz frame bundles consumed up to release; release/hold audit frames excluded",
        "test_use": "freeze all choices before new test collection; diagnostic test curves never select an operating point",
    }
    _write(output / "frozen-refinement.json", frozen)
    print(json.dumps({"stage": "frozen", "head": chosen, "dev_controls": controls,
                      "selected_dev": selected, "new_test_seeds": TEST}), flush=True)
    return frozen


def _online_refined(job):
    seed, perturbation, policy, params, output_dir = job
    output = Path(output_dir)
    verifier = LogisticVisualVerifier.load(output / "l1-calibrated.npz")
    head = ChangeHead.load(output / "change-head.npz")
    scores = []
    def gate(stream, index, allow_stale):
        p, f = _probs_and_features(verifier, stream)
        change = head.predict(p, f)
        accepted = predict({"p": p, "change": change}, index, policy, params, allow_stale)
        scores.append({"index": index, "l1_probability": float(p[index]),
                       "change_probability": float(change[index]), "accepted": accepted,
                       "allow_stale": allow_stale})
        return accepted
    row, _ = run_physical_trial(seed, perturbation, render=True, gate=gate)
    row = {k: v for k, v in row.items() if k != "trace"}
    row["gate_scores"] = scores
    row["fresh_l1_pass_vetoes"] = sum(s["l1_probability"] >= verifier.threshold and not s["accepted"] for s in scores)
    return policy, row


def _csv(path, curves, anchor):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "policy", "params", "selection_eligible", "coverage", "risk",
                         "supported_coverage", "replay_success", "mean_extra_observations"))
        for split, policies in curves.items():
            for policy, points in policies.items():
                for p in points:
                    eligible = (p["params"]["threshold"] == anchor if policy in ("frame", "learned_change", "oracle_change")
                                else p["params"]["threshold"] >= anchor)
                    writer.writerow((split, policy, json.dumps(p["params"]), eligible,
                                     p["coverage"], p["initial_risk"], p["supported_coverage"],
                                     p["success_rate"], p["mean_extra_observations"]))


def evaluate(test_dataset, output_dir, workers=4):
    output = Path(output_dir)
    frozen = json.loads((output / "frozen-refinement.json").read_text())
    seeds = frozen["test_seeds"]
    episodes = load_hdf5(test_dataset, seeds=seeds)
    if {ep["seed"] for ep in episodes} != set(seeds) or len(episodes) != len(seeds) * 3:
        raise ValueError("new test dataset must retain all three branches for every frozen test seed")
    verifier = LogisticVisualVerifier.load(output / "l1-calibrated.npz")
    head = ChangeHead.load(output / "change-head.npz")
    test_caches = _with_head(_feature_cache(verifier, episodes), head)
    grids = frozen["curve_grids"]
    curves = {"dev": frozen["dev_curves"],
              "test": {p: [_point(test_caches, p, params) for params in grids[p]] for p in POLICIES}}
    selected_test = {p: _point(test_caches, p, point["params"]) for p, point in frozen["selected_dev"].items()}
    _csv(output / "l2-risk-coverage.csv", curves, frozen["fixed_l1_threshold"])
    _plot(curves, {"dev": frozen["selected_dev"], "test": selected_test}, output)
    report = {"protocol": PHYSICAL_PROTOCOL, "execution_host": socket.gethostname(),
              "test_dataset": str(Path(test_dataset).resolve()),
              "train_seeds": frozen["train_seeds"], "dev_seeds": frozen["dev_seeds"], "test_seeds": seeds,
              "baseline": "fixed calibrated L1", "fixed_l1_threshold": frozen["fixed_l1_threshold"],
              "selected_dev": frozen["selected_dev"], "dev_controls": frozen["dev_controls"],
              "head_selection": frozen["head_selection"], "offline_selected_test": selected_test,
              "offline_success_scope": "support_at_replayed_release",
              "change_head_test": _head_metrics(test_caches, frozen["head_selection"]["change_threshold"], verifier.threshold),
              "online": {}, "go_no_go": {"decision": "PENDING_ONLINE"},
              "risk_constraint_scope": "empirical checks on 12 test seed clusters; not a population 2% guarantee"}
    path = output / "l2-report.json"
    _write(path, report)
    jobs = [(s, q, p, frozen["selected_dev"][p]["params"], str(output))
            for s in seeds for q in ("nominal", "early", "late") for p in DEPLOYABLE]
    rows = {p: [] for p in DEPLOYABLE}
    with (output / "online-rows.jsonl").open("w") as log:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for policy, row in pool.map(_online_refined, jobs):
                rows[policy].append(row)
                log.write(json.dumps({"policy": policy, **row}) + "\n")
                log.flush()
                print(json.dumps({"policy": policy, "seed": row["seed"], "perturbation": row["perturbation"],
                                  "success": row["success"], "failure_reason": row["failure_reason"]}), flush=True)
    report["online"] = {p: {**metrics(values), "rows": values,
                             "fresh_l1_pass_vetoes": sum(r["fresh_l1_pass_vetoes"] for r in values),
                             "by_perturbation": {q: metrics([r for r in values if r["perturbation"] == q])
                                                 for q in ("nominal", "early", "late")}}
                        for p, values in rows.items()}
    gain = _paired_bootstrap_value(rows["learned_change"], rows["frame"], "success")
    cost = _paired_bootstrap_value(rows["learned_change"], rows["frame"], "extra_observations")
    safe = report["online"]["learned_change"]["false_pass_rate"] <= frozen["target_false_pass"]
    go = safe and (gain["ci95"][0] > 0 or (gain["estimate"] >= -0.02 and cost["estimate"] <= -1))
    report["go_no_go"] = {"decision": "GO" if go else "NO-GO", "baseline": "fixed calibrated L1",
                           "success_gain": gain, "extra_observation_gain": cost,
                           "empirical_false_pass_constraint_met": safe,
                           "criterion": frozen["go_criterion"], "enter_l3": False}
    report["online_total_episodes"] = len(jobs)
    _write(path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("tune", "evaluate"))
    parser.add_argument("--source-dataset")
    parser.add_argument("--baseline-dir")
    parser.add_argument("--test-dataset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "tune":
        result = tune(args.source_dataset, args.baseline_dir, args.output_dir)
        print(json.dumps({"frozen": True, "test_seeds": result["test_seeds"]}))
    else:
        result = evaluate(args.test_dataset, args.output_dir, max(1, args.workers))
        print(json.dumps({"go_no_go": result["go_no_go"]}))


if __name__ == "__main__":
    main()
