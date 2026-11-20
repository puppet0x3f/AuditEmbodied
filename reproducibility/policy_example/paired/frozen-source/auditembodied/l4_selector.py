"""Robot-only recovery selection for the frozen TwoArmLift skills."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil

import numpy as np

from .l0 import _jsonable
from .l4_lift import CANDIDATES, SCENARIOS, LiftConfig, analyze_matrix, run_trial, summary


METHODS = (*CANDIDATES, "proprio")
STATE_KEYS = ("initialization_state", "preclose_state", "initialization_end_state", "prefix_state")
INPUT_KEYS = {"positions", "fingers", "reference_positions"}


@dataclass(frozen=True)
class ProprioRule:
    asymmetry_limit: float = .04
    loaded_lower: float = .004
    loaded_upper: float = .028
    height_difference_limit: float = .03

    def __call__(self, observation):
        fingers = np.abs(np.asarray(observation["fingers"], dtype=float))
        positions = np.asarray(observation["positions"], dtype=float)
        reference = np.asarray(observation["reference_positions"], dtype=float)
        widths = fingers.mean(axis=1)
        loaded = (widths > self.loaded_lower) & (widths < self.loaded_upper)
        asymmetry = np.abs(fingers[:, 0] - fingers[:, 1]) / np.maximum(fingers.sum(axis=1), 1e-9)
        height_error = abs((positions[0, 2] - positions[1, 2]) - (reference[0, 2] - reference[1, 2]))
        if loaded.all():
            candidate = "continue" if height_error <= self.height_difference_limit else "R_right"
            reason = "balanced_loaded_grippers" if candidate == "continue" else "loaded_grippers_height_mismatch"
        elif loaded.sum() == 1:
            keeper = int(np.flatnonzero(loaded)[0])
            if asymmetry[keeper] > self.asymmetry_limit:
                candidate, reason = "R_global", "keeper_finger_asymmetry"
            else:
                candidate = "R_left" if keeper == 1 else "R_right"
                reason = "single_loaded_gripper"
        else:
            candidate, reason = "R_global", "no_loaded_gripper"
        return {"candidate": candidate, "reason": reason,
                "features": {"mean_finger_qpos": widths.tolist(), "loaded": loaded.tolist(),
                             "finger_asymmetry": asymmetry.tolist(), "height_error_m": float(height_error)}}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def safe(row):
    return bool(row["success"] and not row["protection_violation"])


def training_observation(row):
    # Phase names locate offline decision examples; they never enter the policy.
    reference = [f["robot"] for f in row["trace"] if f["stage"] == "nominal_stable_hold"][-1]
    current = [f["robot"] for f in row["trace"] if f["phase"] == "fault"][-1]
    return {"positions": current["positions"], "fingers": current["fingers"],
            "reference_positions": reference["positions"]}


def calibrate(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["seed"], row["scenario"]), {})[row["candidate"]] = row
    scores = []
    for threshold in (.02, .03, .04, .05, 1.):
        rule = ProprioRule(asymmetry_limit=threshold)
        chosen = [g[rule(training_observation(g["continue"]))["candidate"]] for g in groups.values()]
        scores.append({"asymmetry_limit": threshold, "episodes": len(chosen),
                       "safe_successes": sum(safe(r) for r in chosen),
                       "mean_recovery_seconds": float(np.mean([r["recovery_seconds"] for r in chosen])),
                       "mean_recovery_path_m": float(np.mean([r["recovery_path_m"] for r in chosen])),
                       "choices": dict(Counter(r["candidate"] for r in chosen))})
    best = min(scores, key=lambda r: (-r["safe_successes"], r["mean_recovery_seconds"], r["mean_recovery_path_m"]))
    return ProprioRule(asymmetry_limit=best["asymmetry_limit"]), {
        "split": "training seeds 0-9 only; executed candidate outcomes provide offline labels",
        "selection": "maximize safe successes, then minimize mean recovery time and path",
        "grid": scores, "chosen_asymmetry_limit": best["asymmetry_limit"]}


def prepare(output, reference):
    output, reference = Path(output), Path(reference)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "frozen.json").exists():
        raise ValueError("experiment is already frozen")
    rows = read_rows(reference / "matrix/trials.jsonl")
    if {(r["seed"], r["scenario"], r["candidate"]) for r in rows} != {
            (s, q, k) for s in range(10) for q in SCENARIOS for k in CANDIDATES}:
        raise ValueError("calibration requires the original complete training matrix")
    rule, calibration = calibrate(rows)
    sources = ["auditembodied/" + name for name in
               ("l4_lift.py", "l4_selector.py", "l0.py", "continuous.py", "recovery_skills.py")]
    for name in sources:
        target = output / "frozen-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    preserved = {str(reference / name): hashlib.sha256((reference / name).read_bytes()).hexdigest()
                 for name in ("frozen-physical.json", "physical-report.json", "matrix/report.json",
                              "matrix/cases.csv", "matrix/trials.jsonl", "nominal-final/report.json",
                              "nominal-final/trials.jsonl", "frozen-source/auditembodied/l4_lift.py")}
    l3 = json.loads((reference / "l3-frozen.json").read_text())
    for item in l3["references"]:
        preserved.update({str(Path(item["root"]) / name): digest for name, digest in item["sha256"].items()})
    old = json.loads((reference / "frozen-physical.json").read_text())
    manifest = {"scope": "train/development robot-only selection; no visual model or held-out test",
                "train_seeds": list(range(10)), "dev_seeds": list(range(60, 68)),
                "scenarios": list(SCENARIOS), "healthy_control": "sham: 1.2 s unchanged robot hold",
                "protocol": old["config"], "rule": asdict(rule), "reference": str(reference),
                "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources},
                "preserved_sha256": preserved,
                "online_input_keys": sorted(INPUT_KEYS), "rgb_frames": 0,
                "calibration": "training candidate matrix only; freeze before online train/dev runs"}
    (output / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    (output / "frozen.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return calibration


def verify_frozen(manifest):
    for name, expected in manifest["source_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"execution source changed: {name}")
    for name, expected in manifest["preserved_sha256"].items():
        path = Path(name)
        if not path.exists():
            path = Path("/data2/zh/AuditEmbodied") / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"prior result changed: {path}")


def _job(args):
    seed, scenario, method, protocol, parameters = args
    row = run_trial(seed, scenario, method if method in CANDIDATES else "continue", LiftConfig(**protocol),
                    selector=ProprioRule(**parameters) if method == "proprio" else None)
    row["method"] = method
    return row


def paired_cost(pairs, first, second):
    result = {"paired_safe_scenes": len(pairs)}
    for metric in ("recovery_seconds", "recovery_path_m"):
        a = float(np.mean([x[metric] for x, _ in pairs])) if pairs else None
        b = float(np.mean([y[metric] for _, y in pairs])) if pairs else None
        result[metric] = {first: a, second: b, "reduction_fraction": 1. - a / b if b else None}
    return result


def analyze(output, phase):
    output = Path(output)
    manifest = json.loads((output / "frozen.json").read_text())
    verify_frozen(manifest)
    new = read_rows(output / phase / "trials.jsonl")
    rows = [r for r in new if r["scenario"] in SCENARIOS]
    if phase == "train":
        for row in read_rows(Path(manifest["reference"]) / "matrix/trials.jsonl"):
            rows.append({**row, "method": row["candidate"]})
    seeds = manifest[phase + "_seeds"]
    expected = {(s, q, k) for s in seeds for q in SCENARIOS for k in METHODS}
    if len(rows) != len(expected) or {(r["seed"], r["scenario"], r["method"]) for r in rows} != expected:
        raise ValueError("incomplete or duplicate physical comparison")
    groups = {}
    for row in rows:
        groups.setdefault((row["seed"], row["scenario"]), {})[row["method"]] = row
    mismatches = []
    chosen, globals_, oracles, oracle_pairs, global_pairs = [], [], [], [], []
    for key, group in sorted(groups.items()):
        reference = group["continue"]
        for name in STATE_KEYS:
            if name not in reference or any(r.get(name) != reference[name] for r in group.values()):
                mismatches.append([*key, name])
        policy, global_ = group["proprio"], group["R_global"]
        chosen.append(policy); globals_.append(global_)
        candidates = [group[k] for k in CANDIDATES if safe(group[k])]
        oracle = min(candidates, key=lambda r: (r["recovery_seconds"], r["recovery_path_m"])) if candidates else None
        if oracle:
            oracles.append(oracle)
            if safe(policy):
                oracle_pairs.append((policy, oracle))
        if safe(policy) and safe(global_):
            global_pairs.append((policy, global_))
    controls = [r for r in new if r["scenario"] == "sham"]
    if {r["seed"] for r in controls} != set(seeds) or len(controls) != len(seeds):
        raise ValueError("missing healthy controls")
    rule = ProprioRule(**manifest["rule"])
    policy_rows = chosen + controls
    input_valid = all(set(r.get("selection", {}).get("input", {})) == INPUT_KEYS and
                      r["selection"]["rgb_frames"] == 0 and r["selection"]["robot_observations"] == 1 and
                      rule(r["selection"]["input"]) == r["selection"]["decision"] and
                      r["candidate"] == r["selection"]["decision"]["candidate"] for r in policy_rows)
    valid = {"paired_groups": len(groups), "all_prefixes_equal": not mismatches, "mismatches": mismatches,
             "sources_and_prior_results_unchanged": True, "policy_input_valid": input_valid,
             "valid_fault_scenes": sum(g["continue"].get("fault_class_valid", False) for g in groups.values()),
             "new_physical_episodes": len(new), "reused_training_candidate_episodes": len(rows) - len(chosen) if phase == "train" else 0,
             "contact_anchors": sum(r["contact_anchor_count"] for r in new),
             "state_edits_after_initialization": sum(r["state_edit_count_after_initialization"] for r in new),
             "forbidden_calls": sum(len(r["blocked_calls"]) for r in new),
             "numerical_instabilities": sum(r["failure_reason"] == "numerical_instability" for r in new),
             "execution_errors": sum(r["failure_reason"] == "initialization_or_execution_error" for r in new),
             "policy_rgb_frames": sum(r.get("selection", {}).get("rgb_frames", 0) for r in policy_rows),
             "policy_robot_observations": sum(r.get("selection", {}).get("robot_observations", 0) for r in policy_rows)}
    costs = paired_cost(global_pairs, "proprio", "fixed_global")
    checks = {"valid_frozen_comparison": valid["all_prefixes_equal"] and input_valid and valid["valid_fault_scenes"] == len(groups),
              "no_execution_or_edit_failures": all(valid[k] == 0 for k in ("contact_anchors", "state_edits_after_initialization", "forbidden_calls", "numerical_instabilities", "execution_errors")),
              "safe_coverage_not_lower_than_global": sum(safe(r) for r in chosen) >= sum(safe(r) for r in globals_),
              "lower_paired_time_and_path": bool(global_pairs) and all(costs[m]["reduction_fraction"] > 0 for m in ("recovery_seconds", "recovery_path_m")),
              "healthy_controls_safe_without_recovery": all(safe(r) and r["candidate"] == "continue" for r in controls)}
    report = {"phase": phase, "seeds": seeds, "rule": manifest["rule"], "validity": valid,
              "by_method": {k: summary([r for r in rows if r["method"] == k]) for k in METHODS},
              "by_scenario": {q: {k: summary([r for r in rows if r["scenario"] == q and r["method"] == k]) for k in METHODS} for q in SCENARIOS},
              "oracle": {"scenes": len(groups), "safe_successes": len(oracles),
                         "mean_recovery_seconds_on_safe": float(np.mean([r["recovery_seconds"] for r in oracles])) if oracles else None,
                         "mean_recovery_path_m_on_safe": float(np.mean([r["recovery_path_m"] for r in oracles])) if oracles else None},
              "choices": dict(Counter(r["candidate"] for r in chosen)),
              "decision_reasons": dict(Counter(r.get("selection", {}).get("decision", {}).get("reason") for r in chosen)),
              "paired_cost_vs_global": costs, "paired_cost_vs_oracle": paired_cost(oracle_pairs, "proprio", "oracle"),
              "healthy_control": {**summary(controls), "choices": dict(Counter(r["candidate"] for r in controls))},
              "policy_failures": [{k: r.get(k) for k in ("seed", "scenario", "candidate", "success", "failure_reason", "protection", "error", "selection")} for r in policy_rows if not safe(r)],
              "decision": {"result": "Go" if all(checks.values()) else "No-Go", "checks": checks,
                           "scope": "robot-only baseline on train/development; not held-out or visual validation"}}
    (output / phase / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = ("seed", "scenario", "method", "candidate", "success", "protection_violation", "recovery_seconds", "recovery_path_m", "failure_reason")
    with (output / phase / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({k: r.get(k) for k in fields} for r in rows + controls)
    return report


def run_phase(output, phase, workers=4):
    output = Path(output)
    manifest = json.loads((output / "frozen.json").read_text())
    verify_frozen(manifest)
    directory = output / phase
    directory.mkdir(exist_ok=True)
    if (directory / "trials.jsonl").exists():
        raise ValueError("physical records exist; use --analyze-only")
    jobs = [(s, q, k, manifest["protocol"], manifest["rule"]) for s in manifest[phase + "_seeds"]
            for q in (*SCENARIOS, "sham")
            for k in (("proprio",) if phase == "train" or q == "sham" else METHODS)]
    with (directory / "trials.jsonl").open("w") as stream:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for row in pool.map(_job, jobs):
                stream.write(json.dumps(_jsonable(row)) + "\n"); stream.flush()
                print(json.dumps({k: row.get(k) for k in ("seed", "scenario", "method", "candidate", "success", "protection_violation", "failure_reason")}), flush=True)
    return analyze(output, phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", choices=("prepare", "train", "dev"), required=True)
    parser.add_argument("--reference", default="outputs/l4-lift-20261107")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        result = prepare(args.output_dir, args.reference)
    else:
        result = analyze(args.output_dir, args.phase) if args.analyze_only else run_phase(args.output_dir, args.phase, args.workers)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
