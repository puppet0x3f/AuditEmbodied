"""Frozen, paired held-out test of robot-only TwoArmLift recovery selection."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil

import numpy as np

from .l0 import _jsonable
from .l4_selector import (CANDIDATES, INPUT_KEYS, METHODS, SCENARIOS, STATE_KEYS,
                          ProprioRule, _job, paired_cost, read_rows, safe, summary, verify_frozen)


TEST_SEEDS = list(range(800, 820))
CRITERIA = {"minimum_initialized_seed_rate": .9, "minimum_valid_fault_rate": .9,
            "minimum_policy_safe_success_rate": .9, "minimum_healthy_success_rate": .9,
            "maximum_safe_success_deficit_vs_global": 0, "maximum_guard_rate_excess_vs_global": 0.}


def prepare(output, reference, local_audit, remote_audit):
    output, reference = Path(output), Path(reference)
    baseline = json.loads((reference / "frozen.json").read_text())
    verify_frozen(baseline)
    audits = [json.loads(Path(p).read_text()) for p in (local_audit, remote_audit)]
    if any(not a["fresh"] or a["proposed_test_seeds"] != TEST_SEEDS for a in audits):
        raise ValueError("test seeds are not verified fresh on both hosts")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "frozen-test.json").exists() or (output / "trials.jsonl").exists():
        raise ValueError("test already exists; do not replace the frozen experiment")
    for name, source in zip(("seed-audit-local.json", "seed-audit-remote.json"), (local_audit, remote_audit)):
        shutil.copyfile(source, output / name)
    sources = [*baseline["source_sha256"], "auditembodied/l4_test.py"]
    for name in sources:
        target = output / "frozen-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    preserved = dict(baseline["preserved_sha256"])
    for name in ("frozen.json", "frozen-before-reporting-fix.json", "calibration.json", "selection-report.json",
                 "train/report.json", "train/cases.csv", "train/trials.jsonl", "dev/report.json",
                 "dev/cases.csv", "dev/trials.jsonl", "execution-audit.json"):
        path = reference / name
        preserved[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in baseline["physical_execution_source_sha256"]:
        path = reference / "frozen-source" / name
        preserved[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"scope": "independent held-out TwoArmLift validation; no tuning or visual model",
                "test_seeds": TEST_SEEDS, "scenarios": list(SCENARIOS), "methods": list(METHODS),
                "physical_episodes": 320, "fault_comparisons": 300, "healthy_policy_controls": 20,
                "rule": baseline["rule"], "protocol": baseline["protocol"], "criteria": CRITERIA,
                "source_sha256": {n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in sources},
                "preserved_sha256": preserved, "reference": str(reference),
                "oracle": "offline minimum-time then minimum-path SAFE outcome of the four executed fixed candidates; never an online input",
                "healthy_control": "same 1.2 s unchanged robot hold as development; policy only; reported separately",
                "cost": "simulated recovery control time and summed 20 Hz EEF paths; common final 1 s hold excluded",
                "acceptance": "valid frozen protocol; safe success >=90%, not below global and above each fixed local; no higher guard rate; lower paired time and path; initialized healthy controls safely continue",
                "decision_type": "predeclared empirical acceptance, not a formal noninferiority test",
                "bootstrap": {"draws": 2000, "unit": "initialization seed; all three faults resampled together", "analysis_rng_seed": 20261108}}
    (output / "frozen-test.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def paired_prefix(group):
    rows = list(group.values())
    reached = ["prefix_state" in r for r in rows]
    uniform_initial_failure = not any(reached) and all(not r.get("nominal_ready", False) for r in rows) and len({r.get("failure_reason") for r in rows}) == 1
    mismatches = [k for k in STATE_KEYS if any(r.get(k) != rows[0].get(k) for r in rows)]
    present = all(k in rows[0] for k in STATE_KEYS) if all(reached) else "initialization_state" in rows[0]
    comparable = present and not mismatches and (all(reached) or uniform_initial_failure)
    return {"comparable": bool(comparable), "prefix_reached": all(reached),
            "uniform_initial_failure": uniform_initial_failure, "mismatches": mismatches}


def guard_causes(row):
    p = row.get("protection", {})
    causes = []
    if p.get("keeper_grasp_lost_steps", 0): causes.append("keeper_grasp_interrupted_or_absent")
    if p.get("max_unsupported_seconds", 0) > .1: causes.append("unsupported_over_0.1s")
    if p.get("deep_interarm_contact_steps", 0): causes.append("deep_interarm_contact")
    if row.get("blocked_calls"): causes.append("forbidden_call")
    if row.get("failure_reason") == "numerical_instability": causes.append("numerical_instability")
    return causes


def method_summary(rows):
    result = summary(rows)
    result["safe_success_rate"] = result["safe_successes"] / len(rows)
    result["safe_failure_reasons"] = dict(Counter(
        r.get("failure_reason") or "safety_condition_failed" for r in rows if not safe(r)))
    result["guard_causes_overlapping"] = dict(Counter(c for r in rows for c in guard_causes(r)))
    return result


def intervals(groups, settings):
    seeds = sorted({s for s, _ in groups})
    indices = np.random.default_rng(settings["analysis_rng_seed"]).integers(0, len(seeds), (settings["draws"], len(seeds)))
    result = {"unit": settings["unit"], "draws": settings["draws"]}
    for method in ("R_left", "R_right", "R_global"):
        delta = np.array([np.mean([int(safe(g["proprio"])) - int(safe(g[method]))
                                  for (s, _), g in groups.items() if s == seed]) for seed in seeds])
        result[f"safe_success_delta_vs_{method}"] = {
            "point": float(delta.mean()), "ci95": np.quantile(delta[indices].mean(1), [.025, .975]).tolist()}
    for metric in ("recovery_seconds", "recovery_path_m"):
        totals = np.array([[sum(g[k][metric] for (s, _), g in groups.items()
                               if s == seed and safe(g["proprio"]) and safe(g["R_global"]))
                            for k in ("proprio", "R_global")] for seed in seeds])
        a, b = totals[indices, 0].sum(1), totals[indices, 1].sum(1)
        ratios = 1. - a[b > 0] / b[b > 0]
        result[f"paired_{metric}_reduction_vs_global"] = {
            "ci95": np.quantile(ratios, [.025, .975]).tolist() if len(ratios) else None,
            "valid_bootstrap_draws": len(ratios)}
    return result


def verdict(report, criteria):
    v, methods = report["validity"], report["by_method"]
    policy, global_ = methods["proprio"], methods["R_global"]
    costs = report["paired_cost_vs_global"]
    checks = {"frozen_comparable_protocol": v["all_cases_comparable"] and v["all_sources_and_prior_results_frozen"] and v["policy_input_valid"] and v["selected_candidate_execution_consistent"],
              "initialization": v["initialized_seed_rate"] >= criteria["minimum_initialized_seed_rate"],
              "valid_faults": v["valid_fault_rate"] >= criteria["minimum_valid_fault_rate"],
              "no_numeric_edit_or_execution_failures": all(v[k] == 0 for k in ("numerical_instabilities", "execution_errors", "contact_anchors", "state_edits_after_initialization", "forbidden_calls")),
              "minimum_policy_safe_success": policy["safe_success_rate"] >= criteria["minimum_policy_safe_success_rate"],
              "safe_coverage_not_lower_than_global": policy["safe_successes"] >= global_["safe_successes"] - criteria["maximum_safe_success_deficit_vs_global"],
              "covers_more_than_each_fixed_local": all(policy["safe_successes"] > methods[k]["safe_successes"] for k in ("R_left", "R_right")),
              "guard_rate_not_higher_than_global": policy["protection_violation_rate"] <= global_["protection_violation_rate"] + criteria["maximum_guard_rate_excess_vs_global"],
              "lower_paired_time_and_path": costs["paired_safe_scenes"] > 0 and all(costs[m]["reduction_fraction"] > 0 for m in ("recovery_seconds", "recovery_path_m")),
              "healthy_control": report["healthy_control"]["safe_success_rate"] >= criteria["minimum_healthy_success_rate"] and report["healthy_control"]["all_initialized_safely_continue"]}
    return {"result": "Go" if all(checks.values()) else "No-Go", "checks": checks,
            "failed_checks": [k for k, passed in checks.items() if not passed],
            "scope": "frozen robot-only selector on independent test seeds; visual implementation remains deferred"}


def analyze(output):
    output = Path(output)
    manifest = json.loads((output / "frozen-test.json").read_text())
    verify_frozen(manifest)
    rows = read_rows(output / "trials.jsonl")
    expected = {(s, q, k) for s in manifest["test_seeds"] for q in SCENARIOS for k in METHODS}
    expected |= {(s, "sham", "proprio") for s in manifest["test_seeds"]}
    if len(rows) != len(expected) or {(r["seed"], r["scenario"], r["method"]) for r in rows} != expected:
        raise ValueError("incomplete or duplicate held-out records; do not exclude failures")
    groups = {}
    for row in rows:
        if row["scenario"] in SCENARIOS:
            groups.setdefault((row["seed"], row["scenario"]), {})[row["method"]] = row
    checks, oracle_rows, oracle_cases, oracle_pairs, global_pairs = [], [], [], [], []
    consistent = True
    for (seed, scenario), group in sorted(groups.items()):
        checks.append({"seed": seed, "scenario": scenario, **paired_prefix(group)})
        policy, global_ = group["proprio"], group["R_global"]
        candidates = [group[k] for k in CANDIDATES if safe(group[k])]
        oracle = min(candidates, key=lambda r: (r["recovery_seconds"], r["recovery_path_m"])) if candidates else None
        oracle_cases.append({"seed": seed, "scenario": scenario, "safe_success": oracle is not None,
                             "candidate": oracle["candidate"] if oracle else None})
        if oracle:
            oracle_rows.append(oracle)
            if safe(policy): oracle_pairs.append((policy, oracle))
        if safe(policy) and safe(global_): global_pairs.append((policy, global_))
        if "selection" in policy:
            fixed = group[policy["candidate"]]
            consistent &= all(policy.get(k) == fixed.get(k) for k in (
                "success", "protection_violation", "recovery_seconds", "recovery_path_m", "protection",
                "table_setdown_verified", "table_regrasp_and_lift_verified", "hold_seconds"))
    controls = [r for r in rows if r["scenario"] == "sham"]
    main = [r for r in rows if r["scenario"] in SCENARIOS]
    policy = [r for r in rows if r["method"] == "proprio"]
    observed = [r for r in policy if r.get("nominal_ready", False)]
    rule = ProprioRule(**manifest["rule"])
    input_valid = all(set(r.get("selection", {}).get("input", {})) == INPUT_KEYS and
                      r["selection"]["rgb_frames"] == 0 and r["selection"]["robot_observations"] == 1 and
                      rule(r["selection"]["input"]) == r["selection"]["decision"] and
                      r["candidate"] == r["selection"]["decision"]["candidate"] for r in observed)
    initialized = [s for s in manifest["test_seeds"] if all(r.get("nominal_ready", False) for r in rows if r["seed"] == s)]
    valid_faults = sum(all(r.get("fault_class_valid", False) for r in g.values()) for g in groups.values())
    validity = {"episodes": len(rows), "paired_groups": len(groups), "all_cases_comparable": all(c["comparable"] for c in checks),
                "prefix_checks": checks, "initialized_seeds": initialized,
                "initialized_seed_rate": len(initialized) / len(manifest["test_seeds"]),
                "valid_fault_scenes": valid_faults, "valid_fault_rate": valid_faults / len(groups),
                "all_sources_and_prior_results_frozen": True, "policy_input_valid": input_valid,
                "selected_candidate_execution_consistent": bool(consistent),
                "contact_anchors": sum(r["contact_anchor_count"] for r in rows),
                "state_edits_after_initialization": sum(r["state_edit_count_after_initialization"] for r in rows),
                "forbidden_calls": sum(len(r["blocked_calls"]) for r in rows),
                "numerical_instabilities": sum(r["failure_reason"] == "numerical_instability" for r in rows),
                "execution_errors": sum(r["failure_reason"] == "initialization_or_execution_error" for r in rows),
                "policy_rgb_frames": sum(r.get("selection", {}).get("rgb_frames", 0) for r in policy),
                "policy_robot_observations": sum(r.get("selection", {}).get("robot_observations", 0) for r in policy)}
    healthy = method_summary(controls)
    healthy["choices"] = dict(Counter(r["candidate"] for r in controls))
    healthy["all_initialized_safely_continue"] = all(safe(r) and r["candidate"] == "continue" for r in controls if r.get("nominal_ready", False))
    failure_fields = ("seed", "scenario", "method", "candidate", "success", "failure_reason", "failed_stage",
                      "error", "protection", "table_setdown_verified", "table_regrasp_and_lift_verified", "final", "selection")
    failures = [{**{k: r.get(k) for k in failure_fields}, "guard_causes": guard_causes(r)} for r in rows if not safe(r)]
    report = {"scope": manifest["scope"], "test_seeds": manifest["test_seeds"], "criteria": manifest["criteria"],
              "rule": manifest["rule"], "validity": validity,
              "by_method": {k: method_summary([r for r in main if r["method"] == k]) for k in METHODS},
              "by_scenario": {q: {k: method_summary([r for r in main if r["scenario"] == q and r["method"] == k]) for k in METHODS} for q in SCENARIOS},
              "oracle": {"kind": manifest["oracle"], "scenes": len(groups), "safe_successes": len(oracle_rows),
                         "safe_success_rate": len(oracle_rows) / len(groups),
                         "mean_recovery_seconds_on_safe": float(np.mean([r["recovery_seconds"] for r in oracle_rows])) if oracle_rows else None,
                         "mean_recovery_path_m_on_safe": float(np.mean([r["recovery_path_m"] for r in oracle_rows])) if oracle_rows else None,
                         "choices": dict(Counter(r["candidate"] for r in oracle_rows)), "cases": oracle_cases},
              "policy_choices": dict(Counter(r["candidate"] for r in main if r["method"] == "proprio")),
              "paired_cost_vs_global": paired_cost(global_pairs, "proprio", "fixed_global"),
              "paired_cost_vs_oracle": paired_cost(oracle_pairs, "proprio", "oracle"),
              "healthy_control": healthy, "seed_bootstrap": intervals(groups, manifest["bootstrap"]),
              "failures": failures, "cost_definition": manifest["cost"],
              "guard_definition": "locals require keeper grasp at every sampled recovery step; unsupported >0.1 s or inter-arm penetration >1 cm also violate protection; table support allowed for global reset",
              "decision_type": manifest["decision_type"]}
    report["decision"] = verdict(report, manifest["criteria"])
    (output / "test-report.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = ("seed", "scenario", "method", "candidate", "success", "protection_violation", "recovery_seconds", "recovery_path_m", "failure_reason")
    with (output / "test-cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({k: r.get(k) for k in fields} for r in rows)
    return report


def run(output, workers=4):
    output = Path(output)
    manifest = json.loads((output / "frozen-test.json").read_text())
    verify_frozen(manifest)
    if (output / "trials.jsonl").exists(): raise ValueError("test data exists; analysis only, no reruns")
    jobs = [(s, q, k, manifest["protocol"], manifest["rule"]) for s in manifest["test_seeds"]
            for q in (*SCENARIOS, "sham") for k in (("proprio",) if q == "sham" else METHODS)]
    with (output / "trials.jsonl").open("w") as stream:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for row in pool.map(_job, jobs):
                stream.write(json.dumps(_jsonable(row)) + "\n"); stream.flush()
                print(json.dumps({k: row.get(k) for k in ("seed", "scenario", "method", "candidate", "success", "protection_violation", "failure_reason")}), flush=True)
    return analyze(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", choices=("prepare", "run", "analyze"), required=True)
    parser.add_argument("--reference", default="outputs/l4-proprio-20261107")
    parser.add_argument("--seed-audit-local")
    parser.add_argument("--seed-audit-remote")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.phase == "prepare":
        if not args.seed_audit_local or not args.seed_audit_remote: parser.error("prepare requires both seed audits")
        result = prepare(args.output_dir, args.reference, args.seed_audit_local, args.seed_audit_remote)
    elif args.phase == "run": result = run(args.output_dir, args.workers)
    else: result = analyze(args.output_dir)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
