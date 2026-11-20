"""One frozen held-out L3 recovery evaluation; Oracle is an offline reference."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing
from pathlib import Path

import numpy as np

from .l3_matrix import MatrixConfig, _job, _summary
from .l0 import _jsonable


METHODS = ("R1_retarget", "R3", "rule")


def oracle_choice(group):
    """Single-candidate hindsight Oracle; never include the rule in its pool."""
    pool = [group[k] for k in ("R1_retarget", "R3")]
    safe = [r for r in pool if r["success"] and not r["protection_violation"]]
    completed = [r for r in pool if r["success"]]
    feasible = safe or completed
    selected = min(feasible, key=lambda r: (r["recovery_seconds"], r["recovery_path_m"])) if feasible else pool[0]
    return {**selected, "candidate": "oracle", "oracle_source": selected["candidate"],
            "oracle_tier": "safe_success" if safe else "task_success" if completed else "failed_local_fallback"}


def same_state(states):
    return bool(all(s is not None for s in states) and all(
        set(s) == set(states[0]) and all(np.array_equal(states[0][k], s[k]) for k in states[0])
        for s in states[1:]))


def compare_prefixes(group):
    rows = [group[k] for k in METHODS]
    initialized = same_state([r.get("initialization_state") for r in rows])
    init_end = same_state([r.get("initialization_end_state") for r in rows])
    reached = ["prefix_state" in r for r in rows]
    uniform_init_failure = bool(not any(reached) and all(
        r.get("failure_reason") in ("initialization_error", "initialization_not_supported") for r in rows))
    prefix = same_state([r.get("prefix_state") for r in rows]) if all(reached) else None
    return {"initialization_identical": initialized, "initialization_end_identical": init_end,
            "fault_prefix_reached": all(reached), "fault_prefix_identical": prefix,
            "uniform_initialization_failure": uniform_init_failure,
            "comparable": bool(initialized and init_end and (prefix is True or uniform_init_failure))}


def _safe(row):
    return row["success"] and not row["protection_violation"]


def _paired_cost(groups, reference):
    pairs = [(g["rule"], g[reference]) for g in groups.values() if _safe(g["rule"]) and _safe(g[reference])]
    result = {"scenes": len(pairs)}
    if pairs:
        for key in ("recovery_seconds", "recovery_path_m"):
            a = float(np.mean([x[key] for x, _ in pairs])); b = float(np.mean([y[key] for _, y in pairs]))
            result[key] = {"rule": a, reference: b, "reduction_fraction": 1 - a / b}
    return result


def _seed_intervals(groups, draws=2000):
    """Resample initialization seeds, keeping the three faults together."""
    seed_ids = sorted({seed for seed, _ in groups})
    rng = np.random.default_rng(20261107)
    indices = rng.integers(0, len(seed_ids), size=(draws, len(seed_ids)))
    result = {"resampling_unit": "initialization seed", "draws": draws}
    for reference in ("R1_retarget", "R3"):
        deltas = np.array([np.mean([int(_safe(g["rule"])) - int(_safe(g[reference]))
                                   for (s, _), g in groups.items() if s == seed]) for seed in seed_ids])
        samples = deltas[indices].mean(axis=1)
        result[f"safe_success_delta_vs_{reference}"] = {
            "point": float(deltas.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist()}
    for key in ("recovery_seconds", "recovery_path_m"):
        totals = []
        for seed in seed_ids:
            pairs = [(g["rule"], g["R3"]) for (s, _), g in groups.items()
                     if s == seed and _safe(g["rule"]) and _safe(g["R3"])]
            totals.append((sum(a[key] for a, _ in pairs), sum(b[key] for _, b in pairs)))
        totals = np.asarray(totals)
        a = totals[indices, 0].sum(axis=1); b = totals[indices, 1].sum(axis=1)
        ratios = 1 - a[b > 0] / b[b > 0]
        result[f"paired_{key}_reduction_vs_R3"] = {"ci95": np.quantile(ratios, [.025, .975]).tolist()} if len(ratios) else None
    return result


def verify_frozen(output, manifest):
    package = Path(__file__).parent
    for filename in manifest["frozen_source_files"]:
        if (package / filename).read_bytes() != (output / "frozen-source" / filename).read_bytes():
            raise ValueError(f"frozen source changed: {filename}")


def go_no_go(report, criteria):
    p, v = report["by_method"], report["validity"]
    rate = lambda k: p[k]["safe_successes"] / p[k]["episodes"]
    costs = report["paired_costs_vs_R3"]
    checks = {
        "valid_protocol": v["all_cases_comparable"] and v["all_sources_frozen"] and
                          not any(v[k] for k in ("numerical_instabilities", "execution_errors", "contact_anchors",
                                                "manual_state_edits", "forbidden_calls", "wrench_leaks")),
        "initialization": v["initialized_seed_rate"] >= criteria["minimum_initialization_seed_rate"],
        "more_safe_successes_than_local": rate("rule") > rate("R1_retarget"),
        "near_global_success": rate("rule") >= rate("R3") - criteria["global_success_margin"],
        "no_higher_protection_violation_rate": p["rule"]["protection_violation_rate"] <= p["R3"]["protection_violation_rate"],
        "lower_paired_recovery_cost": bool(costs["scenes"] and costs["recovery_seconds"]["reduction_fraction"] > 0
                                           and costs["recovery_path_m"]["reduction_fraction"] > 0),
    }
    return {"decision": "Go" if all(checks.values()) else "No-Go", "checks": checks,
            "failed_checks": [k for k, value in checks.items() if not value]}


def analyze(output):
    output = Path(output)
    manifest = json.loads((output / "frozen-test.json").read_text())
    verify_frozen(output, manifest)
    rows, groups = [], {}
    for line in (output / "trials.jsonl").open():
        row = json.loads(line); trace = row.pop("trace")
        row["trace_monotonic"] = bool(np.all(np.diff([p["time"] for p in trace]) > 0))
        rows.append(row)
        group = groups.setdefault((row["seed"], row["scenario"]), {})
        if row["candidate"] in group: raise ValueError("duplicate held-out trial")
        group[row["candidate"]] = row
    expected = {(s, q) for s in manifest["seeds"] for q in manifest["scenarios"]}
    if set(groups) != expected or any(set(g) != set(METHODS) for g in groups.values()):
        raise ValueError("held-out run is incomplete")
    prefix_checks = [{"seed": s, "scenario": q, **compare_prefixes(g)} for (s, q), g in groups.items()]
    oracle_rows = []
    for group in groups.values():
        chosen = oracle_choice(group); oracle_rows.append(chosen); group["oracle"] = chosen
    all_methods = (*METHODS, "oracle")
    initialized_seeds = {s for s in manifest["seeds"] if all(
        g[k].get("initialization_supported", False) for (seed, _), g in groups.items() if seed == s for k in METHODS)}
    validity = {"actual_episodes": len(rows), "cases": len(groups),
        "all_cases_comparable": all(c["comparable"] for c in prefix_checks),
        "fault_prefixes_reached": sum(c["fault_prefix_reached"] for c in prefix_checks),
        "all_sources_frozen": True, "initialized_seeds": sorted(initialized_seeds),
        "initialized_seed_rate": len(initialized_seeds) / len(manifest["seeds"]),
        "numerical_instabilities": sum(r.get("failure_reason") == "numerical_instability" or
            any(r.get("bad_state_warnings", {}).values()) or r.get("time_rollback_detected", False) or not r["trace_monotonic"] for r in rows),
        "execution_errors": sum(r.get("failure_reason") == "execution_error" for r in rows),
        "contact_anchors": sum(r["contact_anchor_count"] for r in rows),
        "manual_state_edits": sum(r["state_edit_count_after_initialization"] for r in rows),
        "forbidden_calls": sum(len(r["blocked_calls"]) for r in rows),
        "wrench_leaks": sum(r.get("external_wrench_leak_steps", 0) or bool(r.get("external_wrench_at_recovery_start", 0)) for r in rows)}
    report = {"scope": manifest["scope"], "seeds": manifest["seeds"], "criteria": manifest["criteria"],
        "by_method": {k: _summary([g[k] for g in groups.values()]) for k in all_methods},
        "by_scenario": {q: {k: _summary([g[k] for (seed, scene), g in groups.items() if scene == q])
                            for k in all_methods} for q in manifest["scenarios"]},
        "validity": validity, "prefix_checks": prefix_checks,
        "paired_costs_vs_R3": _paired_cost(groups, "R3"),
        "paired_costs_vs_Oracle": _paired_cost(groups, "oracle"),
        "oracle_choice_counts": dict(Counter(r["oracle_source"] for r in oracle_rows)),
        "oracle_pool": ["R1_retarget", "R3"],
        "oracle_note": "offline single-candidate hindsight reference; rule is sequential and is excluded from the Oracle pool; if neither candidate completes, copy frozen R1 failure including its cost and violations",
        "rule_only_safe_cases": [{"seed": s, "scenario": q} for (s, q), g in groups.items()
                                 if _safe(g["rule"]) and not _safe(g["oracle"])],
        "rule_global_escalations": sum(r.get("used_global_recovery", False) for r in rows if r["candidate"] == "rule"),
        "fault_effects": {q: [{"seed": s, "giver_holds": g["R1_retarget"].get("post_fault_predicate", {}).get("giver_holds"),
                              "grasp_relative_change": g["R1_retarget"].get("grasp_relative_change")}
                             for (s, scene), g in groups.items() if scene == q] for q in manifest["scenarios"]},
        "seed_bootstrap": _seed_intervals(groups),
        "failures": [{"seed": r["seed"], "scenario": r["scenario"], "candidate": r["candidate"],
                       "reason": r["failure_reason"], "stage": r.get("failed_stage"),
                       "table_cycle": r.get("table_cycle_verified"),
                       "post_fault_predicate": r.get("post_fault_predicate"),
                       "post_recovery_predicate": r.get("post_recovery_predicate"),
                       "receiver_supported_before_release": r.get("post_recovery_predicate", {}).get("receiver_supports"),
                       "protection": r["protection"]} for r in rows if not r["success"] or r["protection_violation"]]}
    report["go_no_go"] = go_no_go(report, manifest["criteria"])
    (output / "test-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    fields = ("seed", "scenario", "candidate", "success", "protection_violation", "recovery_seconds", "recovery_path_m",
              "used_global_recovery", "failure_reason", "oracle_source", "oracle_tier")
    with (output / "test-cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({k: r.get(k) for k in fields} for r in rows + oracle_rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(); output = Path(args.output_dir)
    if not args.analyze_only:
        manifest = json.loads((output / "frozen-test.json").read_text())
        verify_frozen(output, manifest)
        if (output / "trials.jsonl").exists(): raise ValueError("held-out data already exists; use --analyze-only")
        config = MatrixConfig(**manifest["protocol"])
        jobs = [(s, q, k, config) for s in manifest["seeds"] for q in manifest["scenarios"] for k in METHODS]
        with (output / "trials.jsonl").open("w") as log:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                for row in pool.map(_job, jobs):
                    log.write(json.dumps(_jsonable(row)) + "\n"); log.flush()
                    print(json.dumps({k: row.get(k) for k in ("seed", "scenario", "candidate", "success", "failure_reason")}), flush=True)
        if json.loads((output / "frozen-test.json").read_text()) != manifest:
            raise ValueError("test definition changed during evaluation")
    report = analyze(output)
    print(json.dumps({"go_no_go": report["go_no_go"], "by_method": report["by_method"]}), flush=True)


if __name__ == "__main__":
    main()
