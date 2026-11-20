"""Paired physical retiming experiment using the unchanged L4 recovery runner.

Only three existing local-skill step counts change. Initialization, fault,
controller gains, geometry, global recovery and the final hold stay fixed.
No learned policy, VLM, renderer, state restoration or contact anchor is used.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np

from .l4_lift import LiftConfig, run_trial


SCALES = (.75, 1., 1.5)
LOCAL_FIELDS = ("local_approach_steps", "local_close_steps", "local_relevel_steps")
STATE_KEYS = ("initialization_state", "preclose_state", "initialization_end_state", "prefix_state")
SCENARIOS = ("left_loss", "right_loss")
SEEDS = {"pilot": list(range(8)), "confirmation": list(range(20, 24))}
SOURCE_FILES = ("l4_lift.py", "l0.py", "continuous.py", "recovery_skills.py")


def retimed_config(scale, mode="all_local"):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("time scale must be positive and finite")
    base = LiftConfig()
    if mode not in ("all_local", "relevel_only"):
        raise ValueError("unknown retiming intervention")
    fields = LOCAL_FIELDS if mode == "all_local" else ("local_relevel_steps",)
    return replace(base, **{key: max(1, math.floor(getattr(base, key) * scale + .5))
                           for key in fields})


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prefix_frames(row):
    return [f for f in row["trace"] if f["phase"] in ("initialization", "fault")]


def metrics(row):
    local = row["candidate"] in ("R_left", "R_right")
    keeper = 1 if row["candidate"] == "R_left" else 0 if local else None
    start = row.get("post_fault", {})
    valid = bool(local and start.get("holds", [False, False])[keeper])
    frames = [f for f in row["trace"] if f["phase"] == "recovery"]
    losses = [f for f in frames if local and not f["evaluator"]["holds"][keeper]]
    unstable = [f for f in frames if local and not f["evaluator"]["supports"][keeper]]
    first = losses[0] if valid and losses else None
    return {"stage": row["experiment_stage"], "seed": row["seed"], "scenario": row["scenario"],
            "candidate": row["candidate"], "time_scale": row["time_scale"],
            "retiming_mode": row.get("retiming_mode", "all_local"),
            "fault_valid": row.get("fault_class_valid", False), "keeper": keeper,
            "keeper_valid_at_start": valid if local else None,
            "keeper_stable_at_start": bool(start.get("supports", [False, False])[keeper]) if local else None,
            "keeper_continuous": not losses if valid else None,
            "keeper_lost_steps": len(losses) if local else None,
            "keeper_unstable_steps": len(unstable) if local else None,
            "first_keeper_loss_seconds": first["time"] - row["prefix_state"]["time"] if first else None,
            "first_keeper_loss_stage": first["stage"] if first else None,
            "success": row["success"], "safe_success": bool(row["success"] and not row["protection_violation"]),
            "protection_violation": row["protection_violation"],
            "max_unsupported_seconds": row["protection"]["max_unsupported_seconds"],
            "deep_interarm_contact_steps": row["protection"]["deep_interarm_contact_steps"],
            "recovery_seconds": row.get("recovery_seconds"), "recovery_path_m": row.get("recovery_path_m"),
            "failure_reason": row.get("failure_reason"),
            "search_approaches": sum(f["stage"] == "local_arc_approach" and
                (i == 0 or frames[i-1]["stage"] != "local_arc_approach") for i, f in enumerate(frames)),
            "relevel_steps": sum(f["stage"] == "local_relevel" for f in frames),
            "local_steps": row["local_steps"], "wall_seconds": row["wall_seconds"]}


def _job(spec):
    stage, seed, scenario, candidate, scale, mode = spec
    cfg = retimed_config(scale, mode) if candidate != "R_global" else LiftConfig()
    started = time.monotonic()
    row = run_trial(seed, scenario, candidate, cfg)
    row.update(experiment_stage=stage, time_scale=scale, retiming_mode=mode,
               local_steps={key: getattr(cfg, key) for key in LOCAL_FIELDS},
               wall_seconds=time.monotonic()-started)
    return row


def protocol(mode="all_local", pilot_seeds=None, confirmation_seeds=None, parent=None):
    root = Path(__file__).resolve().parents[1]
    stages = {"pilot": list(SEEDS["pilot"] if pilot_seeds is None else pilot_seeds),
              "confirmation": list(SEEDS["confirmation"] if confirmation_seeds is None else confirmation_seeds)}
    if set(stages["pilot"]) & set(stages["confirmation"]):
        raise ValueError("confirmation seeds must be new")
    return {"question": "At an identical physical fault prefix, does retiming local recovery change keeper grasp continuity?",
            "stage_seeds": stages, "scenarios": list(SCENARIOS), "scales": list(SCALES),
            "retiming_mode": mode,
            "parent_experiment": str(parent) if parent else None,
            "followup_reason": "exploratory phase isolation after uniform retiming changed keeper safety but supplied no safely completed alternative" if parent else None,
            "local_candidate": {"left_loss": "R_left", "right_loss": "R_right"},
            "global_control": "one unchanged R_global per fault state",
            "step_counts": {str(s): {key: getattr(retimed_config(s, mode), key) for key in LOCAL_FIELDS} for s in SCALES},
            "baseline_config": asdict(LiftConfig()),
            "unchanged": ["initialization and fault actions", "motion geometry and search angles",
                          "actuator gains and limits", "8-step local relevel settling", "1-second final hold"],
            "interpretation": "retiming intervention; all_local changes motion speed and closing dwell, relevel_only keeps approach and closing unchanged; feedback can change realized paths",
            "primary_endpoint": "any keeper grasp interruption during local recovery, conditional on keeper present at start",
            "secondary_endpoints": ["safe task completion", "first keeper loss time", "recovery duration and path", "search branch changes"],
            "pilot_trigger": "at least one identical-prefix state with differing keeper continuity and a safely completed local variant",
            "lora_gate": "valid experiment; such states span at least two initialization seeds and include a fresh confirmation seed",
            "no_signal": "do not start LoRA for the timing hypothesis; do not search additional scales after seeing this result",
            "scope": "small training diagnostic, not a held-out policy evaluation or a population safety guarantee",
            "max_planned_episodes": sum(len(v) for v in stages.values()) * len(SCENARIOS) * (len(SCALES)+1),
            "source_sha256": {str(root / "auditembodied" / f): file_hash(root / "auditembodied" / f) for f in SOURCE_FILES},
            "experiment_source_sha256": file_hash(__file__), "qwen_lora_started": False}


def analyze(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["experiment_stage"], row["seed"], row["scenario"]), []).append(row)
    cases, mismatches, flips = [], [], []
    valid_faults = True
    for key, group in sorted(groups.items()):
        reference = next((r for r in group if r["candidate"] != "R_global" and r["time_scale"] == 1.), group[0])
        if len(group) != len(SCALES)+1 or len({(r["candidate"], r["time_scale"]) for r in group}) != len(group):
            mismatches.append([*key, "candidate_set"])
        for name in STATE_KEYS:
            if name not in reference or any(r.get(name) != reference[name] for r in group):
                mismatches.append([*key, name])
        if any(prefix_frames(r) != prefix_frames(reference) for r in group):
            mismatches.append([*key, "entire_observed_prefix"])
        ms = [metrics(r) for r in group]
        cases.extend(ms)
        locals_ = sorted((m for m in ms if m["candidate"] != "R_global"), key=lambda m: m["time_scale"])
        valid_faults &= all(m["fault_valid"] for m in ms)
        if locals_ and all(m["keeper_valid_at_start"] for m in locals_) and len({m["keeper_continuous"] for m in locals_}) > 1:
            flips.append({"experiment_stage": key[0], "seed": key[1], "scenario": key[2],
                          "safe_complete_variant": any(m["safe_success"] for m in locals_), "variants": locals_})
    invalid_execution = [m for m in cases if m["failure_reason"] in
                         ("numerical_instability", "forbidden_state_edit", "initialization_or_execution_error", "nominal_lift_or_hold_failed")]
    valid = bool(rows and not mismatches and valid_faults and not invalid_execution and
                 all(not r.get("blocked_calls") and r.get("contact_anchor_count") == 0 and
                     r.get("state_edit_count_after_initialization") == 0 and r.get("external_wrench_at_end") == 0 for r in rows))
    usable = [f for f in flips if f["safe_complete_variant"]]
    distinct = sorted({f["seed"] for f in usable})
    confirmed = any(f["experiment_stage"] == "confirmation" for f in usable)
    if not valid:
        decision = "INVALID"
    elif len(distinct) >= 2 and confirmed:
        decision = "GO_LIMITED_LORA_PILOT"
    elif not any(r["experiment_stage"] == "confirmation" for r in rows) and usable:
        decision = "GO_CONFIRMATION"
    else:
        decision = "NO_GO_LORA_TIMING"
    by_scale = {}
    for scale in SCALES:
        subset = [m for m in cases if m["candidate"] != "R_global" and m["time_scale"] == scale]
        by_scale[str(scale)] = {"episodes": len(subset), "valid_keeper_starts": sum(m["keeper_valid_at_start"] for m in subset),
                                "keeper_continuous": sum(m["keeper_continuous"] is True for m in subset),
                                "safe_successes": sum(m["safe_success"] for m in subset),
                                "task_successes": sum(m["success"] for m in subset),
                                "keeper_loss_episodes": sum(m["keeper_continuous"] is False for m in subset),
                                "mean_recovery_seconds": float(np.mean([m["recovery_seconds"] for m in subset])) if subset else None}
    return {"episodes": len(rows), "physical_states": len(groups), "initialization_seeds": sorted({r["seed"] for r in rows}),
            "validity": {"valid": valid, "prefix_mismatches": mismatches, "all_faults_valid": valid_faults,
                         "invalid_executions": invalid_execution},
            "by_scale": by_scale, "keeper_safety_flips": flips,
            "usable_flip_seeds": distinct, "fresh_confirmation_flip": confirmed,
            "global_safe_successes": sum(m["safe_success"] for m in cases if m["candidate"] == "R_global"),
            "global_episodes": sum(m["candidate"] == "R_global" for m in cases),
            "decision": decision, "qwen_lora_started": False, "cases": cases}


def historical_baseline(rows):
    source = Path("outputs/l4-lift-20261107/matrix/trials.jsonl")
    if not source.exists():
        return {"available": False}
    previous = {}
    with source.open() as stream:
        for line in stream:
            r = json.loads(line)
            previous[r["seed"], r["scenario"], r["candidate"]] = r
    comparisons = []
    for r in rows:
        if r["experiment_stage"] != "pilot" or r["time_scale"] != 1.:
            continue
        old = previous[r["seed"], r["scenario"], r["candidate"]]
        comparisons.append({"seed": r["seed"], "scenario": r["scenario"], "candidate": r["candidate"],
                            "prefix_identical": all(r[k] == old[k] for k in STATE_KEYS),
                            "trace_identical": r["trace"] == old["trace"],
                            "outcome_identical": all(r.get(k) == old.get(k) for k in
                                ("success", "protection_violation", "protection", "recovery_seconds", "recovery_path_m", "failure_reason"))})
    return {"available": True, "comparisons": len(comparisons),
            "all_prefixes_identical": all(r["prefix_identical"] for r in comparisons),
            "all_traces_identical": all(r["trace_identical"] for r in comparisons),
            "all_outcomes_identical": all(r["outcome_identical"] for r in comparisons), "cases": comparisons}


def export(output, rows):
    report = analyze(rows)
    report["historical_baseline"] = historical_baseline(rows)
    old = json.loads((output / "protocol.json").read_text())
    report["source_unchanged"] = all(file_hash(name) == digest for name, digest in old["source_sha256"].items())
    if not report["source_unchanged"]:
        report["decision"] = "INVALID"
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    with (output / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["cases"][0]))
        writer.writeheader()
        writer.writerows(report["cases"])
    return report


def run(output, stage, workers=4, mode="all_local", pilot_seeds=None, confirmation_seeds=None, parent=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if stage == "pilot":
        if (output / "protocol.json").exists():
            raise ValueError("experiment exists; do not overwrite")
        (output / "protocol.json").write_text(json.dumps(protocol(mode, pilot_seeds, confirmation_seeds, parent), ensure_ascii=False, indent=2)+"\n")
    else:
        previous = json.loads((output / "report.json").read_text())
        if previous["decision"] != "GO_CONFIRMATION":
            raise ValueError("confirmation requires a valid pilot keeper-safety signal")
    saved = json.loads((output / "protocol.json").read_text())
    mode = saved.get("retiming_mode", "all_local")
    destination = output / (stage+"-trials.jsonl")
    if destination.exists():
        raise ValueError("physical records exist; do not overwrite")
    jobs = []
    for seed in saved["stage_seeds"][stage]:
        for scenario in SCENARIOS:
            candidate = "R_left" if scenario == "left_loss" else "R_right"
            jobs.extend((stage, seed, scenario, candidate, scale, mode) for scale in SCALES)
            jobs.append((stage, seed, scenario, "R_global", 1., mode))
    started = time.monotonic()
    with destination.open("w") as stream:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for row in pool.map(_job, jobs):
                stream.write(json.dumps(row)+"\n")
                stream.flush()
                m = metrics(row)
                print(json.dumps({k: m[k] for k in ("stage", "seed", "scenario", "candidate", "time_scale",
                    "keeper_continuous", "safe_success", "recovery_seconds", "wall_seconds")}), flush=True)
    rows = []
    for name in ("pilot", "confirmation"):
        p = output / (name+"-trials.jsonl")
        if p.exists():
            with p.open() as stream:
                rows.extend(json.loads(line) for line in stream)
    report = export(output, rows)
    (output / (stage+"-timing.json")).write_text(json.dumps({"wall_seconds": time.monotonic()-started,
        "episodes": len(jobs), "workers": workers}, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("episodes", "decision", "usable_flip_seeds", "by_scale")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=tuple(SEEDS), default="pilot")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retiming", choices=("all_local", "relevel_only"), default="all_local")
    parser.add_argument("--pilot-seeds", type=int, nargs="+")
    parser.add_argument("--confirmation-seeds", type=int, nargs="+")
    parser.add_argument("--parent-experiment")
    args = parser.parse_args()
    run(args.output_dir, args.stage, args.workers, args.retiming,
        args.pilot_seeds, args.confirmation_seeds, args.parent_experiment)


if __name__ == "__main__":
    main()
