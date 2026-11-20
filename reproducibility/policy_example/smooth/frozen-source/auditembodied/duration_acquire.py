"""Collect new-state, strictly paired relevel outcomes and matched RGB."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

from . import l4_lift
from .duration_pilot import LOCAL_FIELDS, SCALES, SCENARIOS, SOURCE_FILES, STATE_KEYS, file_hash, metrics, prefix_frames, retimed_config
from .vlm_duration_data import capture_one, outcome, plan, read_rows


def acquire_state(job):
    seed, scenario, folder = job
    candidate = "R_left" if scenario == "left_loss" else "R_right"
    rows = []
    for scale in SCALES:
        cfg = retimed_config(scale, "relevel_only")
        started = time.monotonic()
        row = capture_one((seed, scenario, folder), cfg, candidate) if not rows else l4_lift.run_trial(seed, scenario, candidate, cfg)
        row.update(experiment_stage="new_training_states", time_scale=scale, retiming_mode="relevel_only",
                   local_steps={k: getattr(cfg, k) for k in LOCAL_FIELDS}, wall_seconds=time.monotonic()-started)
        rows.append(row)
    return rows


def before_relevel(row):
    frames = row["trace"]
    first = next((i for i, frame in enumerate(frames) if frame.get("stage") == "local_relevel"), len(frames))
    return frames[:first]


def analyze(rows):
    groups, mismatches = {}, []
    for r in rows:
        groups.setdefault((r["seed"], r["scenario"]), []).append(r)
    cases, flips, changed_40_60, controls = [], [], [], []
    for key, group in sorted(groups.items()):
        reference = group[0]
        if sorted(r["local_steps"]["local_relevel_steps"] for r in group) != [30, 40, 60]:
            mismatches.append([*key, "candidate_set"])
        for field in STATE_KEYS:
            if field not in reference or any(r.get(field) != reference[field] for r in group):
                mismatches.append([*key, field])
        for name, function in [("fault_prefix", prefix_frames), ("prefix_before_relevel", before_relevel)]:
            if any(function(r) != function(reference) for r in group):
                mismatches.append([*key, name])
        if "capture" not in reference or reference["capture"]["prefix_state"] != reference.get("prefix_state"):
            mismatches.append([*key, "capture_prefix"])
        labels = {r["local_steps"]["local_relevel_steps"]: outcome(r) for r in group}
        ms = [metrics(r) for r in group]
        cases.extend(ms)
        item = {"seed": key[0], "scenario": key[1], "labels_by_relevel": labels}
        if len({m["keeper_continuous"] for m in ms}) > 1:
            flips.append(item)
        if set(labels) >= {40, 60}:
            if (labels[40] == "A") != (labels[60] == "A"):
                changed_40_60.append(item)
            if labels[40] == labels[60] == "A":
                controls.append(item)
    invalid = [dict(seed=r["seed"], scenario=r["scenario"], reason=r.get("failure_reason")) for r in rows
               if not r.get("fault_class_valid") or r.get("error") or r.get("blocked_calls")
               or r.get("contact_anchor_count") != 0 or r.get("state_edit_count_after_initialization") != 0
               or r.get("external_wrench_at_end") != 0
               or r.get("failure_reason") in ("numerical_instability", "forbidden_state_edit", "initialization_or_execution_error", "nominal_lift_or_hold_failed")]
    return {"episodes": len(rows), "physical_states": len(groups), "seeds": sorted({r["seed"] for r in rows}),
            "valid": bool(rows) and not mismatches and not invalid, "prefix_mismatches": mismatches,
            "invalid_executions": invalid, "classes": dict(Counter(outcome(r) for r in rows)),
            "by_relevel": {str(n): {"episodes": sum(r["local_steps"]["local_relevel_steps"] == n for r in rows),
                 "safe_successes": sum(outcome(r) == "A" and r["local_steps"]["local_relevel_steps"] == n for r in rows)} for n in (30, 40, 60)},
            "keeper_safety_flips": flips, "changed_40_60": changed_40_60, "AA_40_60_controls": controls, "cases": cases}


def build_dataset(output, rows, base):
    old = read_rows(base / "train.jsonl")
    dev = read_rows(base / "development.jsonl")
    if {r["seed"] for r in rows} & {r["seed"] for r in old+dev}:
        raise ValueError("new training seeds overlap the existing splits")
    captures = {(r["seed"], r["scenario"]): r["capture"] for r in rows if "capture" in r}
    added = []
    for r in rows:
        c = captures[r["seed"], r["scenario"]]
        if c["prefix_state"] != r["prefix_state"]:
            raise ValueError("observation and executed outcome prefixes differ")
        added.append({"seed": r["seed"], "scenario": r["scenario"], "image": c["observation_file"],
                      "robot": c["port"], "plan": plan(r), "label": outcome(r)})
    destination = output / "dataset"
    destination.mkdir()
    (destination / "train.jsonl").write_text("".join(json.dumps(r)+"\n" for r in old+added))
    (destination / "development.jsonl").write_bytes((base / "development.jsonl").read_bytes())
    report = {"old_train_examples": len(old), "new_train_examples": len(added), "train_examples": len(old+added),
              "effective_outcome_train_examples": sum(r["label"] != "E" for r in old+added),
              "new_train_classes": dict(Counter(r["label"] for r in added)),
              "train_classes": dict(Counter(r["label"] for r in old+added)),
              "development_examples": len(dev),
              "development_bytes_identical": file_hash(destination / "development.jsonl") == file_hash(base / "development.jsonl"),
              "train_seeds": sorted({r["seed"] for r in old+added}), "development_seeds": sorted({r["seed"] for r in dev}),
              "unknown_branch_labels_used": 0}
    (destination / "data-report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-data-dir", default="outputs/vlm-duration-pilot-20261108")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8, 16)))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output, base = Path(args.output_dir), Path(args.base_data_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("collection exists; do not overwrite")
    (output / "observations").mkdir()
    source = Path(__file__).resolve().parent
    protocol = {"seeds": args.seeds, "scenarios": list(SCENARIOS), "relevel_steps": [30, 40, 60],
        "fixed_approach_close": [16, 22], "planned_candidate_episodes": len(args.seeds)*6,
        "rgb_capture": "during the first candidate, with no extra continue episode",
        "source_sha256": {str(source / name): file_hash(source / name) for name in SOURCE_FILES},
        "base_train_sha256": file_hash(base / "train.jsonl"), "base_dev_sha256": file_hash(base / "development.jsonl"),
        "training_scope": "training-data collection only; no learned-policy test"}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    jobs = [(seed, scenario, str((output / "observations").resolve())) for seed in args.seeds for scenario in SCENARIOS]
    rows, started = [], time.monotonic()
    with (output / "trials.jsonl").open("w") as f:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for group in pool.map(acquire_state, jobs):
                for r in group:
                    rows.append(r)
                    f.write(json.dumps(r)+"\n")
                f.flush()
                print(json.dumps({"seed": group[0]["seed"], "scenario": group[0]["scenario"],
                                  "labels": [outcome(r) for r in group], "completed_episodes": len(rows)}), flush=True)
    report = analyze(rows)
    report.update(wall_seconds=time.monotonic()-started,
        source_unchanged=all(file_hash(name) == digest for name, digest in protocol["source_sha256"].items()))
    report["valid"] = report["valid"] and report["source_unchanged"]
    (output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    if not report["valid"]:
        raise ValueError("physical pairing invalid; dataset not built")
    data_report = build_dataset(output, rows, base)
    print(json.dumps({"physical": {k: report[k] for k in ("episodes", "valid", "classes", "changed_40_60")}, "data": data_report}), flush=True)


if __name__ == "__main__":
    main()
