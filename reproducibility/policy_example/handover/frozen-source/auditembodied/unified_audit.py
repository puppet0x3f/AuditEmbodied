"""Offline, trace-based audit of the two frozen independent evaluations."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCES = {"handover": "outputs/l3-visual-test-20261107/test",
           "lift": "outputs/l4-proprio-test-20261108"}


def safe(row):
    return bool(row["success"] and not row["protection_violation"])


def episode_audit(row, task):
    trace = row["trace"]
    if task == "handover":
        release = [f for f in trace if f["phase"] == "release"]
        held = [f for f in trace if f["phase"] == "hold"]
        released = bool(release and not release[-1]["evaluator"]["giver_holds"] and
                        min(abs(x) for x in release[-1]["robot"]["actual_finger_qpos4"][:2]) >= .03)
        terminal = bool(released and held and row.get("hold_seconds", 0.) >= 1. - 1e-9 and
                        release[-1]["evaluator"]["receiver_supports"] and
                        all(f["evaluator"]["receiver_supports"] and not f["evaluator"]["giver_holds"] for f in held) and
                        not any(row.get("bad_state_warnings", {}).values()))
        recorded_terminal = bool(row["handover_completed"])
        monitored = [f for f in trace if f["phase"] in ("recovery", "release", "hold")]
        supported = lambda f: f["evaluator"]["giver_holds"] or f["evaluator"]["receiver_holds"] or f["evaluator"]["table_contact"]
        sequence = row.get("recovery", {}).get("sequence", [])
        local = row["candidate"] in ("R1", "R1_retarget") or "R1_retarget" in sequence
        keeper_at_start = bool(row.get("post_fault_predicate", {}).get("giver_holds", False)) if local else None
        local_frames = [f for f in trace if f["phase"] == "recovery" and f["stage"].startswith("r1_")] if local else []
        keeper_losses = sum(not f["evaluator"]["giver_holds"] for f in local_frames)
        release_losses = sum(not f["evaluator"]["receiver_supports"] for f in release)
        method = row["candidate"]
    else:
        held = [f for f in trace if f["phase"] == "final_hold"]
        terminal = bool(held and row.get("hold_seconds", 0.) >= 1. - 1e-9 and all(f["evaluator"]["ready"] for f in held))
        recorded_terminal = bool(row.get("task_complete", False))
        monitored = [f for f in trace if f["phase"] in ("recovery", "final_hold")]
        supported = lambda f: any(f["evaluator"]["holds"]) or f["evaluator"]["table_contact"]
        local = row["candidate"] in ("R_left", "R_right")
        keeper = 1 if row["candidate"] == "R_left" else 0
        keeper_at_start = bool(row.get("post_fault", {}).get("holds", [False, False])[keeper]) if local else None
        local_frames = [f for f in trace if f["phase"] == "recovery"] if local else []
        keeper_losses = sum(not f["evaluator"]["holds"][keeper] for f in local_frames)
        release_losses = 0
        method = row["method"]
    if terminal != recorded_terminal:
        raise ValueError(f"terminal reconstruction mismatch: {task}/{row['seed']}/{row['scenario']}/{method}")
    run = longest = unsupported_steps = 0
    for frame in monitored:
        missing = not supported(frame)
        run = run + 1 if missing else 0
        longest = max(longest, run)
        unsupported_steps += int(missing)
    if abs(longest / 20. - row["protection"]["max_unsupported_seconds"]) > 1e-9:
        raise ValueError("trace and recorded unsupported duration disagree")
    return {"task": task, "seed": row["seed"], "scenario": row["scenario"], "method": method,
            "condition": "healthy" if row["scenario"] == "sham" else "main_comparison",
            "terminal_goal_met": terminal, "recorded_task_success": bool(row["success"]), "safe_success": safe(row),
            "terminal_met_without_safe_completion": terminal and not safe(row),
            "protection_violation": bool(row["protection_violation"]), "failure_reason": row["failure_reason"],
            "unsupported_steps": unsupported_steps, "max_unsupported_seconds": longest / 20.,
            "receiver_missing_release_steps": release_losses,
            "local_attempted": local, "local_keeper_missing_precondition": local and not keeper_at_start,
            "local_keeper_loss_steps": keeper_losses,
            "local_keeper_interrupted_after_valid_start": bool(local and keeper_at_start and keeper_losses),
            "recovery_seconds": row.get("recovery_seconds", 0.), "recovery_path_m": row.get("recovery_path_m", 0.),
            "rgb_bundles": row.get("policy_rgb_bundles", row.get("selection", {}).get("rgb_frames", 0)),
            "numerical_instability": row["failure_reason"] == "numerical_instability",
            "contact_anchors": row["contact_anchor_count"], "state_edits_after_initialization": row["state_edit_count_after_initialization"]}


def aggregate(rows):
    result = {"episodes": len(rows)}
    for name in ("terminal_goal_met", "recorded_task_success", "safe_success", "terminal_met_without_safe_completion",
                 "protection_violation", "local_keeper_missing_precondition", "local_keeper_interrupted_after_valid_start",
                 "numerical_instability", "contact_anchors", "state_edits_after_initialization"):
        result[name] = sum(r[name] for r in rows)
    for name in ("recovery_seconds", "recovery_path_m", "rgb_bundles"):
        result["mean_" + name] = float(np.mean([r[name] for r in rows]))
    result["safe_success_rate"] = result["safe_success"] / len(rows)
    result["protection_violation_rate"] = result["protection_violation"] / len(rows)
    result["maximum_unsupported_seconds"] = max(r["max_unsupported_seconds"] for r in rows)
    result["receiver_missing_release_steps"] = sum(r["receiver_missing_release_steps"] for r in rows)
    result["failure_reasons"] = dict(Counter(r["failure_reason"] or "protection_condition_failed" for r in rows if not r["safe_success"]))
    return result


def paired_cost(rows, method, reference, rng_seed, draws=2000):
    groups = {}
    for row in rows:
        groups.setdefault((row["seed"], row["scenario"]), {})[row["method"]] = row
    pairs = [(g[method], g[reference]) for g in groups.values() if g[method]["safe_success"] and g[reference]["safe_success"]]
    seeds = sorted({s for s, _ in groups})
    indices = np.random.default_rng(rng_seed).integers(0, len(seeds), (draws, len(seeds)))
    result = {"method": method, "reference": reference, "paired_safe_scenes": len(pairs), "initialization_seeds": len(seeds)}
    for metric in ("recovery_seconds", "recovery_path_m"):
        a = float(np.mean([x[metric] for x, _ in pairs])); b = float(np.mean([y[metric] for _, y in pairs]))
        totals = np.array([[sum(x[metric] for x, y in pairs if x["seed"] == s),
                            sum(y[metric] for x, y in pairs if x["seed"] == s)] for s in seeds])
        sampled = totals[indices].sum(1)
        reductions = 1. - sampled[sampled[:, 1] > 0, 0] / sampled[sampled[:, 1] > 0, 1]
        result[metric] = {"method_mean": a, "reference_mean": b, "reduction_fraction": 1. - a / b,
                          "ci95": np.quantile(reductions, [.025, .975]).tolist()}
    return result


def figures(pairs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Handover: vision vs rule", "Handover: vision vs proprio", "Handover: vision vs global",
              "Lift: proprio vs global"]
    colors = ["#557086", "#176c9e", "#557086", "#b05b28"]
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.5), sharey=True)
    for ax, metric, title in zip(axes, ("recovery_seconds", "recovery_path_m"), ("Recovery control time", "Summed end-effector path")):
        for i, (record, color) in enumerate(zip(pairs, colors)):
            v = record[metric]; point = v["reduction_fraction"] * 100
            lo, hi = np.asarray(v["ci95"]) * 100
            ax.errorbar(point, i, xerr=[[point-lo], [hi-point]], fmt="o", color=color, capsize=4, lw=1.7)
            ax.text(101, i, f"{point:.1f}%", va="center", ha="right", fontsize=9, color=color)
        ax.axvline(0, color="#aaaaaa", lw=1)
        ax.set(xlim=(-3, 104), xlabel="Reduction vs reference (%)", title=title)
        ax.set_yticks(range(4), labels)
        ax.grid(axis="x", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.text(.5, .015, "Paired safe scenes; 95% seed-cluster bootstrap intervals. Model inference and rendering time excluded.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / ("paired-cost-intervals." + suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    input_paths = {task: root / folder for task, folder in SOURCES.items()}
    protected = [folder / name for task, folder in input_paths.items()
                 for name in ("trials.jsonl", "report.json" if task == "handover" else "test-report.json")]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    cases, summaries, originals = [], {}, {}
    for task, folder in input_paths.items():
        report_name = "report.json" if task == "handover" else "test-report.json"
        originals[task] = json.loads((folder / report_name).read_text())
        with (folder / "trials.jsonl").open() as stream:
            for line in stream:
                cases.append(episode_audit(json.loads(line), task))
        main = [r for r in cases if r["task"] == task and r["condition"] == "main_comparison"]
        methods = list(originals[task]["by_method"])
        summaries[task] = {k: aggregate([r for r in main if r["method"] == k]) for k in methods}
        for method, result in summaries[task].items():
            old = originals[task]["by_method"][method]
            assert result["episodes"] == old["episodes"] and result["safe_success"] == old["safe_successes"]
            assert abs(result["mean_recovery_seconds"] - old["mean_recovery_seconds"]) < 1e-9
            assert abs(result["mean_recovery_path_m"] - old["mean_recovery_path_m"]) < 1e-9
    comparisons = [("handover", "visual", k, 20261107) for k in ("rule", "proprio", "R3")]
    comparisons += [("lift", "proprio", "R_global", 20261108)]
    pairs = []
    for task, method, reference, rng in comparisons:
        rows = [r for r in cases if r["task"] == task and r["condition"] == "main_comparison"]
        result = {"task": task, **paired_cost(rows, method, reference, rng)}
        old = originals[task]["paired_costs"][reference] if task == "handover" else originals[task]["paired_cost_vs_global"]
        for metric in ("recovery_seconds", "recovery_path_m"):
            assert abs(result[metric]["reduction_fraction"] - old[metric]["reduction_fraction"]) < 1e-9
            old_ci = (originals[task]["seed_bootstrap"][f"paired_{metric}_reduction_vs_{reference}"]["ci95"] if task == "handover"
                      else originals[task]["seed_bootstrap"][f"paired_{metric}_reduction_vs_global"]["ci95"])
            assert np.allclose(result[metric]["ci95"], old_ci, atol=1e-12)
        pairs.append(result)
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    assert before == after
    report = {"scope": "offline audit of frozen independent results; no new simulation or policy fitting",
              "actual_physical_records_read": len(cases), "by_task": summaries, "paired_costs": pairs,
              "terminal_met_without_safe_completion": [r for r in cases if r["terminal_met_without_safe_completion"]],
              "healthy_controls": [r for r in cases if r["condition"] == "healthy"],
              "inputs_preserved_sha256": before, "all_inputs_unchanged": True,
              "definitions": {"terminal": "task-specific final release/hold obligations reconstructed from traces",
                              "safe_success": "original recorded task success and original protection conditions; definitions preserved per task",
                              "cost": "simulated recovery actions only; common release/hold, rendering and model inference excluded",
                              "local_scope": "missing keeper at local start distinguished from loss after a valid start",
                              "main_comparison_mix": "preserve original denominators: handover 100/method includes 20 zero-torque controls; lift 60/method contains three fault types, with 20 healthy policy controls separate; do not pool task success rates",
                              "evidence_reference_audit": "no evidence-reference IDs logged; no stale-citation or evidence-age metric is inferred"}}
    (output / "audit-report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "audit-cases.jsonl").open("w") as stream:
        for row in cases: stream.write(json.dumps(row) + "\n")
    table = [{"task": task, "method": method, **result} for task, methods in summaries.items() for method, result in methods.items()]
    fields = [k for k in table[0] if k != "failure_reasons"]
    with (output / "main-results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in table)
    pair_table = [{"task": r["task"], "method": r["method"], "reference": r["reference"], "paired_safe_scenes": r["paired_safe_scenes"],
                   "metric": metric, **r[metric]} for r in pairs for metric in ("recovery_seconds", "recovery_path_m")]
    with (output / "paired-costs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pair_table[0])); writer.writeheader(); writer.writerows(pair_table)
    figures(pairs, output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run(args.root, args.output_dir)
    print(json.dumps({"records": result["actual_physical_records_read"], "terminal_only_cases": len(result["terminal_met_without_safe_completion"]),
                      "all_inputs_unchanged": result["all_inputs_unchanged"]}))
