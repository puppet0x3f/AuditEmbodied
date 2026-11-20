"""Compare isolated outcome optimization with the preceding diagnostics."""
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parents[1]))
from auditembodied.vlm_duration_lora import prediction_metrics, tiny_fit_gate

load = lambda p: json.loads(p.read_text())
tiny = root.parent / "vlm-duration-lora-outcome-tiny256-20261108"
tiny_indices = load(tiny / "protocol.json")["sampling_design"]["cohort_indices"]
tiny_result = load(tiny / "result.json")
result = load(root / "result.json")
assert tiny_result["tiny_fit_gate"]["passed"] and result["completed_steps"] == 512
assert result["data_unchanged"] and result["frozen_start_unchanged"]
sources = {
    "targeted": (root.parent / "vlm-duration-lora-targeted512-20261108", "after"),
    "shared": (root.parent / "vlm-duration-lora-balanced512-20261108", "after"),
    "isolated_tiny": (root, "before"),
    "isolated_full": (root, "after"),
}
summary, table, timings = {}, [], []
for name, (folder, stage) in sources.items():
    summary[name] = {}
    for split, prefix in (("train", "train-"), ("development", "")):
        records = [json.loads(s) for s in (folder / f"{prefix}{stage}-predictions.jsonl").read_text().splitlines()]
        m = prediction_metrics(records)
        local = [r for r in records if r["plan"]["kind"] == "local" and r["label"] != "E"]
        m["valid_local_accuracy"] = {"correct": sum(r["label"] == r["prediction"] for r in local), "examples": len(local)}
        m["local_incomplete"] = {c: {"examples": sum(r["label"] == c for r in local),
            "correct": sum(r["label"] == c and r["prediction"] == c for r in local),
            "false_safe": sum(r["label"] == c and r["prediction"] == "A" for r in local)} for c in "CD"}
        m["safe_coverage_by_kind"] = {}
        for kind in ("local", "global", "continue"):
            xs = [r for r in records if r["label"] == "A" and r["plan"]["kind"] == kind]
            m["safe_coverage_by_kind"][kind] = {"examples": len(xs), "accepted": sum(r["prediction"] == "A" for r in xs)}
        controls = [r for r in m["timing_40_60"]["cases"] if not r["changed_safety"]]
        m["unchanged_timing_controls"] = {"pairs": len(controls), "both_correct": sum(r["both_safety_decisions_correct"] for r in controls)}
        if split == "train":
            m["original_tiny_retention"] = tiny_fit_gate([records[i] for i in tiny_indices])
        frontier = []
        for budget in range(4):
            best = {"false_safe_budget": budget, "true_safe_accepted": 0, "false_safe_accepted": 0,
                    "local_true_safe_accepted": 0, "threshold": None}
            for threshold in sorted({r["class_probabilities"][0] for r in records}):
                xs = [r for r in records if r["class_probabilities"][0] >= threshold]
                fp, tp = sum(r["label"] != "A" for r in xs), sum(r["label"] == "A" for r in xs)
                if fp <= budget and tp > best["true_safe_accepted"]:
                    best.update(true_safe_accepted=tp, false_safe_accepted=fp, threshold=threshold,
                                local_true_safe_accepted=sum(r["label"] == "A" and r["plan"]["kind"] == "local" for r in xs))
            frontier.append(best)
        m["posthoc_false_acceptance_frontier"] = frontier
        summary[name][split] = m
        table.append({"method": name, "split": split, "accuracy": m["accuracy"],
            "false_safe": m["false_safe_accepts"], "true_safe_accepted": round(m["safe_recall"]*sum(r["label"] == "A" for r in records)),
            "safe_AUROC": m["safe_probability_auroc"], "local_correct": m["valid_local_accuracy"]["correct"],
            "local_examples": len(local), "local_C_correct": m["local_incomplete"]["C"]["correct"],
            "local_D_correct": m["local_incomplete"]["D"]["correct"],
            "process_false_safe": m["process_violations"]["accepted"],
            "local_pair_correct": m["within_state_local_ranking"]["correct"],
            "local_pairs": m["within_state_local_ranking"]["pairs"],
            "AA_controls_both_correct": m["unchanged_timing_controls"]["both_correct"]})
        timings.extend({"method": name, "split": split, **r} for r in m["timing_40_60"]["cases"])
training = [json.loads(s) for s in (root / "training.jsonl").read_text().splitlines()]
assert [r["step"] for r in training] == list(range(1, 513))
report = {"tiny_gate": tiny_result["tiny_fit_gate"], "tiny_steps": 256, "full_steps": 512,
    "additional_outcome_steps": 768, "new_physical_episodes": 0,
    "tiny_training_seconds": tiny_result["training_seconds"], "full_training_seconds": result["training_seconds"],
    "data_unchanged": result["data_unchanged"], "frozen_start_unchanged": result["frozen_start_unchanged"],
    "all_losses_finite": all(math.isfinite(r["loss"]) for r in training),
    "full_presentations": sum(len(r["labels"]) for r in training),
    "actual_class_presentations": dict(Counter(c for r in training for c in r["labels"])),
    "comparison": summary,
    "note": "The isolated recipe continues the shared checkpoint for 256 tiny + 512 full steps; learning rate is 1e-4 then 3e-5. This is not an equal-total-budget comparison or an isolated causal test of task interference. Frontiers are post-hoc; no physical policy test."}
(root / "comparison.json").write_text(json.dumps(report, indent=2)+"\n")
for name, rows in (("comparison.csv", table), ("timing-40-60-comparison.csv", timings)):
    with (root / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
for r in table:
    print(json.dumps(r))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
colors = {"targeted": "#0072b2", "shared": "#8c949e", "isolated_full": "#009e73"}
for j, name in enumerate(colors):
    m = summary[name]["development"]
    bars = axes[0].bar(np.arange(2)+(j-1)*.24, [m["safe_recall"]*40, m["false_safe_accepts"]], .23, color=colors[name], label=name)
    axes[0].bar_label(bars, labels=[str(round(m["safe_recall"]*40)), str(m["false_safe_accepts"])], padding=3, fontsize=8)
    axes[2].plot(range(4), [r["true_safe_accepted"] for r in m["posthoc_false_acceptance_frontier"]], "o-", color=colors[name], label=name)
for pair in summary["isolated_full"]["development"]["timing_40_60"]["cases"]:
    flip = pair["changed_safety"]
    axes[1].plot([40, 60], [pair["p_safe_40"], pair["p_safe_60"]], "o-", color="#d55e00" if flip else "#0072b2",
                 alpha=1 if flip else .4, label="True B/A pair" if flip else None)
    if flip:
        for x,y,label in [(40,pair["p_safe_40"],pair["prediction_40"]),(60,pair["p_safe_60"],pair["prediction_60"])]:
            axes[1].annotate(label,(x,y),xytext=(5,5),textcoords="offset points")
axes[0].set(xticks=range(2),xticklabels=["Safe accepted / 40","Unsafe accepted / 16"],ylim=(0,43),title="Development coverage and false accepts")
axes[1].set(xticks=[40,60],ylim=(-.05,1.1),xlabel="Relevel steps",ylabel="Uncalibrated joint A score",title="Isolated full: strict 40/60 pairs")
axes[2].set(xticks=range(4),ylim=(0,43),xlabel="Allowed false accepts on development",ylabel="True safe candidates accepted / 40",title="Post-hoc score frontier")
for ax in axes:
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=8)
fig.savefig(root / "isolated-comparison.png",dpi=200)
fig.savefig(root / "isolated-comparison.pdf")
