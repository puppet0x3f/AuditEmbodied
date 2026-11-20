"""Summarize the three final 512-step diagnostics on unchanged data."""
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parents[1]))
from auditembodied.vlm_duration_lora import prediction_metrics

folders = {
    "baseline": root.parent / "vlm-duration-lora-512-20261108",
    "targeted": root.parent / "vlm-duration-lora-targeted512-20261108",
    "balanced": root,
}
load = lambda p: json.loads(p.read_text())
protocol = load(root / "protocol.json")
result = load(root / "result.json")
training = [json.loads(s) for s in (root / "training.jsonl").read_text().splitlines()]
assert result["completed_steps"] == 512 and result["data_unchanged"]
assert [r["step"] for r in training] == list(range(1, 513))
summary, table, timing_rows = {}, [], []
for name, folder in folders.items():
    old = load(folder / "protocol.json")
    assert all(old[k] == protocol[k] for k in ("data_sha256", "steps", "rank", "alpha", "learning_rate", "model_path", "seed"))
    summary[name] = {}
    for split, filename in (("train", "train-after-predictions.jsonl"), ("development", "after-predictions.jsonl")):
        records = [json.loads(s) for s in (folder / filename).read_text().splitlines()]
        m = prediction_metrics(records)
        m["safe_coverage_by_kind"] = {}
        for kind in ("local", "global", "continue"):
            xs = [r for r in records if r["label"] == "A" and r["plan"]["kind"] == kind]
            m["safe_coverage_by_kind"][kind] = {"examples": len(xs), "accepted": sum(r["prediction"] == "A" for r in xs)}
        local = [r for r in records if r["plan"]["kind"] == "local"]
        roles = defaultdict(list)
        for r in local:
            roles[r["seed"], r["scenario"], r["plan"]["repair_arm"]].append(r)
        m["max_within_role_start_E_score_range"] = max(max(r["class_probabilities"][4] for r in xs)-min(r["class_probabilities"][4] for r in xs) for xs in roles.values())
        m["valid_local_predicted_E"] = sum(r["label"] != "E" and r["prediction"] == "E" for r in local)
        safe40 = [r for r in local if r["label"] == "A" and r["plan"]["local_relevel_steps"] == 40]
        m["safe_40_coverage"] = {"examples": len(safe40), "accepted": sum(r["prediction"] == "A" for r in safe40)}
        controls = [r for r in m["timing_40_60"]["cases"] if not r["changed_safety"]]
        m["unchanged_timing_controls"] = {"pairs": len(controls), "both_correct": sum(r["both_safety_decisions_correct"] for r in controls)}
        if name == "balanced":
            report_name = "train-after.json" if split == "train" else "after.json"
            m["start_feasibility"] = load(folder / report_name)["start_feasibility"]
            m["conditional_confusion_for_valid_starts"] = {c: dict(Counter(r["conditional_prediction"] for r in records if r["label"] == c)) for c in "ABCD"}
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
                      "false_safe": m["false_safe_accepts"], "safe_accepted": sum(r["label"] == "A" and r["prediction"] == "A" for r in records),
                      "safe_examples": sum(r["label"] == "A" for r in records),
                      "safe_AUROC": m["safe_probability_auroc"],
                      "process_unsafe_accepted": m["process_violations"]["accepted"],
                      "process_unsafe_examples": m["process_violations"]["examples"],
                      "local_pair_correct": m["within_state_local_ranking"]["correct"],
                      "local_pairs": m["within_state_local_ranking"]["pairs"],
                      "safe40_accepted": m["safe_40_coverage"]["accepted"],
                      "safe40_examples": m["safe_40_coverage"]["examples"],
                      "AA_controls_both_correct": m["unchanged_timing_controls"]["both_correct"],
                      "AA_controls": m["unchanged_timing_controls"]["pairs"]})
        timing_rows.extend({"method": name, "split": split, **r} for r in m["timing_40_60"]["cases"])

report = {"completed_steps": 512, "data_unchanged": True, "new_physical_episodes": 0,
          "training_seconds": result["training_seconds"], "peak_gpu_gib": result["peak_allocated_gpu_bytes"]/2**30,
          "all_losses_finite": all(math.isfinite(r["loss"]) for r in training),
          "actual_update_counts": dict(Counter(r["sample_kind"] for r in training)),
          "answer_presentations": dict(Counter(c for r in training for c in r["labels"])),
          "sample_presentations": sum(len(r["labels"]) for r in training),
          "comparison": summary,
          "note": "Single-run recipe comparison: prompt factorization and sampling changed together. Frontiers are post-hoc development diagnostics; no threshold deployed and no independent physical-loop test."}
(root / "comparison.json").write_text(json.dumps(report, indent=2)+"\n")
for name, rows in (("comparison.csv", table), ("timing-40-60-comparison.csv", timing_rows)):
    with (root / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
print(json.dumps({k: v for k, v in report.items() if k != "comparison"}))
for r in table:
    print(json.dumps(r))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
fig, axes = plt.subplots(1, 3, figsize=(14, 4.1), constrained_layout=True)
colors = {"baseline": "#8c949e", "targeted": "#0072b2", "balanced": "#009e73"}
for j, name in enumerate(folders):
    m = summary[name]["development"]
    xs = np.arange(2)+(j-1)*.24
    bars = axes[0].bar(xs, [m["safe_recall"]*40, m["false_safe_accepts"]], .23, color=colors[name], label=name)
    axes[0].bar_label(bars, labels=[str(round(m["safe_recall"]*40)), str(m["false_safe_accepts"])], padding=3, fontsize=8)
    frontier = m["posthoc_false_acceptance_frontier"]
    axes[2].plot(range(4), [r["true_safe_accepted"] for r in frontier], "o-", color=colors[name], label=name)
for pair in summary["balanced"]["development"]["timing_40_60"]["cases"]:
    flip = pair["changed_safety"]
    axes[1].plot([40, 60], [pair["p_safe_40"], pair["p_safe_60"]], "o-", color="#d55e00" if flip else "#0072b2", alpha=1 if flip else .45,
                 linewidth=2 if flip else 1, label="True B/A pair" if flip else None)
    if flip:
        for x, y, c in [(40, pair["p_safe_40"], pair["prediction_40"]), (60, pair["p_safe_60"], pair["prediction_60"])]:
            axes[1].annotate(c, (x, y), xytext=(5, 5), textcoords="offset points")
axes[0].set(xticks=range(2), xticklabels=["Safe accepted / 40", "Unsafe accepted / 16"], ylim=(0, 43), title="Development coverage and false accepts", ylabel="Candidates")
axes[1].set(xticks=[40, 60], ylim=(-.05, 1.1), xlabel="Relevel steps", ylabel="Uncalibrated joint A score", title="Balanced model: eight matched state pairs")
axes[2].set(xticks=range(4), ylim=(0, 43), xlabel="Allowed false accepts on development", ylabel="True safe candidates accepted / 40", title="Post-hoc score frontier")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8)
fig.savefig(root / "balanced-comparison.png", dpi=200)
fig.savefig(root / "balanced-comparison.pdf")
