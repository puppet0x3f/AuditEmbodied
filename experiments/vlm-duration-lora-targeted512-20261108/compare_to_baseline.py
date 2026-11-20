"""Compare the final fixed-budget models using the unchanged observed data."""
from pathlib import Path
from collections import Counter
import csv
import json
import math
import statistics
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parents[1]))
from auditembodied.vlm_duration_lora import prediction_metrics

baseline = root.parent / "vlm-duration-lora-512-20261108"
load = lambda p: json.loads(p.read_text())
configs = {"baseline": load(baseline / "protocol.json"), "targeted": load(root / "protocol.json")}
fixed = ("steps", "rank", "alpha", "learning_rate", "seed", "data_sha256", "instruction", "model_path")
if any(configs["baseline"][k] != configs["targeted"][k] for k in fixed):
    raise ValueError("a declared fixed comparison setting differs")
training = [json.loads(x) for x in (root / "training.jsonl").read_text().splitlines()]
if [r["step"] for r in training] != list(range(1, 513)):
    raise ValueError("the final fixed 512-step checkpoint is required")
result = load(root / "result.json")
if not result["data_unchanged"]:
    raise ValueError("input data changed during training")

summary, table, contrasts = {}, [], []
for variant, folder in (("baseline", baseline), ("targeted", root)):
    summary[variant] = {}
    for split, filename in (("train", "train-after-predictions.jsonl"), ("development", "after-predictions.jsonl")):
        records = [json.loads(x) for x in (folder / filename).read_text().splitlines()]
        m = prediction_metrics(records)
        by_kind = {}
        for kind in ("local", "global", "continue"):
            positive = [r for r in records if r["plan"]["kind"] == kind and r["label"] == "A"]
            by_kind[kind] = {"safe_examples": len(positive), "accepted": sum(r["prediction"] == "A" for r in positive)}
        m["safe_coverage_by_kind"] = by_kind
        unchanged = [r for r in m["timing_40_60"]["cases"] if not r["changed_safety"]]
        m["unchanged_timing_controls"] = {"pairs": len(unchanged),
                                          "both_safety_correct": sum(r["both_safety_decisions_correct"] for r in unchanged)}
        frontier = []
        for budget in range(sum(r["label"] != "A" for r in records)+1):
            best = {"false_safe_budget": budget, "safe_accepted": 0, "false_safe_accepted": 0,
                    "local_safe_accepted": 0, "threshold": None}
            for threshold in sorted({r["class_probabilities"][0] for r in records}):
                chosen = [r for r in records if r["class_probabilities"][0] >= threshold]
                fp = sum(r["label"] != "A" for r in chosen)
                tp = sum(r["label"] == "A" for r in chosen)
                if fp <= budget and tp > best["safe_accepted"]:
                    best.update(safe_accepted=tp, false_safe_accepted=fp, threshold=threshold,
                                local_safe_accepted=sum(r["label"] == "A" and r["plan"]["kind"] == "local" for r in chosen))
            frontier.append(best)
        m["empirical_false_acceptance_frontier"] = frontier
        m["frontier_note"] = "Post-hoc training/development comparison, not an operating-point selection or a safety guarantee."
        summary[variant][split] = m
        table.append({"variant": variant, "split": split, "examples": len(records),
                      "accuracy": m["accuracy"], "safe_accepts": m["safe_accepts"],
                      "false_safe_accepts": m["false_safe_accepts"], "safe_recall": m["safe_recall"],
                      "safe_precision": m["safe_precision"], "safe_AUROC": m["safe_probability_auroc"],
                      "E_false_safe": m["false_safe_by_class"]["E"]["accepted"],
                      "E_correct_class": m["confusion"]["E"].get("E", 0),
                      "C_false_safe": m["false_safe_by_class"]["C"]["accepted"],
                      "C_correct_class": m["confusion"]["C"].get("C", 0),
                      "process_false_safe": m["process_violations"]["accepted"],
                      "process_violations": m["process_violations"]["examples"],
                      "local_pair_correct": m["within_state_local_ranking"]["correct"],
                      "local_pairs": m["within_state_local_ranking"]["pairs"],
                      "local_A_B_correct": m["within_state_local_A_vs_B"]["correct"],
                      "local_A_B_pairs": m["within_state_local_A_vs_B"]["pairs"],
                      "local_safe_accepted": by_kind["local"]["accepted"],
                      "local_safe_examples": by_kind["local"]["safe_examples"],
                      "changed_timing_both_correct": m["timing_40_60"]["changed_pair_both_safety_correct"],
                      "unchanged_timing_both_correct": m["unchanged_timing_controls"]["both_safety_correct"]})
        contrasts.extend({"variant": variant, "split": split, **r} for r in m["timing_40_60"]["cases"])
run_summary = {"fixed_settings_match": True, "completed_steps": 512, "data_unchanged": True,
               "new_physical_episodes": 0, "training_seconds": result["training_seconds"],
               "peak_gpu_gib": result["peak_allocated_gpu_bytes"]/2**30,
               "all_losses_finite": all(math.isfinite(r["loss"]) for r in training),
               "actual_update_counts": dict(Counter(r["sample_kind"] for r in training)),
               "actual_class_presentations": dict(Counter(c for r in training for c in r["labels"])),
               "sample_presentations": sum(len(r["labels"]) for r in training),
               "mean_loss_first64": statistics.mean(r["loss"] for r in training[:64]),
               "mean_loss_last64": statistics.mean(r["loss"] for r in training[-64:]),
               "comparison": summary,
               "note": "Training recipe comparison, with sampling and auxiliary losses changed together; no per-component causal attribution or held-out physical policy result."}
(root / "comparison.json").write_text(json.dumps(run_summary, indent=2)+"\n")
for name, rows in (("comparison.csv", table), ("timing-40-60-comparison.csv", contrasts)):
    with (root / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
for variant, offset, color in (("baseline", -.18, "#8c949e"), ("targeted", .18, "#0072b2")):
    train, dev = summary[variant]["train"], summary[variant]["development"]
    nums = [train["false_safe_by_class"][c]["accepted"] for c in ("E", "C")]+[train["process_violations"]["accepted"]]
    bars = axes[0].bar(np.arange(3)+offset, nums, .34, label=variant, color=color)
    axes[0].bar_label(bars, padding=3, fontsize=8)
    values = [dev["process_violations"]["accepted"]/3,
              dev["false_safe_by_class"]["C"]["accepted"]/13, dev["safe_recall"]]
    bars = axes[1].bar(np.arange(3)+offset, values, .34, label=variant, color=color)
    labels = [f"{dev['process_violations']['accepted']}/3", f"{dev['false_safe_by_class']['C']['accepted']}/13",
              f"{round(dev['safe_recall']*40)}/40"]
    axes[1].bar_label(bars, labels=labels, padding=3, fontsize=8)
    pair = next(r for r in dev["timing_40_60"]["cases"] if r["changed_safety"])
    axes[2].plot([40, 60], [pair["p_safe_40"], pair["p_safe_60"]], "o-", label=variant, color=color)
    for x, y, predicted in ((40, pair["p_safe_40"], pair["prediction_40"]),
                             (60, pair["p_safe_60"], pair["prediction_60"])):
        axes[2].annotate(predicted, (x, y), xytext=(3, 4), textcoords="offset points", fontsize=8)
axes[0].set(xticks=range(3), xticklabels=["Missing keeper", "Incomplete", "Process unsafe"],
            ylabel="False-safe training predictions", title="Training errors (lower is better)")
axes[1].set(xticks=range(3), xticklabels=["Process unsafe", "Incomplete", "Safe accepted"],
            ylim=(0, 1.18), ylabel="Fraction of each development group",
            title="Development safety AND coverage")
axes[2].set(xticks=[40, 60], ylim=(-.05, 1.12), xlabel="Relevel steps", ylabel="Uncalibrated class A score",
            title="Development seed 23 left: true B → A")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8)
fig.savefig(root / "targeted-comparison.png", dpi=200)
fig.savefig(root / "targeted-comparison.pdf")
print(json.dumps({"run": {k: v for k, v in run_summary.items() if k != "comparison"}, "metrics": table}))
