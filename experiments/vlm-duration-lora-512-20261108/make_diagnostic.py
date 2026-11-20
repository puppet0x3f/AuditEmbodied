"""Summarize the final, fixed 512-step checkpoint without selecting a checkpoint."""
from pathlib import Path
import csv
import json
import math
import statistics
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parents[1]))
from auditembodied.vlm_duration_lora import prediction_metrics


def read(name):
    return json.loads((root / name).read_text())


training = [json.loads(x) for x in (root / "training.jsonl").read_text().splitlines()]
if [r["step"] for r in training] != list(range(1, 513)):
    raise ValueError("the complete fixed 512-step run is required")
result = read("result.json")
if not result["data_unchanged"]:
    raise ValueError("the input data changed during training")
current = {"train": read("train-after.json"), "development": read("after.json")}
old = read("reference-32-development.json")
summary = {
    "steps": len(training), "data_unchanged": True,
    "all_losses_finite": all(math.isfinite(r["loss"]) for r in training),
    "physical_gpu": 2, "peak_gpu_gib": result["peak_allocated_gpu_bytes"]/2**30,
    "training_seconds": result["training_seconds"],
    "optimizer_examples": len(training) + sum(r["paired"] for r in training),
    "median_step_seconds": statistics.median(r["seconds"] for r in training),
    "mean_loss_first64": statistics.mean(r["loss"] for r in training[:64]),
    "mean_loss_last64": statistics.mean(r["loss"] for r in training[-64:]),
    "reference_32_development": old,
    "final_512": current,
    "note": "Train/development fit diagnostic only; no new physical data, checkpoint selection, operating-point calibration or policy deployment. Timing pairs are not independent physical trials."
}
(root / "diagnostic-summary.json").write_text(json.dumps(summary, indent=2)+"\n")

cases = []
for split, metrics in current.items():
    cases.extend({"checkpoint": 512, "split": split, **r} for r in metrics["timing_40_60"]["cases"])
cases.extend({"checkpoint": 32, "split": "development", **r} for r in old["timing_40_60"]["cases"])
with (root / "timing-40-60.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
    writer.writeheader()
    writer.writerows(cases)

state_rows = []
for split, filename in (("train", "train-after-predictions.jsonl"), ("development", "after-predictions.jsonl")):
    groups = {}
    for line in (root / filename).read_text().splitlines():
        r = json.loads(line)
        groups.setdefault((r["seed"], r["scenario"]), []).append(r)
    for (seed, scenario), rows in sorted(groups.items()):
        m = prediction_metrics(rows)
        state_rows.append({"split": split, "seed": seed, "scenario": scenario,
                           "examples": len(rows), "accuracy": m["accuracy"],
                           "false_safe_accepts": m["false_safe_accepts"],
                           "local_pair_correct": m["within_state_local_ranking"]["correct"],
                           "local_pairs": m["within_state_local_ranking"]["pairs"],
                           "local_A_B_correct": m["within_state_local_A_vs_B"]["correct"],
                           "local_A_B_pairs": m["within_state_local_A_vs_B"]["pairs"]})
with (root / "candidate-ranking-by-state.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(state_rows[0]))
    writer.writeheader()
    writer.writerows(state_rows)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
loss = np.array([r["loss"] for r in training])
axes[0].plot(np.arange(1, 513), loss, color="#c7cdd4", linewidth=.7, label="Optimizer-step loss")
axes[0].plot(np.arange(32, 513), np.convolve(loss, np.ones(32)/32, mode="valid"),
             color="#166a91", linewidth=2, label="32-step moving mean")
axes[0].set(xlabel="Optimizer step", ylabel="Training objective", title="Fixed 512-step training budget")
axes[0].legend(fontsize=8)
changed = [r for r in cases if r["checkpoint"] == 512 and r["changed_safety"]]
for index, r in enumerate(changed):
    offset = (index - (len(changed)-1)/2)*.035
    color = "#d55e00" if r["split"] == "development" else "#0072b2"
    label = f"{r['split']} {r['seed']} {r['repair_arm']}"
    axes[1].plot(np.array([40, 60])+offset, [r["p_safe_40"], r["p_safe_60"]], "o-", color=color, label=label)
    for x, y, predicted in ((40+offset, r["p_safe_40"], r["prediction_40"]),
                             (60+offset, r["p_safe_60"], r["prediction_60"])):
        axes[1].annotate(predicted, (x, y), xytext=(3, 5), textcoords="offset points", fontsize=8)
axes[1].set(xticks=[40, 60], xlabel="Relevel steps (all other plan fields identical)",
             ylabel="Class A token score (uncalibrated)", ylim=(-.05, 1.12),
             title="Observed outcome B at 40, A at 60")
axes[1].legend(fontsize=8, loc="best")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig(root / "fit-and-timing.png", dpi=200)
fig.savefig(root / "fit-and-timing.pdf")
print(json.dumps({"steps": 512, "training_seconds": result["training_seconds"],
                  "metrics": {split: {k: m[k] for k in ("accuracy", "safe_probability_auroc", "within_state_local_ranking", "within_state_local_A_vs_B", "timing_40_60")}
                              for split, m in current.items()}}))
