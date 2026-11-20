"""Compare the same outcome optimizer budget with and without new states."""
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parents[1]))
from auditembodied.vlm_duration_lora import prediction_metrics, timing_contrasts, tiny_fit_gate

load = lambda p: json.loads(p.read_text())
old = root.parent / "vlm-duration-lora-outcome-full512-20261108"
physical = root.parent / "l4-duration-newstates-20261108"
configs = [load(p / "protocol.json") for p in (old, root)]
fixed = ["model_path", "steps", "learning_rate", "rank", "alpha", "seed", "pair_weight", "sampling", "auxiliary_task_weight", "init_adapter", "frozen_start_adapter", "outcome_instruction", "start_instruction"]
assert all(configs[0][k] == configs[1][k] for k in fixed)
assert configs[0]["data_sha256"]["development.jsonl"] == configs[1]["data_sha256"]["development.jsonl"]
result = load(root / "result.json")
assert result["completed_steps"] == 512 and result["data_unchanged"] and result["frozen_start_unchanged"]
read = lambda p: [json.loads(s) for s in p.read_text().splitlines()]


def frontier(records):
    out = []
    for budget in range(4):
        best = {"false_safe_budget": budget, "true_safe_accepted": 0, "false_safe_accepted": 0, "local_safe_accepted": 0, "threshold": None}
        for threshold in sorted({r["class_probabilities"][0] for r in records}):
            xs = [r for r in records if r["class_probabilities"][0] >= threshold]
            fp, tp = sum(r["label"] != "A" for r in xs), sum(r["label"] == "A" for r in xs)
            if fp <= budget and tp > best["true_safe_accepted"]:
                best.update(true_safe_accepted=tp, false_safe_accepted=fp, threshold=threshold,
                            local_safe_accepted=sum(r["label"] == "A" and r["plan"]["kind"] == "local" for r in xs))
        out.append(best)
    return out


def metrics(records):
    m = prediction_metrics(records)
    local = [r for r in records if r["plan"]["kind"] == "local" and r["label"] != "E"]
    m["valid_local_accuracy"] = {"correct": sum(r["label"] == r["prediction"] for r in local), "examples": len(local)}
    m["local_incomplete"] = {c: {"examples": sum(r["label"] == c for r in local),
        "correct": sum(r["label"] == c and r["prediction"] == c for r in local),
        "false_safe": sum(r["label"] == c and r["prediction"] == "A" for r in local)} for c in "CD"}
    m["safe_coverage_by_kind"] = {kind: {"examples": sum(r["label"] == "A" and r["plan"]["kind"] == kind for r in records),
        "accepted": sum(r["label"] == r["prediction"] == "A" and r["plan"]["kind"] == kind for r in records)} for kind in ["local", "global", "continue"]}
    controls = [r for r in m["timing_40_60"]["cases"] if not r["changed_safety"]]
    m["AA_controls"] = {"pairs": len(controls), "both_correct": sum(r["both_safety_decisions_correct"] for r in controls)}
    m["posthoc_frontier"] = frontier(records)
    return m


def development_gate(records, threshold=None):
    rs = records if threshold is None else [{**r, "prediction": "A" if r["class_probabilities"][0] >= threshold else "rejected"} for r in records]
    timing = timing_contrasts(rs)
    critical = [r for r in timing["cases"] if r["changed_safety"]]
    controls = [r for r in timing["cases"] if not r["changed_safety"]]
    tp = sum(r["label"] == r["prediction"] == "A" for r in rs)
    process = sum(r["label"] in "BD" and r["prediction"] == "A" for r in rs)
    aa = sum(r["both_safety_decisions_correct"] for r in controls)
    flip = len(critical) == 1 and critical[0]["both_safety_decisions_correct"]
    return {"passed": bool(flip and process == 0 and tp >= 28 and aa >= 5), "threshold": threshold,
            "critical_flip_correct": bool(flip), "process_false_accepts": process, "safe_accepted": tp,
            "AA_controls_both_accepted": aa,
            "other_false_accepts": sum(r["label"] not in "ABD" and r["prediction"] == "A" for r in rs),
            "note": "Development-only behavioral gate, not an independent safety result."}


summary, table, timings = {}, [], []
for name, folder in [("old_data", old), ("new_data", root)]:
    train = read(folder / "train-after-predictions.jsonl")
    dev = read(folder / "after-predictions.jsonl")
    splits = {"old_train": [r for r in train if r["seed"] < 8], "development": dev}
    if name == "new_data":
        splits["new_train"] = [r for r in train if 8 <= r["seed"] < 16]
    summary[name] = {}
    for split, rs in splits.items():
        m = metrics(rs)
        if split == "old_train":
            indices = load(root.parent / "vlm-duration-lora-outcome-tiny256-20261108/protocol.json")["sampling_design"]["cohort_indices"]
            m["tiny_retention"] = tiny_fit_gate([rs[i] for i in indices])
            m["retained_C_hard_cases"] = [{"seed":rs[i]["seed"], "scenario":rs[i]["scenario"], "label":rs[i]["label"], "prediction":rs[i]["prediction"]} for i in [34,74]]
        summary[name][split] = m
        table.append({"method":name,"split":split,"examples":len(rs),"accuracy":m["accuracy"],
            "false_safe":m["false_safe_accepts"],"safe_accepted":round(m["safe_recall"]*sum(r["label"]=="A" for r in rs)),
            "safe_AUROC":m["safe_probability_auroc"],"process_false_safe":m["process_violations"]["accepted"],
            "local_pair_correct":m["within_state_local_ranking"]["correct"],"local_pairs":m["within_state_local_ranking"]["pairs"],
            "AA_both_correct":m["AA_controls"]["both_correct"],"AA_pairs":m["AA_controls"]["pairs"]})
        timings.extend({"method":name,"split":split,**r} for r in m["timing_40_60"]["cases"])
dev = read(root / "after-predictions.jsonl")
gates = [development_gate(dev,t) for t in sorted({r["class_probabilities"][0] for r in dev})]
passing = [r for r in gates if r["passed"]]
gate_report = {"default_argmax":development_gate(dev),
    "posthoc_passing_operating_points":passing,
    "proposed_point":max(passing,key=lambda r:(r["safe_accepted"],-r["other_false_accepts"])) if passing else None,
    "deployed":False,"note":"Threshold scan is post-hoc on development; no physical policy run was performed."}
(root / "development-gate.json").write_text(json.dumps(gate_report,indent=2)+"\n")
training=read(root / "training.jsonl")
report={"same_optimizer_settings":True,"same_initial_adapter":True,"development_bytes_identical":True,
    "completed_steps":512,"old_presentations":sum(len(r["labels"]) for r in read(old / "training.jsonl")),
    "new_presentations":sum(len(r["labels"]) for r in training),"training_seconds":result["training_seconds"],
    "all_losses_finite":all(math.isfinite(r["loss"]) for r in training),"frozen_start_unchanged":True,
    "new_physical_episodes":48,"collection":load(physical / "report.json"),
    "comparison":summary,"development_gate":gate_report,
    "note":"Single-run equal optimizer-step comparison; no statistical significance or independent physical-policy claim."}
(root / "comparison.json").write_text(json.dumps(report,indent=2)+"\n")
for name,rows in [("comparison.csv",table),("timing-40-60-comparison.csv",timings)]:
    with (root / name).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for r in table:print(json.dumps(r))
print('development_gate',json.dumps(gate_report))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(14,4.1),constrained_layout=True)
cases=report["collection"]["cases"]
for seed,scenario in [(8,"left_loss"),(10,"right_loss")]:
    xs=sorted([r for r in cases if r["seed"]==seed and r["scenario"]==scenario],key=lambda r:r["time_scale"])
    axes[0].plot([r["local_steps"]["local_relevel_steps"] for r in xs],[r["keeper_lost_steps"] for r in xs],"o-",label=f"seed {seed} {scenario}")
for name,color in [("old_data","#8c949e"),("new_data","#009e73")]:
    m=summary[name]["development"]
    p=next(r for r in m["timing_40_60"]["cases"] if r["changed_safety"])
    axes[1].plot([40,60],[p["p_safe_40"],p["p_safe_60"]],"o-",color=color,label=name)
    for x,y,label in [(40,p["p_safe_40"],p["prediction_40"]),(60,p["p_safe_60"],p["prediction_60"])]:
        axes[1].annotate(label,(x,y),xytext=(3,5),textcoords="offset points")
    axes[2].plot(range(4),[r["true_safe_accepted"] for r in m["posthoc_frontier"]],"o-",color=color,label=name)
axes[0].set(xticks=[30,40,60],xlabel="Relevel steps",ylabel="Keeper-grasp missing control steps",title="New physical timing interventions")
axes[1].set(xticks=[40,60],ylim=(-.05,1.1),xlabel="Relevel steps",ylabel="Uncalibrated joint A score",title="Development seed 23 left: true B/A")
axes[2].set(xticks=range(4),ylim=(0,43),xlabel="Allowed false accepts on development",ylabel="True safe candidates accepted / 40",title="Post-hoc score frontier")
for ax in axes:ax.spines[["top","right"]].set_visible(False);ax.legend(fontsize=8)
fig.savefig(root / "data-augmentation-comparison.png",dpi=200)
fig.savefig(root / "data-augmentation-comparison.pdf")
