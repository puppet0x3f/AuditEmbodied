"""Train/development recovery selection from already executed lift candidates.

No simulator or VLM is created. Unknown healthy-control branches stay unknown.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .l4_selector import (CANDIDATES, INPUT_KEYS, SCENARIOS, STATE_KEYS,
                          ProprioRule, read_rows, safe, training_observation,
                          verify_frozen)


REFERENCE = Path("outputs/l4-proprio-20261107")
TRAIN_MATRIX = Path("outputs/l4-lift-20261107/matrix/trials.jsonl")
FEATURE_NAMES = ("mean_finger_left", "mean_finger_right", "loaded_left", "loaded_right",
                 "finger_asymmetry_left", "finger_asymmetry_right", "relative_height_error_m")
DEPTHS = (1, 2, 3, 4, 5, 6, None)
LEAF_SIZES = (1, 2, 4)
OUTCOME_KEYS = ("success", "protection_violation", "recovery_seconds", "recovery_path_m",
                "protection", "hold_seconds", "table_setdown_verified", "table_regrasp_and_lift_verified")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def features(port, rule):
    """Exactly the observable summaries used by the original policy, no action."""
    if set(port) != INPUT_KEYS: raise ValueError("robot port must contain exactly the three original fields")
    fingers = np.abs(np.asarray(port["fingers"], float))
    positions, reference = np.asarray(port["positions"], float), np.asarray(port["reference_positions"], float)
    widths = fingers.mean(1)
    loaded = (widths > rule.loaded_lower) & (widths < rule.loaded_upper)
    asymmetry = np.abs(fingers[:, 0] - fingers[:, 1]) / np.maximum(fingers.sum(1), 1e-9)
    height = abs((positions[0, 2]-positions[1, 2])-(reference[0, 2]-reference[1, 2]))
    return np.r_[widths, loaded.astype(float), asymmetry, height]


def fit_tree(x, y, max_depth, min_leaf):
    """Deterministic class-balanced CART with weighted Gini splitting."""
    x, y = np.asarray(x, float), np.asarray(y, int)
    counts = np.bincount(y, minlength=len(CANDIDATES))
    present = counts > 0
    class_weights = np.zeros(len(CANDIDATES))
    class_weights[present] = len(y) / (present.sum() * counts[present])
    weights = class_weights[y]

    def distribution(indices):
        counts = np.bincount(y[indices], weights=weights[indices], minlength=len(CANDIDATES))
        total = float(counts.sum())
        return counts, total, float(1. - np.sum((counts / total) ** 2))

    def build(indices, depth):
        counts, total, impurity = distribution(indices)
        node = {"action": CANDIDATES[int(counts.argmax())], "training_samples": len(indices)}
        if impurity <= 1e-12 or len(indices) < 2*min_leaf or (max_depth is not None and depth >= max_depth):
            return node
        best = None; best_gain = 1e-12
        for column in range(x.shape[1]):
            levels = np.unique(x[indices, column])
            for threshold in (levels[:-1] + levels[1:]) / 2:
                mask = x[indices, column] <= threshold
                left, right = indices[mask], indices[~mask]
                if min(len(left), len(right)) < min_leaf: continue
                _, wl, il = distribution(left); _, wr, ir = distribution(right)
                gain = impurity - (wl*il + wr*ir) / total
                if gain > best_gain:
                    best_gain = gain; best = (column, float(threshold), left, right)
        if best is not None:
            column, threshold, left, right = best
            node.update(feature=column, threshold=threshold,
                        left=build(left, depth+1), right=build(right, depth+1))
        return node

    return build(np.arange(len(y)), 0)


def predict(tree, x):
    result = []
    for row in np.asarray(x, float):
        node = tree
        while "feature" in node:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        result.append(node["action"])
    return result


def node_count(tree):
    return 1 if "feature" not in tree else 1 + node_count(tree["left"]) + node_count(tree["right"])


def oracle(case):
    feasible = [r for r in case["outcomes"].values() if safe(r)]
    return min(feasible, key=lambda r: (r["recovery_seconds"], r["recovery_path_m"])) if feasible else None


def load_cases(split, rule):
    policies = read_rows(REFERENCE / split / "trials.jsonl")
    fixed = read_rows(TRAIN_MATRIX) if split == "train" else [
        r for r in policies if r["scenario"] in SCENARIOS and r["method"] in CANDIDATES]
    expected_seeds = set(range(10)) if split == "train" else set(range(60, 68))
    groups = {}
    for row in fixed:
        key = (row["seed"], row["scenario"])
        if row["candidate"] in groups.setdefault(key, {}): raise ValueError("duplicate candidate outcome")
        groups[key][row["candidate"]] = row
    if set(groups) != {(s, q) for s in expected_seeds for q in SCENARIOS}:
        raise ValueError("wrong or incomplete training/development split")
    observed = {(r["seed"], r["scenario"]): r for r in policies if r.get("method") == "proprio"}
    if set(observed) != {(s, q) for s in expected_seeds for q in (*SCENARIOS, "sham")}:
        raise ValueError("missing policy observation or healthy control")
    cases = []
    for key, row in sorted(observed.items()):
        port = row["selection"]["input"]
        decision = rule(port)
        if decision != row["selection"]["decision"] or row["candidate"] != decision["candidate"]:
            raise ValueError("original rule no longer reproduces saved decisions")
        vector = features(port, rule)
        f = decision["features"]
        if not np.array_equal(vector, np.r_[f["mean_finger_qpos"], f["loaded"], f["finger_asymmetry"], f["height_error_m"]]):
            raise ValueError("shared feature front end differs")
        if key[1] == "sham":
            if row["candidate"] != "continue" or not safe(row): raise ValueError("unverified healthy continue")
            outcomes = {"continue": row}
        else:
            outcomes = groups[key]
            if set(outcomes) != set(CANDIDATES): raise ValueError("candidate matrix incomplete")
            for candidate in outcomes.values():
                if any(candidate.get(k) != row.get(k) for k in STATE_KEYS): raise ValueError("physical prefix differs")
                if training_observation(candidate) != port: raise ValueError("candidate and policy observations differ")
            chosen = outcomes[row["candidate"]]
            if any(chosen.get(k) != row.get(k) for k in OUTCOME_KEYS):
                raise ValueError("original policy differs from selected fixed candidate")
        cases.append({"seed": key[0], "scenario": key[1], "healthy": key[1] == "sham", "port": port,
                      "x": vector, "outcomes": outcomes, "original": row})
    return cases


def summarize(cases, choices):
    scored = [(c, c["outcomes"][a]) for c, a in zip(cases, choices) if a in c["outcomes"]]
    rows = [r for _, r in scored]
    pairs = [(r, oracle(c)) for c, r in scored if safe(r) and oracle(c) is not None]
    oracle_safe = sum(oracle(c) is not None for c in cases)
    result = {"scenes": len(cases), "scored_scenes": len(rows), "unobserved_outcomes": len(cases)-len(rows),
              "safe_successes": sum(safe(r) for r in rows),
              "safe_success_rate": sum(safe(r) for r in rows)/len(cases) if len(rows) == len(cases) else None,
              "protection_violations": sum(r["protection_violation"] for r in rows),
              "protection_violation_rate": sum(r["protection_violation"] for r in rows)/len(cases) if len(rows) == len(cases) else None,
              "mean_recovery_seconds": float(np.mean([r["recovery_seconds"] for r in rows])) if len(rows) == len(cases) else None,
              "mean_recovery_path_m": float(np.mean([r["recovery_path_m"] for r in rows])) if len(rows) == len(cases) else None,
              "choices": dict(Counter(choices)), "oracle_safe_scenes": oracle_safe,
              "safe_success_deficit_to_oracle": oracle_safe-sum(safe(r) for r in rows) if len(rows) == len(cases) else None,
              "joint_safe_oracle_scenes": len(pairs),
              "mean_time_gap_to_oracle_on_joint_safe": float(np.mean([a["recovery_seconds"]-b["recovery_seconds"] for a,b in pairs])) if pairs else None,
              "mean_path_gap_to_oracle_on_joint_safe": float(np.mean([a["recovery_path_m"]-b["recovery_path_m"] for a,b in pairs])) if pairs else None,
              "failure_reasons": dict(Counter(r.get("failure_reason") or "protection_condition_failed" for r in rows if not safe(r)))}
    return result


def calibrate_tree(cases):
    x = np.stack([c["x"] for c in cases])
    y = np.array([CANDIDATES.index(oracle(c)["candidate"]) for c in cases])
    seeds = sorted({c["seed"] for c in cases}); reports = []
    for depth in DEPTHS:
        for leaf in LEAF_SIZES:
            choices = [None] * len(cases); sizes = []
            for seed in seeds:
                held = np.array([c["seed"] == seed for c in cases])
                tree = fit_tree(x[~held], y[~held], depth, leaf); sizes.append(node_count(tree))
                for index, choice in zip(np.flatnonzero(held), predict(tree, x[held])): choices[index] = choice
            fault = [i for i,c in enumerate(cases) if not c["healthy"]]
            healthy = [i for i,c in enumerate(cases) if c["healthy"]]
            result = summarize([cases[i] for i in fault], [choices[i] for i in fault])
            reports.append({"max_depth": depth, "min_samples_leaf": leaf, "mean_cv_nodes": float(np.mean(sizes)),
                "healthy_continue": sum(choices[i] == "continue" for i in healthy), "healthy_scenes": len(healthy),
                "fault": result})
    eligible = [r for r in reports if r["healthy_continue"] == r["healthy_scenes"]]
    if not eligible: raise ValueError("no predeclared tree configuration retains healthy continue in training-seed cross validation")
    selected = min(eligible, key=lambda r: (-r["fault"]["safe_successes"], r["fault"]["protection_violations"],
        r["fault"]["mean_recovery_seconds"], r["fault"]["mean_recovery_path_m"], r["mean_cv_nodes"],
        100 if r["max_depth"] is None else r["max_depth"], -r["min_samples_leaf"]))
    tree = fit_tree(x, y, selected["max_depth"], selected["min_samples_leaf"])
    return tree, {"selection_split": "leave one training initialization seed out; all four states of each seed held out together",
        "supervision": "minimum-time then minimum-path actually safe candidate; healthy continue is observed safe with zero cost, alternatives unobserved",
        "evaluation": "other safe candidates are accepted and evaluated by actual cost, not marked wrong for differing from the training target",
        "selected": selected, "grid": reports, "training_target_counts": dict(Counter(CANDIDATES[i] for i in y))}


def evaluate(cases, tree, rule):
    ablated = replace(rule, asymmetry_limit=1.)
    choices = {"original_rule": [rule(c["port"])["candidate"] for c in cases],
               "direct_tree": predict(tree, np.stack([c["x"] for c in cases])),
               "no_keeper_fragility": [ablated(c["port"])["candidate"] for c in cases]}
    faults = [c for c in cases if not c["healthy"]]
    for action in CANDIDATES: choices[action] = [action] * len(cases)
    choices["oracle"] = [oracle(c)["candidate"] for c in cases]
    results = {}; details = []
    for method, predictions in choices.items():
        ids = [i for i,c in enumerate(cases) if not c["healthy"]]
        results[method] = summarize(faults, [predictions[i] for i in ids])
        for c, action in zip(cases, predictions):
            if c["healthy"] and method in CANDIDATES: continue
            row = c["outcomes"].get(action)
            reference = oracle(c)
            details.append({"seed": c["seed"], "scenario": c["scenario"], "healthy": c["healthy"],
                "method": method, "candidate": action, "observed": row is not None,
                "safe_success": safe(row) if row else None, "task_success": row["success"] if row else None,
                "protection_violation": row["protection_violation"] if row else None,
                "recovery_seconds": row["recovery_seconds"] if row else None,
                "recovery_path_m": row["recovery_path_m"] if row else None,
                "failure_reason": row.get("failure_reason") if row else "unobserved_candidate",
                "oracle_candidate": reference["candidate"] if reference else None,
                "keeper_finger_asymmetry": c["x"][4:6].tolist(),
                "protection": row.get("protection") if row else None})
    healthy = {}
    for method in ("original_rule", "direct_tree", "no_keeper_fragility"):
        ids = [i for i,c in enumerate(cases) if c["healthy"]]
        healthy[method] = summarize([cases[i] for i in ids], [choices[method][i] for i in ids])
    return {"faults": results, "healthy": healthy, "cases": details,
            "by_scenario": {q: {m: summarize([c for c in cases if c["scenario"] == q],
                [a for c,a in zip(cases,p) if c["scenario"] == q]) for m,p in choices.items()} for q in SCENARIOS}}


def run(output):
    output = Path(output)
    if output.exists(): raise ValueError("offline output already exists")
    rule = ProprioRule(**json.loads((REFERENCE / "frozen.json").read_text())["rule"])
    # This checks hashes only; no test observations or results enter fitting.
    frozen = json.loads(Path("outputs/l4-proprio-test-20261108/frozen-test.json").read_text())
    verify_frozen(frozen)
    stopped = Path("outputs/vlm-token-readout-20261108/vlm-report.json")
    if not json.loads(stopped.read_text())["decision"]["current_frozen_vlm_optimization_stopped"]:
        raise ValueError("VLM stop state unexpectedly changed")
    preserved = dict(json.loads(Path("outputs/vlm-token-readout-20261108/protocol.json").read_text())["preserved_input_sha256"])
    for name in Path("outputs/vlm-token-readout-20261108").rglob("*"):
        if name.is_file() and "__pycache__" not in name.parts: preserved[str(name)] = file_hash(name)
    sources = [TRAIN_MATRIX, REFERENCE/"train/trials.jsonl", REFERENCE/"dev/trials.jsonl"]
    for name in sources: preserved[str(name)] = file_hash(name)
    if not all(file_hash(p) == h for p,h in preserved.items()): raise ValueError("prior artifact changed")
    output.mkdir(parents=True)
    protocol = {"scope": "existing training/development candidate replay only; no new closed-loop or held-out predictions",
        "input_keys": sorted(INPUT_KEYS), "features": FEATURE_NAMES, "feature_front_end": "same frozen observable summaries as original rule",
        "actions": CANDIDATES, "rule": asdict(rule), "ablation": "asymmetry_limit 0.04 -> 1.0, disabling only keeper fragility branch",
        "direct_baseline": "class-balanced Gini CART; numpy implementation; no recovery admissibility rules or fallback",
        "depth_grid": DEPTHS, "leaf_size_grid": LEAF_SIZES,
        "selection": "training-seed leave-one-out; healthy continue required, then maximize fault safe successes, minimize guards/time/path/complexity",
        "train_seeds": list(range(10)), "dev_seeds": list(range(60,68)),
        "development": "evaluation only after selected tree saved; no recalibration or model selection",
        "cost": "mean recovery control sim time and summed EEF paths over all fault scenes including failures; common final hold excluded",
        "oracle_gap": "safe coverage deficit and paired time/path gaps only where both selector and time-first candidate Oracle safely succeed",
        "healthy_unknowns": "only continue executed; other choices stay unobserved, never counted as unsafe or assigned invented cost",
        "preserved_sha256": preserved, "runtime_source_sha256": file_hash(__file__)}
    (output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    training = load_cases("train", rule)
    started = time.perf_counter(); tree, calibration = calibrate_tree(training)
    calibration["fitting_seconds"] = time.perf_counter()-started
    (output/"calibration.json").write_text(json.dumps(calibration,indent=2)+"\n")
    (output/"direct-tree.json").write_text(json.dumps({"features": FEATURE_NAMES, "tree": tree,
        "selection": calibration["selected"], "training_only": True},indent=2)+"\n")
    # The final tree is now fixed; development is only evaluated below.
    development = load_cases("dev", rule)
    results = {"train": evaluate(training, tree, rule), "dev": evaluate(development, tree, rule)}
    for split,cases in (("train",training),("dev",development)):
        old = json.loads((REFERENCE/split/"report.json").read_text())
        for metric in ("safe_successes", "mean_recovery_seconds", "mean_recovery_path_m", "protection_violation_rate"):
            if not np.isclose(results[split]["faults"]["original_rule"][metric],old["by_method"]["proprio"][metric],rtol=0,atol=1e-12):
                raise ValueError("original rule no longer matches original report")
    unchanged = all(file_hash(p) == h for p,h in preserved.items())
    verify_frozen(frozen)
    if not unchanged or protocol["runtime_source_sha256"] != file_hash(__file__): raise ValueError("frozen inputs or source changed")
    report = {"scope": protocol["scope"], "calibration": calibration, "results": results,
        "validity": {"fault_scenes": 54, "healthy_controls": 18, "executed_fixed_candidate_outcomes_reused": 216,
            "same_prefixes_and_policy_ports": True, "original_policy_matches_fixed_candidate_outcomes": True,
            "original_rule_metrics_reproduced": True, "preserved_files": len(preserved), "all_preserved": unchanged,
            "vlm_optimization_still_stopped": True},
        "new_closed_loop_episodes": 0, "heldout_policy_predictions": 0, "new_vlm_inference": 0,
        "interpretation": "finite-library one-shot offline replay using actually executed branches, not a new independent closed-loop test"}
    (output/"offline-report.json").write_text(json.dumps(report,indent=2)+"\n")
    export(output, report)
    return report


def export(output, report):
    import csv
    rows = [{"split":split,"method":method,**metrics} for split,result in report["results"].items() for method,metrics in result["faults"].items()]
    with (output/"main-results.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (output/"selected-cases.jsonl").open("w") as stream:
        for split,result in report["results"].items():
            for row in result["cases"]:stream.write(json.dumps({"split":split,**row})+"\n")
    with (output/"healthy-controls.json").open("w") as stream:
        json.dump({s:r["healthy"] for s,r in report["results"].items()},stream,indent=2);stream.write("\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods=("original_rule","direct_tree","no_keeper_fragility","oracle")
    labels=("Frozen rule","Direct tree","No keeper fragility","Candidate Oracle")
    colors=("#467b9e","#8460a7","#d17c40","#999999")
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    data=report["results"]["dev"]["faults"]
    for ax,(key,title,ylabel) in zip(axes,(("safe_success_rate","Safe coverage","Safe success (%)"),
            ("mean_recovery_seconds","Recovery control time","Seconds"),("mean_recovery_path_m","Recovery path","Summed EEF path (m)"))):
        values=[data[m][key]*(100 if key=="safe_success_rate" else 1) for m in methods]
        bars=ax.bar(np.arange(len(methods)),values,color=colors)
        ax.set(title=title,ylabel=ylabel,xticks=np.arange(len(methods)),xticklabels=labels)
        ax.tick_params(axis="x",labelrotation=25,labelsize=8);ax.spines[["top","right"]].set_visible(False)
        ax.bar_label(bars,labels=[f"{v:.1f}" if key=="safe_success_rate" else f"{v:.3f}" for v in values],padding=3,fontsize=9)
        ax.set_ylim(0,max(values)*1.18);ax.grid(axis="y",alpha=.15)
    fig.suptitle("Development: replay of 24 previously executed candidate sets",fontsize=12)
    fig.tight_layout()
    for suffix in ("png","pdf","svg"):fig.savefig(output/f"development-attribution.{suffix}",dpi=190,bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output-dir",required=True)
    args=parser.parse_args();report=run(args.output_dir)
    print(json.dumps({"completed":True,"all_preserved":report["validity"]["all_preserved"],"new_closed_loop_episodes":0}))
