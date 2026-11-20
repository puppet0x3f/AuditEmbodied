"""Offline diagnosis of an existing VLM baseline; creates no simulator."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import time

import numpy as np

from .vlm_baseline import (CAMERAS, TARGETS, file_hash, head_metrics, load_split,
                           parse_scores, prediction_arrays, robot_description,
                           routes, routing_metrics)


ROLE_PROMPT = """Determine the CURRENT physical support of a hammer by two robot grippers.
The first image is the external front camera; the second is another external camera.
The THIRD image, robot0_eye_in_hand, is attached to ROBOT 0, the giver.
The FOURTH image, robot1_eye_in_hand, is attached to ROBOT 1, the receiver.
Robot measurements below are always ordered robot 0 then robot 1.
Giver and receiver are task roles, not statements about who currently holds the object.
Any of these states is possible: neither supports; only robot 0 supports;
only robot 1 supports; both support. Judge each gripper independently from the images
and its actual finger readings, registered grasp gap and normalized gap error.
Closed fingers alone do not establish a grasp, but both hands CAN support simultaneously.
giver_supports means robot 0 actually grasps and supports the hammer above the table.
receiver_supports means robot 1 actually grasps and supports the hammer above the table.
coordinated_support means both support together without a severe inter-arm collision.
Use only these four images and measured robot state. Positions are world metres,
joint angles radians, finger displacements metres. Output only JSON probabilities
between 0 and 1 with keys giver_supports, receiver_supports, coordinated_support.
"""
VARIANTS = ("repeat_original", "vertical_flip", "explicit_roles", "shuffle_rgb")


def threshold_curve(labels, probabilities):
    y, p = np.asarray(labels, bool), np.asarray(probabilities, float)
    if not y.any() or y.all():
        raise ValueError("both classes are required for a threshold diagnostic")
    levels = np.r_[np.nextafter(p.max(), np.inf), np.unique(p)[::-1]]
    pos, neg = p[y], p[~y]
    rows = [{"threshold": float(t), "recall": float(np.mean(pos >= t)),
             "false_positive_rate": float(np.mean(neg >= t))} for t in levels]
    wins = pos[:, None] > neg[None, :]
    ties = pos[:, None] == neg[None, :]
    return {"auc": float(np.mean(wins + .5 * ties)),
            "best_recall_at_fpr_02": max((r for r in rows if r["false_positive_rate"] <= .02), key=lambda r: r["recall"]),
            "minimum_fpr_for_recall_90": min((r for r in rows if r["recall"] >= .9), key=lambda r: r["false_positive_rate"]),
            "levels": [{"score": float(v), "positive": int(np.sum(pos == v)), "negative": int(np.sum(neg == v))}
                       for v in np.unique(p)], "curve": rows}


def head_replacements(labels, probabilities, thresholds):
    expected = routes(labels)
    result = []
    for length in range(4):
        for heads in itertools.combinations(range(3), length):
            states = np.asarray(probabilities) >= np.asarray(thresholds)
            states[:, heads] = np.asarray(labels, bool)[:, heads]
            predicted = routes(states)
            result.append({"replaced_heads": [TARGETS[h] for h in heads],
                           "agreement": float(np.mean(predicted == expected)),
                           "continue": int(np.sum(predicted == "continue")),
                           "unsafe_continue": int(np.sum((predicted == "continue") & (expected != "continue")))})
    return result


def balanced_training_indices(obs, truth, per_state=6):
    # Labels are used only to select diagnostic examples, never for model input.
    selected, seen = {}, set()
    for index in np.argsort(truth["seeds"], kind="stable"):
        uid = str(obs["ids"][index])
        if uid in seen:
            continue
        seen.add(uid)
        key = tuple(int(x) for x in truth["labels"][index])
        selected.setdefault(key, []).append(int(index))
    indices = []
    for key in sorted(selected):
        candidates = selected[key]
        order, used_seeds = [], set()
        for index in candidates:
            seed = int(truth["seeds"][index])
            if seed not in used_seeds:
                order.append(index); used_seeds.add(seed)
        order += [i for i in candidates if i not in order]
        if len(order) < per_state:
            raise ValueError("insufficient distinct training observations")
        indices.extend(order[:per_state])
    return np.asarray(indices, int)


def analyze(source, output):
    source, output = Path(source), Path(output); output.mkdir(parents=True, exist_ok=True)
    original = json.loads((source / "vlm-report.json").read_text())
    prompt = original["selected_prompt"]
    thresholds = original["models_on_same_development_observations"]["VLM"]["thresholds"]
    result = {"scope": "existing training/development predictions only; no test predictions or simulation",
              "thresholds_held_fixed": thresholds, "threshold_optima": "descriptive optimistic within-subset bounds, not selected deployment thresholds",
              "splits": {}}
    rows = []
    for split in ("train", "dev"):
        obs, truth = load_split(source, split)
        scores, valid = prediction_arrays(source, split, prompt, obs["ids"])
        conflicts = {}
        for uid, label in zip(obs["ids"], truth["labels"]):
            conflicts.setdefault(str(uid), set()).add(tuple(label))
        groups = {}
        masks = {"calibration": np.ones(len(scores), bool), "decisions": truth["decision"], "initial": truth["initial"]}
        for name, mask in masks.items():
            labels, p = truth["labels"][mask], scores[mask]
            group = {"observations": int(mask.sum()), "heads": {}}
            for i, target in enumerate(TARGETS):
                if len(np.unique(labels[:, i])) == 2:
                    diagnostic = threshold_curve(labels[:, i], p[:, i]); group["heads"][target] = diagnostic
                    for r in diagnostic["curve"]:
                        rows.append({"split": split, "subset": name, "head": target, **r})
            if name == "initial":
                group["routing"] = routing_metrics(labels, p >= thresholds)
                group["truth_head_replacement"] = head_replacements(labels, p, thresholds)
            groups[name] = group
        states, counts = np.unique(truth["labels"], axis=0, return_counts=True)
        modal, frequency = np.unique(scores, axis=0, return_counts=True)
        mode = int(frequency.argmax())
        result["splits"][split] = {"structured_rate": float(valid.mean()), "conflicting_input_labels": sum(len(v) > 1 for v in conflicts.values()),
                                    "label_states": [{"state": s.tolist(), "count": int(n)} for s, n in zip(states, counts)],
                                    "modal_output": modal[mode].tolist(), "modal_output_count": int(frequency[mode]),
                                    "unique_output_vectors": len(modal), "subsets": groups}
    (output / "existing-score-diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "threshold-curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    figures(result, output)
    return result


def figures(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True, sharey=True)
    for ax, (i, label) in zip(axes, ((1, "Receiver support"), (2, "Coordinated support"))):
        for split, color in (("train", "#3576ae"), ("dev", "#d06a32")):
            d = result["splits"][split]["subsets"]["decisions"]["heads"][TARGETS[i]]
            c = d["curve"]
            ax.plot([100*r["false_positive_rate"] for r in c], [100*r["recall"] for r in c], "o-", ms=4, color=color,
                    label=f"{split.capitalize()} (AUC {d['auc']:.3f})")
        ax.axvline(2, linestyle="--", color="#777777", lw=1)
        ax.axhline(90 if i == 1 else 80, linestyle=":", color="#777777", lw=1)
        ax.set(title=label, xlabel="False positive rate (%)", xlim=(-2, 102), ylim=(-2, 102))
        ax.grid(alpha=.15); ax.legend(fontsize=9, loc="lower right")
    axes[0].set_ylabel("Positive recall (%)")
    fig.suptitle("Existing VLM scores: threshold changes cannot separate support states", fontsize=11)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(Path(output) / ("threshold-separability." + suffix), dpi=190, bbox_inches="tight")
    plt.close(fig)


def prepare_probe(source, output):
    source, output = Path(source), Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output / "probe-protocol.json").exists():
        raise ValueError("probe protocol already exists")
    obs, truth = load_split(source, "train"); indices = balanced_training_indices(obs, truth)
    donors = []
    for j, index in enumerate(indices):
        for shift in range(1, len(indices)):
            donor = indices[(j + shift) % len(indices)]
            if not np.array_equal(truth["labels"][index], truth["labels"][donor]):
                donors.append(int(donor)); break
    baseline = json.loads((source / "protocol.json").read_text())
    names = [str(p) for p in source.iterdir() if p.is_file()]
    protocol = {"source": str(source), "model_path": baseline["model_path"],
                "split": "train", "indices": indices.tolist(), "donor_indices": donors,
                "input_ids": obs["ids"][indices].tolist(), "evaluator_labels": truth["labels"][indices].tolist(),
                "variants": list(VARIANTS), "prompts": {"original": baseline["prompts"]["grounded"], "explicit_roles": ROLE_PROMPT},
                "purpose": "small diagnostic; no model updates or deployment selection; shuffled images test sensitivity, not task accuracy",
                "new_observations": 0, "preserved_previous_artifacts": {name: file_hash(name) for name in names}}
    (output / "probe-protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


def probe(output, batch_size=8):
    # Evaluator labels are not accessed by model inference.
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    output = Path(output); protocol = json.loads((output / "probe-protocol.json").read_text())
    if (output / "probe-predictions.jsonl").exists():
        raise ValueError("probe predictions already exist")
    with np.load(Path(protocol["source"]) / "train-observations.npz") as archive:
        obs = {k: archive[k] for k in archive.files}
    processor = AutoProcessor.from_pretrained(protocol["model_path"], local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(protocol["model_path"], dtype=torch.bfloat16,
              attn_implementation="sdpa", local_files_only=True).to("cuda:0").eval()
    indices, donors = protocol["indices"], protocol["donor_indices"]
    timings = []
    with (output / "probe-predictions.jsonl").open("w") as stream:
        for variant in protocol["variants"]:
            for offset in range(0, len(indices), batch_size):
                chunk = list(range(offset, min(offset + batch_size, len(indices))))
                torch.cuda.synchronize(); start = time.perf_counter()
                texts, images = [], []
                for j in chunk:
                    index = indices[j]; rgb = obs["rgb"][donors[j] if variant == "shuffle_rgb" else index]
                    if variant == "vertical_flip": rgb = rgb[:, ::-1]
                    content = []
                    for name, frame in zip(CAMERAS, rgb):
                        content.extend([{"type": "text", "text": name}, {"type": "image"}]); images.append(Image.fromarray(frame))
                    prompt = protocol["prompts"]["explicit_roles" if variant == "explicit_roles" else "original"]
                    content.append({"type": "text", "text": prompt + "\n" + robot_description(obs["robot"][index])})
                    texts.append(processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True))
                inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
                if len(inputs["image_grid_thw"]) != 4 * len(chunk): raise ValueError("wrong image count")
                inputs = inputs.to("cuda:0")
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=96, do_sample=False)
                answers = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                torch.cuda.synchronize(); elapsed = time.perf_counter() - start
                for j, answer in zip(chunk, answers):
                    values, valid = parse_scores(answer)
                    stream.write(json.dumps({"variant": variant, "input_id": str(obs["ids"][indices[j]]),
                        "raw_response": answer, "probabilities": values.tolist(), "valid": valid}) + "\n")
                stream.flush(); timings.append({"variant": variant, "batch_size": len(chunk), "seconds": elapsed})
                print(json.dumps({"variant": variant, "done": offset + len(chunk), "seconds": elapsed}), flush=True)
    (output / "probe-runtime.json").write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "batch_timings": timings}, indent=2) + "\n")


def score_probe(output):
    from .visual_recovery import calibrate_heads
    output = Path(output); protocol = json.loads((output / "probe-protocol.json").read_text())
    source = Path(protocol["source"]); obs, truth = load_split(source, "train"); indices = np.asarray(protocol["indices"])
    records = [json.loads(line) for line in (output / "probe-predictions.jsonl").read_text().splitlines()]
    old, _ = prediction_arrays(source, "train", "grounded", obs["ids"])
    result = {"scope": "24 balanced unique training inputs only; metrics are diagnostic and in-sample, not independent validation", "variants": {}}
    thresholds = np.asarray(json.loads((source / "vlm-report.json").read_text())["models_on_same_development_observations"]["VLM"]["thresholds"])
    by_variant = {}
    for variant in protocol["variants"]:
        rows = [r for r in records if r["variant"] == variant]
        by_id = {r["input_id"]: r for r in rows}
        if len(rows) != len(indices) or set(by_id) != set(protocol["input_ids"]): raise ValueError("incomplete diagnostic")
        by_variant[variant] = np.asarray([by_id[k]["probabilities"] for k in protocol["input_ids"]])
    original = by_variant["repeat_original"]
    result["repeat_vs_old_prediction_change_fraction"] = float(np.mean(np.any(original != old[indices], axis=1)))
    for variant, p in by_variant.items():
        labels = truth["labels"][indices]
        d = {"valid_responses": sum(r["valid"] for r in records if r["variant"] == variant), "observations": len(indices),
             "changed_from_repeat_fraction": float(np.mean(np.any(p != original, axis=1))),
             "mean_absolute_score_change": np.mean(np.abs(p - original), axis=0).tolist()}
        if variant != "shuffle_rgb":
            t, _ = calibrate_heads(p, labels)
            d.update(fixed_threshold_heads=head_metrics(labels, p, thresholds),
                     training_probe_calibrated_thresholds=t.tolist(), training_probe_calibrated_heads=head_metrics(labels, p, t),
                     separability={name: threshold_curve(labels[:, i], p[:, i]) for i, name in enumerate(TARGETS)})
        result["variants"][variant] = d
    unchanged = all(file_hash(name) == value for name, value in protocol["preserved_previous_artifacts"].items())
    if not unchanged: raise ValueError("previous baseline artifacts changed")
    result["previous_baseline_artifacts_unchanged"] = unchanged
    (output / "probe-report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def prepare_candidate(source, output):
    import shutil
    source, output = Path(source), Path(output)
    if not (output / "probe-report.json").exists():
        raise ValueError("complete training probe before candidate selection")
    target = output / "explicit-roles-full"
    if target.exists(): raise ValueError("candidate already prepared")
    target.mkdir()
    for split in ("train", "dev"):
        for kind in ("observations", "evaluator"):
            name = f"{split}-{kind}.npz"; shutil.copyfile(source / name, target / name)
    protocol = json.loads((source / "protocol.json").read_text())
    protocol["prompts"] = {"explicit_roles": ROLE_PROMPT}
    protocol["scope"] = "single training-selected diagnostic candidate; offline training/development only"
    protocol["training"] = "one candidate selected from training-only role/format probe; frozen pretrained weights; no further prompt search"
    protocol["diagnostic_selection"] = {"split": "train", "probe_report_sha256": file_hash(output / "probe-report.json"),
                                       "reason": "role grounding increased receiver AUC and recall in the 24-input training probe; full validation tests whether safe recall improves"}
    preserved = json.loads((output / "probe-protocol.json").read_text())["preserved_previous_artifacts"]
    protocol["preserved_input_sha256"].update(preserved)
    protocol["runtime_sources"]["auditembodied/vlm_diagnostics.py"] = file_hash(__file__)
    (target / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("analyze", "prepare-probe", "probe", "score-probe", "prepare-candidate"), required=True)
    parser.add_argument("--source", default="outputs/vlm-baseline-20261108")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.phase == "analyze": result = analyze(args.source, args.output_dir)
    elif args.phase == "prepare-probe": result = prepare_probe(args.source, args.output_dir)
    elif args.phase == "probe": result = probe(args.output_dir)
    elif args.phase == "prepare-candidate": result = prepare_candidate(args.source, args.output_dir)
    else: result = score_probe(args.output_dir)
    print(json.dumps({"phase": args.phase, "completed": True}))
