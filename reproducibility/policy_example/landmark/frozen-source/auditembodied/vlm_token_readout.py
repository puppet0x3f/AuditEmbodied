"""One fixed binary-token VLM comparison on saved training/development inputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np

from . import vlm_baseline as baseline
from .vlm_diagnostics import ROLE_PROMPT, threshold_curve


READOUT = "binary_tokens"
ROLE_CONTEXT = ROLE_PROMPT.split(" Output only JSON probabilities", 1)[0]
QUESTIONS = (
    "Does ROBOT 0, the giver, actually grasp and support the hammer above the table?",
    "Does ROBOT 1, the receiver, actually grasp and support the hammer above the table?",
    "Do both robots support the hammer together without a severe inter-arm collision?",
)
ANSWER_INSTRUCTION = "Answer exactly Yes or No, without any other text."
DEFAULT_SOURCE = "outputs/vlm-diagnostics-20261108/explicit-roles-full"


def answer_scores(log_probabilities):
    """Yes/No conditional score and their joint mass in the full vocabulary."""
    values = np.asarray(log_probabilities, dtype=np.float64)
    if values.shape[-1] != 2 or not np.isfinite(values).all():
        raise ValueError("two finite answer-token log likelihoods required")
    normalizer = np.logaddexp(values[..., 0], values[..., 1])
    return np.exp(values[..., 0] - normalizer), np.exp(normalizer)


def acceptance_metrics(labels, accepted):
    y, accepted = np.asarray(labels, bool), np.asarray(accepted, bool)
    tp, fp = int(np.sum(y & accepted)), int(np.sum(~y & accepted))
    n = int(accepted.sum())
    return {"observations": len(y), "accepted": n, "true_positive": tp, "false_positive": fp,
            "coverage": n / len(y), "risk": fp / n if n else None,
            "recall": tp / max(1, int(y.sum())),
            "false_positive_rate": fp / max(1, int((~y).sum()))}


def risk_coverage(labels, scores):
    scores = np.asarray(scores, float)
    thresholds = np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores)[::-1]]
    return [{"threshold": float(t), **acceptance_metrics(labels, scores >= t)} for t in thresholds]


def release_margin(probabilities, thresholds):
    """Sweep both release thresholds together while holding giver recognition fixed.

    At offset zero this is exactly the frozen receiver & (coordinated | ~giver)
    gate, including equality at the giver threshold.
    """
    p, t = np.asarray(probabilities), np.asarray(thresholds)
    giver = p[:, 0] >= t[0]
    return np.minimum(p[:, 1] - t[1], np.where(giver, p[:, 2] - t[2], np.inf))


def prepare(output, source):
    output, source = Path(output), Path(source)
    if output.exists(): raise ValueError("token readout output already exists")
    original = json.loads((source / "protocol.json").read_text())
    preserved = dict(original["preserved_input_sha256"])
    for folder in (Path("outputs/vlm-baseline-20261108"), Path("outputs/vlm-diagnostics-20261108")):
        for name in folder.rglob("*"):
            if name.is_file() and "__pycache__" not in name.parts:
                preserved[str(name)] = baseline.file_hash(name)
    if not all(baseline.file_hash(p) == h for p, h in preserved.items()):
        raise ValueError("earlier baseline or source data changed")
    output.mkdir(parents=True)
    for split in ("train", "dev"):
        for kind in ("observations", "evaluator"):
            name = f"{split}-{kind}.npz"; shutil.copyfile(source / name, output / name)
    protocol = {**original, "scope": "one frozen-weight binary-token readout; training/development only, no closed-loop episodes",
                "source": str(source), "prompts": {READOUT: ROLE_CONTEXT},
                "questions": dict(zip(baseline.TARGETS, QUESTIONS)), "answer_instruction": ANSWER_INSTRUCTION,
                "answer_tokens": {"Yes": 9454, "No": 2753},
                "readout": "log P(Yes|input,question) - log P(No|input,question); sigmoid for conditional two-answer support score",
                "answer_mass": "P(Yes)+P(No) over full vocabulary; diagnostic, not a physical support probability",
                "forward": {"dtype": "bfloat16", "attention": "sdpa", "logits_to_keep": 1, "use_cache": False},
                "training": "one predeclared readout; training sanity check and descriptive calibration; no prompt, polarity or weight search",
                "development": "unchanged 312 calibration observations, 80 decision observations and 40 initial routing observations",
                "latency": "one observation means three independent questions batched together; includes preprocessing, all three image encodings, forward and log likelihood readout; acquisition/loading excluded",
                "valid_readout": "all six token log likelihoods finite; valid three-head rate replaces inapplicable generated-JSON format rate at the same 99% requirement",
                "risk_coverage": "coverage=accepted/all; selective risk=false accepted/accepted (undefined if none); FPR=false accepted/all negatives, used for original 2% calibration budget",
                "stop_rule": "if unchanged admission criteria fail, stop the current frozen VLM optimization route",
                "preserved_input_sha256": preserved,
                "runtime_sources": {name: baseline.file_hash(name) for name in (
                    "auditembodied/vlm_token_readout.py", "auditembodied/vlm_baseline.py", "auditembodied/vlm_diagnostics.py")}}
    protocol.pop("generation", None)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


def infer(output, split, batch_size=4):
    # No evaluator file, labels, fault metadata, or simulator is loaded here.
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    output = Path(output); protocol = json.loads((output / "protocol.json").read_text())
    target = output / f"{split}-{READOUT}-predictions.jsonl"
    if target.exists(): raise ValueError("readout predictions already exist")
    if split == "dev" and not (output / "selected-prompt.json").exists():
        raise ValueError("complete training check before development inference")
    with np.load(output / f"{split}-observations.npz") as archive:
        obs = {k: archive[k] for k in archive.files}
    processor = AutoProcessor.from_pretrained(protocol["model_path"], local_files_only=True)
    processor.tokenizer.padding_side = "left"
    for word, token in protocol["answer_tokens"].items():
        if processor.tokenizer.encode(word, add_special_tokens=False) != [token]:
            raise ValueError("binary answers no longer match the frozen single tokens")
    started = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(protocol["model_path"], dtype=torch.bfloat16,
                attn_implementation="sdpa", local_files_only=True).to("cuda:0").eval()
    torch.cuda.synchronize(); load_seconds = time.perf_counter() - started
    unique, seen = [], set()
    for i, uid in enumerate(obs["ids"]):
        if uid not in seen: unique.append(i); seen.add(uid)
    token_ids = list(protocol["answer_tokens"].values())

    def evaluate(indices):
        torch.cuda.synchronize(); start = time.perf_counter()
        texts, images = [], []
        for index in indices:
            for question in protocol["questions"].values():
                content = []
                for camera, rgb in zip(baseline.CAMERAS, obs["rgb"][index]):
                    content.extend([{"type": "text", "text": camera}, {"type": "image"}])
                    images.append(Image.fromarray(rgb))
                content.append({"type": "text", "text": protocol["prompts"][READOUT] + "\n" +
                    baseline.robot_description(obs["robot"][index]) + "\nQuestion: " + question + "\n" + protocol["answer_instruction"]})
                texts.append(processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True))
        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
        if len(inputs["image_grid_thw"]) != 12 * len(indices): raise ValueError("wrong three-head image count")
        if not bool((inputs["attention_mask"][:, -1] == 1).all()): raise ValueError("last token is padded")
        inputs = inputs.to("cuda:0")
        torch.cuda.synchronize(); prepared = time.perf_counter()
        with torch.inference_mode():
            result = model(**inputs, logits_to_keep=1, use_cache=False, return_dict=True)
            logits = result.logits[:, -1, :].float()
            logp = torch.log_softmax(logits, dim=-1)
            answers = logp[:, token_ids].cpu().numpy().reshape(len(indices), 3, 2)
            top = logits.argmax(-1).cpu().numpy().reshape(len(indices), 3)
        probability, mass = answer_scores(answers)
        torch.cuda.synchronize(); elapsed = time.perf_counter() - start
        timing = {"batch_size": len(indices), "query_batch_size": 3 * len(indices),
                  "preprocess_seconds": prepared-start, "forward_and_readout_seconds": elapsed-(prepared-start),
                  "end_to_end_seconds": elapsed, "amortized_seconds_per_observation": elapsed/len(indices)}
        rows = [{"input_id": str(obs["ids"][i]), "prompt_id": READOUT, "valid": True,
                 "probabilities": probability[j].tolist(), "answer_log_likelihoods_yes_no": answers[j].tolist(),
                 "answer_token_mass": mass[j].tolist(), "top_token_ids": top[j].tolist()}
                for j, i in enumerate(indices)]
        return rows, timing

    timings = []
    with target.open("w") as stream:
        for offset in range(0, len(unique), batch_size):
            rows, timing = evaluate(unique[offset:offset+batch_size]); timing["warmup"] = not timings
            timings.append(timing)
            for row in rows: stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(json.dumps({"split": split, "unique_done": offset+len(rows), "unique_total": len(unique),
                              "batch_seconds": timing["end_to_end_seconds"]}), flush=True)
    single = []
    if split == "dev":
        for position in np.linspace(0, len(unique)-1, 40, dtype=int):
            _, timing = evaluate([unique[position]]); single.append(timing)
    runtime = {"python_runtime": __import__("sys").executable, "torch": torch.__version__, "transformers": transformers.__version__,
               "gpu": torch.cuda.get_device_name(0), "model_load_seconds": load_seconds, "batch_size": batch_size,
               "queries_per_observation": 3, "unique_inputs": len(unique), "raw_observation_examples": len(obs["ids"]),
               "raw_image_shape": list(obs["rgb"].shape[1:]), "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
               "batch_timings": timings, "warm_single_observation_timings": single}
    (output / f"{split}-runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    return runtime


def comparisons(output, current):
    output = Path(output); models = {
        "original_decimal": (Path("outputs/vlm-baseline-20261108"), "grounded"),
        "role_decimal": (Path(DEFAULT_SOURCE), "explicit_roles"),
        "binary_token": (output, READOUT)}
    rows, curves, selected, split_scores = [], [], {}, {}
    for split in ("train", "dev"):
        obs, truth = baseline.load_split(output, split)
        for name, (folder, prompt) in models.items():
            result = current if name == "binary_token" else json.loads((folder / "vlm-report.json").read_text())
            t = np.asarray(result["models_on_same_development_observations"]["VLM"]["thresholds"])
            p, _ = baseline.prediction_arrays(folder, split, prompt, obs["ids"])
            split_scores[(split, name)] = p
            decision = truth["decision"]; initial = truth["initial"]
            metrics = baseline.head_metrics(truth["labels"][decision], p[decision], t)
            for i, target in enumerate(baseline.TARGETS):
                y, scores = truth["labels"][decision, i], p[decision, i]
                auc = threshold_curve(y, scores)["auc"]
                rows.append({"split": split, "model": name, "head": target, "auc": auc, **metrics[target]})
                selected[(split, name, target)] = acceptance_metrics(y, scores >= t[i])
                for point in risk_coverage(y, scores): curves.append({"split": split, "model": name, "head": target, **point})
            positive = baseline.routes(truth["labels"][initial]) == "continue"
            score = release_margin(p[initial], t)
            selected[(split, name, "release_gate")] = acceptance_metrics(positive, score >= 0)
            for point in risk_coverage(positive, score): curves.append({"split": split, "model": name, "head": "release_gate", **point})
    for filename, data in (("head-comparison.csv", rows), ("risk-coverage.csv", curves)):
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    fixed = [{"split": s, "model": m, "head": h, **v} for (s, m, h), v in selected.items()]
    (output / "frozen-operating-points.json").write_text(json.dumps(fixed, indent=2) + "\n")
    risk_figure(output, curves, selected)
    return {"development": [r for r in rows if r["split"] == "dev"],
            "frozen_operating_points": fixed}


def risk_figure(output, curves, selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, (head, title) in zip(axes.flat, (("giver_supports", "Giver support (80 decisions)"),
            ("receiver_supports", "Receiver support (80 decisions)"), ("coordinated_support", "Coordinated support (80 decisions)"),
            ("release_gate", "Initial release gate (40 observations)"))):
        for model, color, label in (("original_decimal", "#888888", "Original decimal"),
                                   ("role_decimal", "#d07832", "Role/state decimal"), ("binary_token", "#5b4495", "Binary token likelihood")):
            points = [r for r in curves if r["split"] == "dev" and r["model"] == model and r["head"] == head and r["risk"] is not None]
            ax.plot([100*r["coverage"] for r in points], [100*r["risk"] for r in points], color=color, lw=1.5, label=label)
            point = selected[("dev", model, head)]
            if point["risk"] is not None:
                ax.plot(100*point["coverage"], 100*point["risk"], "*", ms=11, color=color, mec="white", mew=.6)
        ax.set(title=title, xlabel="Accepted observations (%)", ylabel="Errors among accepted (%)", xlim=(-1, 101), ylim=(-1, 101))
        ax.grid(alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(.5, .025))
    fig.suptitle("Development risk–coverage: binary-token readout versus frozen decimal baselines", fontsize=12)
    fig.text(.5, .005, "Stars: non-empty frozen gates. Zero coverage has undefined risk. Curves are diagnostic, not new deployment settings.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .075, 1, .96))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"risk-coverage.{suffix}", dpi=190, bbox_inches="tight")
    plt.close(fig)


def report(output):
    output = Path(output)
    result = baseline.report(output)
    protocol = json.loads((output / "protocol.json").read_text())
    result["readout"] = {"name": READOUT, "answer_tokens": protocol["answer_tokens"], "definition": protocol["readout"],
                         "valid_rate_definition": protocol["valid_readout"]}
    result["inference_latency"]["VLM"]["definition"] = protocol["latency"]
    answer_diagnostics = {}
    for split in ("train", "dev"):
        obs, _ = baseline.load_split(output, split)
        records = [json.loads(line) for line in (output / f"{split}-{READOUT}-predictions.jsonl").read_text().splitlines()]
        by_id = {r["input_id"]: r for r in records}
        mass = np.asarray([by_id[k]["answer_token_mass"] for k in obs["ids"]])
        top = np.asarray([by_id[k]["top_token_ids"] for k in obs["ids"]])
        answer_diagnostics[split] = {"median_yes_no_token_mass": np.median(mass, axis=0).tolist(),
            "minimum_yes_no_token_mass": mass.min(axis=0).tolist(),
            "top_token_is_yes_or_no_rate": np.mean(np.isin(top, list(protocol["answer_tokens"].values())), axis=0).tolist()}
    result["answer_token_diagnostics"] = answer_diagnostics
    result["comparisons"] = comparisons(output, result)
    unchanged = all(baseline.file_hash(p) == h for p, h in protocol["preserved_input_sha256"].items())
    sources_unchanged = all(baseline.file_hash(p) == h for p, h in protocol["runtime_sources"].items())
    if not unchanged or not sources_unchanged: raise ValueError("preserved files or inference sources changed")
    result["preservation"] = {"checked_input_files": len(protocol["preserved_input_sha256"]),
                              "all_preserved": unchanged, "recorded_sources_unchanged": sources_unchanged}
    no_go = result["decision"]["result"] == "No-Go"
    result["decision"]["current_frozen_vlm_optimization_stopped"] = no_go
    result["decision"]["stop_rule"] = protocol["stop_rule"]
    result["new_camera_observations"] = 0
    (output / "vlm-report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "train", "select", "dev", "report"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    args = parser.parse_args()
    if args.phase == "prepare": result = prepare(args.output_dir, args.source)
    elif args.phase == "select": result = baseline.select_prompt(args.output_dir)
    elif args.phase == "report": result = report(args.output_dir)
    else: result = infer(args.output_dir, args.phase)
    print(json.dumps({"phase": args.phase, "completed": True}))
