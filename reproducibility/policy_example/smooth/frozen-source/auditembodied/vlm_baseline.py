"""Offline train/development VLM comparison; never creates a simulator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np


TARGETS = ("giver_supports", "receiver_supports", "coordinated_support")
CAMERAS = ("frontview", "agentview", "robot0_eye_in_hand", "robot1_eye_in_hand")
BASE_PROMPT = """Assess a two-Panda hammer handover from the four simultaneous camera images and measured robot state.
Robot 0 is the giver; robot 1 is the receiver. Positions are world metres, joint angles radians, finger displacements metres.
The three probabilities mean: giver_supports = giver actually grasps and supports the object above the table;
receiver_supports = receiver actually grasps and supports the object above the table;
coordinated_support = both support it together without a severe inter-arm collision.
A receiver-only stable grasp can finish the handover after the giver opens. Closed fingers or proximity alone do not prove support.
Use only the supplied images and robot measurements. Return a JSON object with exactly these probability keys:
giver_supports, receiver_supports, coordinated_support. Values must be numbers between 0 and 1. No explanation.
"""
PROMPTS = {
    "direct": BASE_PROMPT,
    "grounded": BASE_PROMPT + """Before answering, check the object relative to each gripper across all views. Distinguish contact from nearby empty space.
Compare actual fingers with the registered grasp gaps and normalized gap errors: both fully open and closed-on-empty fingers can lack a grasp.
Treat these measurements as evidence rather than as guaranteed contact. A supporting hand must carry the object, not merely touch it on the table.
Use agreement between camera geometry and robot measurements; give low probability when the grasp evidence is weak.
"""}
CRITERIA = {"minimum_structured_response_rate": .99, "minimum_giver_balanced_accuracy": .85,
            "minimum_receiver_recall": .90, "minimum_coordinated_recall": .80,
            "release_false_positive_budget": .02, "maximum_initial_unsafe_continue_rate": .02,
            "minimum_initial_route_agreement": .80}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def input_hash(rgb, robot):
    return hashlib.sha256(np.asarray(rgb, np.uint8).tobytes() + np.asarray(robot, np.float32).tobytes()).hexdigest()


def robot_description(robot):
    r = np.asarray(robot, np.float32)
    if r.shape != (46,): raise ValueError("exactly 46 robot measurements required")
    # Nine significant digits preserve float32 values on round-trip.
    vector = lambda x: "[" + ",".join(format(float(v), ".9g") for v in x) + "]"
    return "\n".join(["eef_positions_giver_receiver=" + vector(r[:6]),
                      "arm_joint_angles_giver_receiver=" + vector(r[6:20]),
                      "actual_finger_joints_giver_receiver=" + vector(r[20:24]),
                      "registered_grasp_gap_per_finger_giver_receiver=" + vector(r[24:26]),
                      "eef_rotation_matrices_row_major_giver_receiver=" + vector(r[26:44]),
                      "normalized_grasp_gap_error_giver_receiver=" + vector(r[44:46])])


def parse_scores(text):
    try:
        start = text.index("{")
        data, _ = json.JSONDecoder().raw_decode(text[start:])
        values = np.asarray([float(data[k]) for k in TARGETS], dtype=float)
        if np.all(np.isfinite(values)) and np.all((values >= 0) & (values <= 1)): return values, True
    except (ValueError, KeyError, TypeError): pass
    return np.zeros(3), False


def routes(states):
    states = np.asarray(states, bool)
    ready = states[:, 1] & (states[:, 2] | ~states[:, 0])
    return np.where(ready, "continue", np.where(states[:, 0], "R1_retarget", "R3"))


def prepare(output, reference, model_path):
    output, reference, model_path = Path(output), Path(reference), Path(model_path)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists(): raise ValueError("baseline already prepared")
    source_hashes, counts = {}, {}
    for split, allowed in (("train", set(range(10))), ("dev", set(range(60, 68)))):
        rgb, robot, labels, ids, decision, initial, seeds = [], [], [], [], [], [], []
        files = sorted((reference / split / "robot").glob("*.npz"))
        if {int(f.name.split("-", 1)[0]) for f in files} != allowed: raise ValueError("wrong train/dev seed split")
        for filename in files:
            seed = int(filename.name.split("-", 1)[0])
            target_file = reference / split / "evaluator" / filename.name
            source_hashes[str(filename)] = file_hash(filename)
            source_hashes[str(target_file)] = file_hash(target_file)
            with np.load(filename) as archive: observed = {k: archive[k] for k in archive.files}
            with np.load(target_file) as archive: truth = {k: archive[k] for k in archive.files}
            if not np.array_equal(observed["time"], truth["time"]): raise ValueError("misaligned labels")
            if observed["rgb"].shape[1:] != (4, 64, 64, 3) or observed["robot"].shape[1:] != (46,):
                raise ValueError("observation budget differs from frozen verifier")
            marked = truth["decision"].astype(bool)
            first = int(np.flatnonzero(marked)[0])
            selected = marked if split == "train" else marked | (truth["stages"] == "r1_recovery_action")
            for index in np.flatnonzero(selected):
                image, measured = observed["rgb"][index], observed["robot"][index]
                rgb.append(image); robot.append(measured); labels.append(truth["labels"][index])
                ids.append(input_hash(image, measured)); decision.append(bool(marked[index])); initial.append(bool(index == first)); seeds.append(seed)
        np.savez_compressed(output / (split + "-observations.npz"), rgb=np.asarray(rgb, np.uint8), robot=np.asarray(robot, np.float32), ids=np.asarray(ids))
        np.savez_compressed(output / (split + "-evaluator.npz"), labels=np.asarray(labels, np.int8), ids=np.asarray(ids),
                            decision=np.asarray(decision), initial=np.asarray(initial), seeds=np.asarray(seeds))
        counts[split] = {"episodes": len(files), "selected_observations": len(ids), "unique_inputs": len(set(ids)),
                         "decision_observations": sum(decision), "initial_decisions": sum(initial)}
    weights = sorted(model_path.glob("*.safetensors"))
    model_files = {str(f): {"bytes": f.stat().st_size, "sha256": file_hash(f)} for f in
                   [*weights, model_path / "config.json", model_path / "tokenizer_config.json", model_path / "preprocessor_config.json"]}
    from .l4_selector import verify_frozen
    preserved = json.loads(Path("outputs/l4-proprio-test-20261108/frozen-test.json").read_text())
    verify_frozen(preserved)
    for task, folder, report in (("handover", Path("outputs/l3-visual-test-20261107/test"), "report.json"),
                                ("lift", Path("outputs/l4-proprio-test-20261108"), "test-report.json")):
        for name in ("trials.jsonl", report): source_hashes[str(folder / name)] = file_hash(folder / name)
    protocol = {"scope": "offline train/development only; no new closed-loop episodes and no held-out VLM predictions",
                "reference": str(reference), "model_path": str(model_path), "model": "Qwen3-VL-8B-Instruct",
                "model_card": "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct", "model_files": model_files,
                "prompts": PROMPTS, "criteria": CRITERIA, "counts": counts,
                "input_contract": {"cameras": CAMERAS, "raw_rgb_shape": [4, 64, 64, 3], "robot_dimensions": 46,
                                   "no_inputs": ["seed", "fault", "stage", "time", "contact", "object truth", "evaluator labels"]},
                "generation": {"do_sample": False, "max_new_tokens": 96, "dtype": "bfloat16", "attention": "sdpa"},
                "training": "frozen pretrained weights; choose between two prompts on training decision observations with train-calibrated heads",
                "development": "selected prompt only; calibrate same 312-frame mask as frozen heads, evaluate same 80 decision observations and 40 initial routing decisions",
                "invalid_response": "count as invalid; zero support probabilities; cannot pass admission through rejecting everything",
                "recovery_policy": "same receiver-and-(coordinated-or-not-giver) rule; same frozen R1/R3; offline initial routing only",
                "latency": "record warm single-observation preprocessing+generation latency and batched throughput; no unvalidated real-time deadline is asserted",
                "preserved_input_sha256": source_hashes,
                "runtime_sources": {"auditembodied/vlm_baseline.py": file_hash(__file__)}}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


def load_split(output, split):
    with np.load(Path(output) / (split + "-observations.npz")) as archive: obs = {k: archive[k] for k in archive.files}
    with np.load(Path(output) / (split + "-evaluator.npz")) as archive: truth = {k: archive[k] for k in archive.files}
    if not np.array_equal(obs["ids"], truth["ids"]): raise ValueError("evaluation inputs do not match labels")
    return obs, truth


def prediction_arrays(output, split, prompt_id, ids):
    rows = [json.loads(line) for line in (Path(output) / f"{split}-{prompt_id}-predictions.jsonl").read_text().splitlines()]
    by_id = {r["input_id"]: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != set(ids.tolist()): raise ValueError("missing or duplicate predictions")
    return np.asarray([by_id[k]["probabilities"] for k in ids]), np.asarray([by_id[k]["valid"] for k in ids])


def head_metrics(labels, probabilities, thresholds):
    from .visual_recovery import binary_metrics
    result = {}
    for i, name in enumerate(TARGETS):
        metrics = binary_metrics(labels[:, i], probabilities[:, i] >= thresholds[i])
        metrics["balanced_accuracy"] = .5 * (metrics["recall"] + 1. - metrics["false_positive_rate"])
        result[name] = metrics
    return result


def select_prompt(output):
    from .visual_recovery import calibrate_heads
    output = Path(output); protocol = json.loads((output / "protocol.json").read_text())
    obs, truth = load_split(output, "train")
    candidates = {}
    for prompt_id in protocol["prompts"]:
        p, valid = prediction_arrays(output, "train", prompt_id, obs["ids"])
        thresholds, calibration = calibrate_heads(p, truth["labels"])
        metrics = head_metrics(truth["labels"], p, thresholds)
        candidates[prompt_id] = {"structured_response_rate": float(valid.mean()), "thresholds": thresholds.tolist(),
                                 "metrics": metrics, "calibration": calibration,
                                 "mean_balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in metrics.values()]))}
    best = max(candidates, key=lambda k: (candidates[k]["structured_response_rate"], candidates[k]["mean_balanced_accuracy"]))
    selected = {"prompt_id": best, "prompt": protocol["prompts"][best], "train_candidates": candidates,
                "selection_split": "training only; selected before any development model inference"}
    (output / "selected-prompt.json").write_text(json.dumps(selected, indent=2) + "\n")
    return selected


def infer(output, split, batch_size=8):
    # This phase imports no simulator, evaluator or training labels.
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    output = Path(output); protocol = json.loads((output / "protocol.json").read_text())
    with np.load(output / (split + "-observations.npz")) as archive: obs = {k: archive[k] for k in archive.files}
    prompts = list(protocol["prompts"]) if split == "train" else [json.loads((output / "selected-prompt.json").read_text())["prompt_id"]]
    if any((output / f"{split}-{p}-predictions.jsonl").exists() for p in prompts): raise ValueError("inference records exist")
    processor = AutoProcessor.from_pretrained(protocol["model_path"], local_files_only=True)
    processor.tokenizer.padding_side = "left"
    start = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(protocol["model_path"], dtype=torch.bfloat16,
                attn_implementation="sdpa", local_files_only=True).to("cuda:0").eval()
    torch.cuda.synchronize(); load_seconds = time.perf_counter() - start
    first = {}; unique = []
    for i, key in enumerate(obs["ids"]):
        if key not in first: first[key] = i; unique.append(i)
    timings = []; grids = []

    def evaluate_batch(indices, prompt_id):
        torch.cuda.synchronize(); started = time.perf_counter()
        texts, images = [], []
        for index in indices:
            content = []
            for name, array in zip(CAMERAS, obs["rgb"][index]):
                content.extend([{"type": "text", "text": name}, {"type": "image"}])
                images.append(Image.fromarray(array))
            content.append({"type": "text", "text": protocol["prompts"][prompt_id] + "\n" + robot_description(obs["robot"][index])})
            texts.append(processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True))
        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
        grid = inputs["image_grid_thw"].tolist()
        if len(grid) != 4 * len(indices): raise ValueError("wrong number of images reached the VLM")
        inputs = inputs.to("cuda:0")
        torch.cuda.synchronize(); prepared = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=protocol["generation"]["max_new_tokens"], do_sample=False)
        torch.cuda.synchronize(); generated_at = time.perf_counter()
        answers = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        elapsed = time.perf_counter() - started
        records = []
        for index, answer in zip(indices, answers):
            values, valid = parse_scores(answer)
            records.append({"input_id": str(obs["ids"][index]), "prompt_id": prompt_id, "raw_response": answer,
                            "probabilities": values.tolist(), "valid": valid})
        timing = {"batch_size": len(indices), "preprocess_seconds": prepared-started,
                  "generate_seconds": generated_at-prepared, "end_to_end_seconds": elapsed,
                  "amortized_seconds_per_observation": elapsed/len(indices), "generated_tokens": generated.shape[1]-inputs["input_ids"].shape[1]}
        return records, timing, grid

    for prompt_id in prompts:
        with (output / f"{split}-{prompt_id}-predictions.jsonl").open("w") as stream:
            for offset in range(0, len(unique), batch_size):
                records, timing, grid = evaluate_batch(unique[offset:offset+batch_size], prompt_id)
                timing["warmup"] = not timings; timings.append(timing)
                if not grids: grids = grid
                for record in records: stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(json.dumps({"split": split, "prompt": prompt_id, "unique_done": offset+len(records),
                                  "unique_total": len(unique), "valid_in_batch": sum(r["valid"] for r in records),
                                  "batch_seconds": timing["end_to_end_seconds"]}), flush=True)
    single = []
    if split == "dev" and batch_size == 1:
        single = [r for r in timings if not r["warmup"]]
    elif split == "dev":
        for index in [unique[i] for i in np.linspace(0, len(unique)-1, 20, dtype=int)]:
            _, timing, _ = evaluate_batch([index], prompts[0]); single.append(timing)
    runtime = {"python_runtime": __import__("sys").executable, "torch": torch.__version__, "transformers": transformers.__version__,
               "gpu": torch.cuda.get_device_name(0), "model_load_seconds": load_seconds, "batch_size": batch_size,
               "unique_inputs": len(unique), "raw_observation_examples": len(obs["ids"]),
               "raw_image_shape": list(obs["rgb"].shape[1:]), "sample_image_grid_thw": grids,
               "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(), "batch_timings": timings,
               "warm_single_observation_timings": single}
    (output / (split + "-runtime.json")).write_text(json.dumps(runtime, indent=2) + "\n")
    return runtime


def routing_metrics(labels, states):
    expected, predicted = routes(labels), routes(states)
    unsupported = ~np.asarray(labels[:, 1], bool)
    ineligible = expected != "continue"
    false_continue = (predicted == "continue") & ineligible
    return {"observations": len(labels), "agreement": float(np.mean(expected == predicted)),
            "unsafe_continue_count": int(false_continue.sum()), "receiver_unsupported_examples": int(unsupported.sum()),
            "release_ineligible_examples": int(ineligible.sum()),
            "unsafe_continue_rate": float(false_continue.sum()/max(1, ineligible.sum())),
            "expected_scope_counts": {k: int(np.sum(expected == k)) for k in ("continue", "R1_retarget", "R3")},
            "predicted_scope_counts": {k: int(np.sum(predicted == k)) for k in ("continue", "R1_retarget", "R3")}}


def report(output):
    from .visual_recovery import MultiSupportVerifier, calibrate_heads
    output = Path(output); protocol = json.loads((output / "protocol.json").read_text())
    selected = json.loads((output / "selected-prompt.json").read_text()); prompt_id = selected["prompt_id"]
    obs, truth = load_split(output, "dev"); p, valid = prediction_arrays(output, "dev", prompt_id, obs["ids"])
    thresholds, calibration = calibrate_heads(p, truth["labels"])
    decision, initial = truth["decision"], truth["initial"]
    metrics = head_metrics(truth["labels"][decision], p[decision], thresholds)
    routing = routing_metrics(truth["labels"][initial], p[initial] >= thresholds)
    models = {"VLM": {"thresholds": thresholds.tolist(), "decision_heads": metrics,
                       "initial_routing": routing, "structured_response_rate": float(valid.mean())}}
    timings = {}
    for name, filename in (("visual", "visual-support.npz"), ("proprio", "support-robot.npz")):
        model = MultiSupportVerifier.load(Path(protocol["reference"]) / filename)
        q = model.predict_proba(obs["rgb"], obs["robot"])
        models[name] = {"thresholds": model.thresholds.tolist(),
                        "decision_heads": head_metrics(truth["labels"][decision], q[decision], model.thresholds),
                        "initial_routing": routing_metrics(truth["labels"][initial], q[initial] >= model.thresholds)}
        original = json.loads((Path(protocol["reference"]) / "visual-head-report.json").read_text())
        original_mode = "image_robot" if name == "visual" else "robot"
        for target in TARGETS:
            for key in ("tp", "tn", "fp", "fn"):
                if models[name]["decision_heads"][target][key] != original["models"][original_mode]["dev_decisions"][target][key]:
                    raise ValueError("frozen comparator no longer matches its recorded development results")
        latency = []
        for index in np.flatnonzero(decision):
            started = time.perf_counter(); model.predict_proba(obs["rgb"][index:index+1], obs["robot"][index:index+1])
            latency.append(time.perf_counter()-started)
        timings[name] = {"device": "CPU", "observations": len(latency), "p50_seconds": float(np.median(latency)),
                         "p95_seconds": float(np.quantile(latency, .95))}
    runtime = json.loads((output / "dev-runtime.json").read_text())
    single = [r["end_to_end_seconds"] for r in runtime["warm_single_observation_timings"]]
    timings["VLM"] = {"device": runtime["gpu"], "observations": len(single),
                       "p50_seconds": float(np.median(single)), "p95_seconds": float(np.quantile(single, .95)),
                       "definition": "warm batch-1 preprocessing, four-image encoding, generation and decoding; image acquisition excluded",
                       "batched_amortized_seconds": float(np.mean([r["amortized_seconds_per_observation"] for r in runtime["batch_timings"] if not r["warmup"]]))}
    criteria = protocol["criteria"]
    checks = {"structured_responses": float(valid.mean()) >= criteria["minimum_structured_response_rate"],
              "giver_recognition": metrics["giver_supports"]["balanced_accuracy"] >= criteria["minimum_giver_balanced_accuracy"],
              "receiver_recall": metrics["receiver_supports"]["recall"] >= criteria["minimum_receiver_recall"],
              "coordinated_recall": metrics["coordinated_support"]["recall"] >= criteria["minimum_coordinated_recall"],
              "calibration_release_fpr": all(calibration[k]["false_positive_rate"] <= criteria["release_false_positive_budget"] for k in TARGETS[1:]),
              "initial_release_safety": routing["unsafe_continue_rate"] <= criteria["maximum_initial_unsafe_continue_rate"],
              "initial_routing_agreement": routing["agreement"] >= criteria["minimum_initial_route_agreement"],
              "latency_recorded": bool(single) and all(np.isfinite(single))}
    unchanged = all(file_hash(name) == value for name, value in protocol["preserved_input_sha256"].items())
    if not unchanged: raise ValueError("source data or frozen test artifacts changed")
    from .l4_selector import verify_frozen
    verify_frozen(json.loads(Path("outputs/l4-proprio-test-20261108/frozen-test.json").read_text()))
    result = {"scope": protocol["scope"], "model": protocol["model"], "selected_prompt": prompt_id,
              "training": selected, "counts": protocol["counts"], "dev_calibration": calibration,
              "models_on_same_development_observations": models, "inference_latency": timings,
              "all_source_data_and_frozen_test_files_unchanged": unchanged,
              "new_closed_loop_episodes": 0, "heldout_vlm_predictions": 0, "pretrained_weights_updated": False,
              "decision": {"result": "Go" if all(checks.values()) else "No-Go", "checks": checks,
                           "failed_checks": [k for k,v in checks.items() if not v],
                           "meaning": "admission to a future simulator closed-loop comparison, not measured physical VLM task success"}}
    (output / "vlm-report.json").write_text(json.dumps(result, indent=2) + "\n")
    rows = [{"model": name, "target": target, **v} for name, model in models.items() for target, v in model["decision_heads"].items()]
    with (output / "dev-head-comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "train", "select", "dev", "report"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference", default="outputs/l3-visual-20261107")
    parser.add_argument("--model-path", default="/data2/zh/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.phase == "prepare": result = prepare(args.output_dir, args.reference, args.model_path)
    elif args.phase == "select": result = select_prompt(args.output_dir)
    elif args.phase == "report": result = report(args.output_dir)
    else: result = infer(args.output_dir, args.phase, args.batch_size)
    print(json.dumps(result))
