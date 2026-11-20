"""Small RGB/robot support heads and a visual recovery scope selector."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .l1 import _features


TARGETS = ("giver_supports", "receiver_supports", "coordinated_support")
ROBOT_DIM = 46


def visual_features(rgb, robot, mode="image_robot", grid=8):
    robot = np.asarray(robot, dtype=np.float32)
    if robot.ndim != 2 or robot.shape[1] != ROBOT_DIM:
        raise ValueError("expected exactly 46 robot-only coordinates")
    if mode == "robot":
        return robot
    pooled = _features(rgb, np.zeros((len(robot), 20), dtype=np.float32), grid)[:, :-20]
    if mode == "image":
        return pooled
    if mode != "image_robot":
        raise ValueError(mode)
    return np.concatenate([pooled, robot], axis=1)


@dataclass
class MultiSupportVerifier:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    thresholds: np.ndarray
    mode: str = "image_robot"
    grid: int = 8

    def probabilities_from_features(self, x):
        logits = ((x - self.mean) / self.scale) @ self.weights + self.bias
        return 1 / (1 + np.exp(-np.clip(logits, -40, 40)))

    def predict_proba(self, rgb, robot):
        return self.probabilities_from_features(visual_features(rgb, robot, self.mode, self.grid))

    def save(self, path):
        np.savez_compressed(path, mean=self.mean, scale=self.scale, weights=self.weights,
                            bias=self.bias, thresholds=self.thresholds, mode=self.mode, grid=self.grid)

    @classmethod
    def load(cls, path):
        v = np.load(path)
        return cls(v["mean"], v["scale"], v["weights"], v["bias"], v["thresholds"], str(v["mode"]), int(v["grid"]))


def fit_heads(x, y, mode="image_robot", epochs=300, lr=.08, l2=.003):
    """Class-balanced logistic heads; only the training split enters fitting."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if y.shape != (len(x), 3) or np.any(y.sum(0) == 0) or np.any(y.sum(0) == len(y)):
        raise ValueError("each support target needs positive and negative training examples")
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-5] = 1
    z = (x - mean) / scale
    weights, bias = np.zeros((x.shape[1], 3)), np.zeros(3)
    prevalence = y.mean(0)
    balanced = y / (2 * prevalence) + (1 - y) / (2 * (1 - prevalence))
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-np.clip(z @ weights + bias, -40, 40)))
        error = (p - y) * balanced
        weights -= lr * (z.T @ error / len(y) + l2 * weights)
        bias -= lr * error.mean(0)
    return MultiSupportVerifier(mean, scale, weights, bias, np.full(3, .5), mode)


def binary_metrics(y, prediction):
    y, prediction = np.asarray(y, bool), np.asarray(prediction, bool)
    tp, tn = int(np.sum(y & prediction)), int(np.sum(~y & ~prediction))
    fp, fn = int(np.sum(~y & prediction)), int(np.sum(y & ~prediction))
    return {"frames": len(y), "positives": int(y.sum()), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": (tp + tn) / max(1, len(y)), "recall": tp / max(1, tp + fn),
            "precision": tp / max(1, tp + fp), "false_positive_rate": fp / max(1, fp + tn)}


def calibrate_heads(probabilities, y, false_positive_budget=.02):
    thresholds, reports = [], {}
    for i, name in enumerate(TARGETS):
        scores = np.unique(probabilities[:, i])
        candidates = sorted(set([.1, .2, .35, .5, .65, .8, .9, .95, .99, 1.000001,
                                 *(.5 * (scores[:-1] + scores[1:])).tolist()]))
        options = []
        for threshold in candidates:
            metrics = binary_metrics(y[:, i], probabilities[:, i] >= threshold)
            if i == 0:
                # Carrier recognition chooses R1 versus R3. Optimize both
                # classes here; the receiver/joint heads certify release.
                balanced_accuracy = .5 * (metrics["recall"] + 1 - metrics["false_positive_rate"])
                options.append((balanced_accuracy, metrics["recall"], -abs(threshold-.5), threshold, metrics))
            elif metrics["false_positive_rate"] <= false_positive_budget:
                options.append((metrics["recall"], metrics["precision"], -abs(threshold - .5), threshold, metrics))
        _, _, _, threshold, selected = max(options, key=lambda x: x[:3])
        thresholds.append(threshold)
        reports[name] = {"threshold": threshold, "selection": "balanced carrier accuracy" if i == 0 else "release false-positive constraint",
                         "false_positive_budget": None if i == 0 else false_positive_budget, **selected}
    return np.array(thresholds), reports


def load_feature_split(folder):
    """Join evaluator labels offline; stage names and timestamps are not features."""
    x, y, decisions, calibration = [], [], [], []
    for filename in sorted((Path(folder) / "robot").glob("*.npz")):
        observed = np.load(filename)
        truth = np.load(Path(folder) / "evaluator" / filename.name)
        if not len(observed["rgb"]):
            continue
        if not np.array_equal(observed["time"], truth["time"]):
            raise ValueError("robot and evaluator captures are misaligned")
        x.append(visual_features(observed["rgb"], observed["robot"]))
        y.append(truth["labels"])
        decisions.append(truth["decision"])
        # Include closed-but-unsupported local attempts that the baseline
        # subsequently hides by escalating to R3. Stage names stratify
        # offline calibration only and never enter the model features.
        calibration.append(truth["decision"] | (truth["stages"] == "r1_recovery_action"))
    return np.concatenate(x), np.concatenate(y), np.concatenate(decisions), np.concatenate(calibration)


def train_and_report(output, recalibrate=False):
    output = Path(output)
    train_x, train_y, _, _ = load_feature_split(output / "train")
    dev_x, dev_y, decision, calibration_mask = load_feature_split(output / "dev")
    report = {"train_frames": len(train_y), "dev_frames": len(dev_y), "dev_decision_frames": int(decision.sum()),
              "train_positive_counts": dict(zip(TARGETS, train_y.sum(0).tolist())),
              "dev_calibration_frames": int(calibration_mask.sum()),
              "calibration": "train fitting; dev decisions and local closing observations; balanced accuracy for carrier recognition, 2% FPR budget for receiver/joint release heads; dev score gaps included; stages used only for offline stratification",
              "weights_refitted_this_run": not recalibrate,
              "online_inputs": "four RGB images and exactly 46 robot-observable values; no stages, fault ids, object/contact truth",
              "models": {}}
    rng = np.random.default_rng(20261107)
    shuffled = dev_x.copy(); shuffled[:, :-ROBOT_DIM] = dev_x[rng.permutation(len(dev_x)), :-ROBOT_DIM]
    for mode in ("image_robot", "image", "robot"):
        select = (lambda x: x) if mode == "image_robot" else (lambda x: x[:, :-ROBOT_DIM]) if mode == "image" else (lambda x: x[:, -ROBOT_DIM:])
        model_path = output / ("visual-support.npz" if mode == "image_robot" else f"support-{mode}.npz")
        model = MultiSupportVerifier.load(model_path) if recalibrate else fit_heads(select(train_x), train_y, mode)
        p = model.probabilities_from_features(select(dev_x))
        model.thresholds, calibration = calibrate_heads(p[calibration_mask], dev_y[calibration_mask])
        score = lambda a, b: {name: binary_metrics(b[:, i], a[:, i] >= model.thresholds[i]) for i, name in enumerate(TARGETS)}
        report["models"][mode] = {"calibration": calibration,
            "train": score(model.probabilities_from_features(select(train_x)), train_y),
            "dev": score(p, dev_y), "dev_decisions": score(p[decision], dev_y[decision])}
        if mode == "image_robot":
            shuffled_p = model.probabilities_from_features(shuffled)
            report["models"][mode]["dev_decisions_shuffled_rgb"] = score(shuffled_p[decision], dev_y[decision])
        model.save(model_path)
    (output / "visual-head-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


class VisualRecoverySelector:
    """Continue on visual support, otherwise try local R1 or physical R3.

    The perception callback returns RGB and measured robot coordinates only.
    Skill return values include Oracle diagnostics in the legacy runner;
    they are deliberately discarded before every selection decision.
    """

    def __init__(self, verifier):
        self.verifier = verifier

    def run(self, env, expert, episode, capture, registered_positions, registered_rotations,
            reference_gaps, table_position, global_recovery, *, observe):
        checks = []

        def judge():
            observed = observe()
            probabilities = self.verifier.predict_proba(observed["rgb"][None], observed["robot"][None])[0]
            states = probabilities >= self.verifier.thresholds
            checks.append({"probabilities": dict(zip(TARGETS, probabilities.tolist())),
                           "thresholds": self.verifier.thresholds.tolist()})
            giver, receiver, coordinated = states
            # Receiver-only support can already complete the handover.
            ready = bool(receiver and (coordinated or not giver))
            return bool(giver), ready

        giver, ready = judge()
        sequence = []
        if ready:
            sequence.append("continue")
        elif giver:
            from .recovery_skills import PoseRetargetedR1Recovery
            PoseRetargetedR1Recovery().run(env, expert, episode, capture,
                                         registered_positions, registered_rotations)
            sequence.append("R1_retarget")
            _, ready = judge()
        used_global = not ready
        if used_global:
            global_recovery.run(env, expert, episode, capture, registered_positions,
                                reference_gaps, table_position)
            sequence.append("R3")
        return {"skill": "visual", "used_global_recovery": used_global,
                "sequence": sequence, "checks": checks,
                "decision_source": "RGB and robot-only three-head support estimates"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--recalibrate", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train_and_report(args.output_dir, args.recalibrate), ensure_ascii=False))
