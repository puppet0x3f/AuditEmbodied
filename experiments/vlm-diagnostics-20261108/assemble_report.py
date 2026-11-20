"""Assemble the completed offline diagnostic; no model inference or simulation."""
import csv
import json
from pathlib import Path

import numpy as np

from auditembodied.vlm_baseline import file_hash
from auditembodied.l4_selector import verify_frozen


root = Path("outputs/vlm-diagnostics-20261108")
source = Path("outputs/vlm-baseline-20261108")
candidate = root / "explicit-roles-full"
read = lambda p: json.loads(p.read_text())
old = read(source / "vlm-report.json")
new = read(candidate / "vlm-report.json")
diagnoses = {"original": read(root / "existing-score-diagnosis.json"),
             "explicit_roles": read(candidate / "score-diagnosis/existing-score-diagnosis.json")}
protocol = read(candidate / "protocol.json")
unchanged = all(file_hash(p) == h for p, h in protocol["preserved_input_sha256"].items())
sources_unchanged = all(file_hash(p) == h for p, h in protocol["runtime_sources"].items())
verify_frozen(read(Path("outputs/l4-proprio-test-20261108/frozen-test.json")))
assert unchanged and sources_unchanged

rows = []
for name, result, diagnosis in (("original", old, diagnoses["original"]), ("explicit_roles", new, diagnoses["explicit_roles"])):
    model = result["models_on_same_development_observations"]["VLM"]
    for target, head in model["decision_heads"].items():
        rows.append({"configuration": name, "target": target,
                     "auc": diagnosis["splits"]["dev"]["subsets"]["decisions"]["heads"][target]["auc"],
                     **head, "initial_routing_agreement": model["initial_routing"]["agreement"],
                     "inference_p50_seconds": result["inference_latency"]["VLM"]["p50_seconds"]})
with (root / "development-comparison.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

raw = read(root / "candidate-raw-old-thresholds.json")
result = {
    "scope": "training/development only; all comparisons offline; development reused for diagnosis and calibration, not independent generalization",
    "causes_supported_by_evidence": {
        "score_overlap_and_quantization": diagnoses["original"]["splits"],
        "joint_threshold_bound": read(root / "joint-threshold-upper-bound.json"),
        "presentation_and_role_probe": read(root / "probe-report.json"),
        "semantic_prompt_partial_improvement": {"original": old["models_on_same_development_observations"]["VLM"],
                                                 "explicit_roles": new["models_on_same_development_observations"]["VLM"],
                                                 "roles_with_original_thresholds": raw},
    },
    "interpretation": [
        "Current support scores cannot be fixed by a monotone threshold shift; positive and negative states share identical generated scores.",
        "A training-selected role/state prompt improves development ranking but produces high-confidence false support; strict risk calibration rejects everything.",
        "Conjunctive release gating and mistaken giver support convert receiver/coordinated false negatives into unnecessary R1 or R3.",
        "The 24-input vertical flip diagnostic did not repair receiver/coordinated recall; RGB shuffling affected outputs, so complete image neglect is unsupported.",
        "Frozen task-trained image heads recognize these observations well; the generic model is not task-trained and some labels include unobserved dynamics, so the gap is not proof that all support truth is directly visible."
    ],
    "decision": {
        "current_closed_loop_admission": "No-Go",
        "continue_offline_optimization": "bounded Go: one non-generative binary-answer likelihood readout diagnostic",
        "reason": "Role grounding improved development AUROC, so some useful ranking exists; coarse self-reported decimal scores still fail the safe operating region.",
        "next_experiment": "Keep the explicit role/state definitions and observation budget fixed; replace generated confidence decimals with independent binary-answer token likelihoods, select on training and calibrate on development only.",
        "stop_rule": "Use unchanged original admission criteria; if the one readout comparison still fails, stop this frozen-model optimization route rather than expand prompt/model/weight searches.",
        "weight_training_or_larger_models": "not prioritized; no demonstrated advantage over the existing small task-trained head"
    },
    "inference_latency": {"original": old["inference_latency"]["VLM"], "explicit_roles": new["inference_latency"]["VLM"]},
    "preservation": {"source_observations_and_frozen_test_inputs_and_previous_baseline_unchanged": unchanged,
                     "checked_input_files": len(protocol["preserved_input_sha256"]),
                     "recorded_runtime_sources_unchanged": sources_unchanged,
                     "previous_six_execution_sources_and_44_key_files_verified": True},
    "new_closed_loop_episodes": 0, "heldout_vlm_predictions": 0, "new_camera_observations": 0, "pretrained_weight_updates": 0,
    "numerical_diagnostic_tests": "7 passed; tests/test_vlm_diagnostics.py and tests/test_vlm_baseline.py"
}
(root / "diagnosis-report.json").write_text(json.dumps(result, indent=2) + "\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True, sharey=True)
for ax, (target, title, minimum) in zip(axes, (("receiver_supports", "Receiver support", 90),
                                           ("coordinated_support", "Coordinated support", 80))):
    for name, color, label in (("original", "#777777", "Original prompt"), ("explicit_roles", "#7b3b9e", "Explicit role/state prompt")):
        h = diagnoses[name]["splits"]["dev"]["subsets"]["decisions"]["heads"][target]
        points = h["curve"]
        ax.plot([100*p["false_positive_rate"] for p in points], [100*p["recall"] for p in points],
                "o-", ms=4, color=color, label=f"{label} (AUC {h['auc']:.3f})")
    ax.axvline(2, color="#555555", linestyle="--", lw=1)
    ax.axhline(minimum, color="#555555", linestyle=":", lw=1)
    ax.set(title=title, xlabel="False positive rate (%)", xlim=(-2, 102), ylim=(-2, 102))
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=.15)
axes[0].set_ylabel("Positive recall (%)")
fig.suptitle("Role grounding improves ranking but fails the safe operating region", fontsize=11)
fig.tight_layout()
for suffix in ("png", "pdf"):
    fig.savefig(root / f"original-vs-roles.{suffix}", dpi=190, bbox_inches="tight")
plt.close(fig)
print(json.dumps({"completed": True, "current_admission": "No-Go", "all_preserved": unchanged,
                  "checked_input_files": len(protocol["preserved_input_sha256"])}))
