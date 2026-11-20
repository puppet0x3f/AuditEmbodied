"""Fixed-budget native-token Qwen3-VL LoRA training on executed recovery data.

Only observation, task and candidate-plan fields enter the model. Seed and
scenario metadata are used for splitting/pairing and never enter prompts.
This is a training/development check, not a closed-loop policy evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np


INSTRUCTION = """Predict the outcome of a specified recovery plan for two Panda arms lifting a handled pot.
Robot 0 is left; robot 1 is right. Camera order: external front, external agent, left wrist, right wrist.
Use only the images, measured robot coordinates and the candidate plan.
Goal: both handles grasped and stably supported, pot clearance >=0.10 m, tilt <=15 degrees,
handle height difference <=0.03 m, sustained for a full second. A global restart must truly set down and regrasp.
Process guards: a local plan must keep its keeper grasp throughout recovery; avoid deep inter-arm collision
and intervals without any gripper or table support longer than 0.10 seconds.
Control rate is 20 Hz. Motion geometry, gains and force limits are fixed; only listed local step counts vary.
Choose the predicted JOINT outcome, not the currently observed support state:
A: task completed, process guards passed.
B: task completed, process guard violated.
C: task not completed, process guards passed.
D: task not completed, process guard violated.
E: required keeper grasp absent at the start of the proposed local plan.
Answer exactly one letter A, B, C, D or E."""


def prompt_for(example):
    return INSTRUCTION + "\nMeasured robot state (metres): " + json.dumps(example["robot"], separators=(",", ":")) + \
        "\nCandidate plan: " + json.dumps(example["plan"], separators=(",", ":"))


START_INSTRUCTION = """Check only the CURRENT prerequisite for a local recovery by two Panda arms lifting a handled pot.
Robot 0 is left; robot 1 is right. Camera order: external front, external agent, left wrist, right wrist.
Use the images, measured robot coordinates and repair/keeper roles. Is the specified keeper gripper
already grasping its handle? Do not predict future success or future loss of support.
A: required keeper grasp is present now.
E: required keeper grasp is absent now.
Answer exactly one letter A or E."""

OUTCOME_INSTRUCTION = INSTRUCTION.replace(
    "Choose the predicted JOINT outcome, not the currently observed support state:",
    "For a local plan, predict conditionally on its required keeper grasp being present at the start.\n"
    "This condition does not imply that the keeper remains supported during execution.\n"
    "Choose the predicted JOINT outcome of the future plan:").replace(
    "E: required keeper grasp absent at the start of the proposed local plan.\n", "").replace(
    "Answer exactly one letter A, B, C, D or E.", "Answer exactly one letter A, B, C or D.")


def start_prompt_for(example):
    roles = {k: example["plan"][k] for k in ("repair_arm", "keeper_arm")}
    return START_INSTRUCTION + "\nMeasured robot state (metres): " + json.dumps(example["robot"], separators=(",", ":")) + \
        "\nLocal recovery roles: " + json.dumps(roles, separators=(",", ":"))


def start_key(example):
    return json.dumps([example["image"], example["robot"],
                       example["plan"]["repair_arm"], example["plan"]["keeper_arm"]], sort_keys=True)


def observed_state_key(example):
    # A seed/fault can now have both h0 and post-wait h1 observations.
    return (example['seed'],example['scenario'],example.get('image',''),
            json.dumps(example.get('robot',{}),sort_keys=True))


def task_prompt_for(example, task):
    if task == "start":
        return start_prompt_for(example)
    return prompt_for(example).replace(INSTRUCTION, OUTCOME_INSTRUCTION, 1) if task == "outcome" else prompt_for(example)


def task_label(example, task):
    if task == 'support_time':
        return example['support_time']['label']
    return ("E" if example["label"] == "E" else "A") if task == "start" else example["label"]


def outcome_updates(train, steps, seed, tiny=False):
    """Cycle executed A-D outcomes with a frozen, separate start adapter."""
    valid = [i for i, r in enumerate(train) if r["label"] != "E"]
    matched = {}
    for i in valid:
        r, p = train[i], train[i]["plan"]
        if p["kind"] == "local" and p["local_relevel_steps"] in (40, 60):
            fixed = {k: v for k, v in p.items() if k != "local_relevel_steps"}
            matched.setdefault((start_key(r), json.dumps(fixed, sort_keys=True)), {})[p["local_relevel_steps"]] = i
    strict = []
    for pair in matched.values():
        if set(pair) == {40, 60} and train[pair[40]]["label"] == "B" and train[pair[60]]["label"] == "A":
            strict.append((pair[60], pair[40]))
    if tiny:
        incomplete = [i for i in valid if train[i]["label"] == "C" and train[i]["plan"]["kind"] == "local"][:4]
        safe40 = [next(i for i in valid if train[i]["label"] == "A" and train[i]["plan"]["kind"] == "local"
                       and train[i]["plan"]["local_relevel_steps"] == 40 and start_key(train[i]) == start_key(train[j]))
                  for j in incomplete]
        unsafe_incomplete = [i for i in valid if train[i]["label"] == "D"]
        selected = sorted(set(incomplete+safe40+unsafe_incomplete+[i for pair in strict for i in pair]))
        if len(strict) != 2 or len(selected) != 16 or len(incomplete) != 4 or len(unsafe_incomplete) != 4:
            raise ValueError("tiny diagnostic requires two strict B/A pairs, four safe40, four local C and four D")
    else:
        selected = valid
    paired_indices = {i for pair in strict for i in pair}
    groups = strict+[(i,) for i in selected if i not in paired_indices]
    rng, updates = random.Random(seed), []
    while len(updates) < steps:
        order = groups.copy()
        rng.shuffle(order)
        for indices in order:
            if len(updates) == steps:
                break
            updates.append({"kind": "strict_timing" if len(indices) == 2 else "outcome_cycle",
                            "task": "outcome", "indices": list(indices),
                            "labels": [train[i]["label"] for i in indices]})
    counts = Counter(i for update in updates for i in update["indices"])
    return updates, {"cohort_indices": selected, "cohort_examples": len(selected),
                     "cohort_classes": dict(Counter(train[i]["label"] for i in selected)),
                     "cohort_kinds": dict(Counter(train[i]["plan"]["kind"] for i in selected)),
                     "update_counts": dict(Counter(r["kind"] for r in updates)),
                     "presentations": sum(counts.values()),
                     "candidate_presentations": dict(counts),
                     "note": "Shuffled complete cycles; strict B/A candidates share an update, CE is averaged as before."}


def risk_replay_updates(train, previous, steps, seed, threshold=None):
    """Training-only hard-risk replay with matched safe actions and A/A controls."""
    if len(previous) != len(train) or any(
        any(a[k] != b[k] for k in ("seed", "scenario", "plan", "label")) for a, b in zip(train, previous)
    ):
        raise ValueError("hard reference must match the training set, in order")
    admitted=lambda i:(previous[i]['class_probabilities'][0]>=threshold if threshold is not None
                       else previous[i]['prediction']=='A')
    hard_B = [i for i, r in enumerate(train) if r["label"] == "B" and admitted(i)]
    hard_C = [i for i, r in enumerate(train) if r["label"] == "C" and admitted(i)]
    hard_A = [i for i, r in enumerate(train) if r["label"] == "A" and not admitted(i)]
    good, safe_groups = {}, {}
    for i, r in enumerate(train):
        if r["label"] == "A" and r["plan"]["kind"] == "local":
            key = start_key(r)
            good.setdefault(key, []).append(i)
            fixed = {k: v for k, v in r["plan"].items() if k != "local_relevel_steps"}
            safe_groups.setdefault((key, json.dumps(fixed, sort_keys=True)), []).append(i)

    def observed_pair_or_single(negative):
        r = train[negative]
        choices = good.get(start_key(r), [])
        if not choices:
            return (negative,)
        fields = ("local_approach_steps", "local_close_steps", "local_relevel_steps")
        positive=min(choices, key=lambda i: (sum(abs(train[i]["plan"][k]-r["plan"][k]) for k in fields), i))
        return (positive,negative)

    controls = [(a, b) for group in safe_groups.values() for j, a in enumerate(group)
                for b in group[j+1:] if train[a]["plan"]["local_relevel_steps"] != train[b]["plan"]["local_relevel_steps"]]
    retained_process = [i for i, r in enumerate(train) if r["label"] in "BD" and i not in hard_B]
    if not all((hard_B, hard_C, hard_A, controls, retained_process)):
        raise ValueError("risk replay requires hard B/C, rejected safe A, retained process failures and A/A controls")
    pools = {
        "new_B_pair": [observed_pair_or_single(i) for i in hard_B],
        "process_pair": [observed_pair_or_single(i) for i in retained_process],
        "C_pair": [observed_pair_or_single(i) for i in hard_C],
        "AA_pair": controls,
        "hard_A": [(i,) for i in hard_A],
        "background": [(i,) for i, r in enumerate(train) if r["label"] != "E"],
    }
    block = ["new_B_pair"]*3+["process_pair"]+["C_pair"]*2+["AA_pair"]*3+["hard_A"]*2+["background"]*5
    if threshold is not None:
        # 192 background presentations at 512 updates cover all 180 valid
        # combined examples, while retaining train-only hard-risk replay.
        block = ["new_B_pair"]*3+["process_pair"]+["C_pair"]*2+["AA_pair"]*3+["hard_A"]+["background"]*6
    rng, queues, updates = random.Random(seed), {}, []
    while len(updates) < steps:
        order = block.copy()
        rng.shuffle(order)
        for kind in order:
            if len(updates) == steps:
                break
            if not queues.get(kind):
                queues[kind] = pools[kind].copy()
                rng.shuffle(queues[kind])
            indices = list(queues[kind].pop())
            updates.append({"kind": kind, "task": "outcome", "indices": indices,
                            "labels": [train[i]["label"] for i in indices]})
    counts = Counter(i for r in updates for i in r["indices"])
    return updates, {"hard_B_indices": hard_B, "hard_C_indices": hard_C, "hard_A_indices": hard_A,
        "hard_B_positive_pairs": [list(pair) for pair in pools["new_B_pair"] if len(pair)==2],
        "C_positive_pairs": [list(pair) for pair in pools["C_pair"] if len(pair)==2],
        "unpaired_risk_indices": sorted({x[0] for kind in ('new_B_pair','process_pair','C_pair') for x in pools[kind] if len(x)==1}),
        "fixed_admission_threshold":threshold,
        "AA_control_pairs": len(controls), "retained_process_candidates": len(retained_process),
        "update_counts": dict(Counter(r["kind"] for r in updates)),
        "presentations": sum(counts.values()), "candidate_presentations": dict(counts),
        "class_presentations": dict(Counter(c for r in updates for c in r["labels"])),
        "note": "Only previous TRAIN errors determine replay. Actual safe/unsafe pairs share image, robot observations and roles; negatives without an observed safe alternative use single-example CE and no ranking loss. A/A controls have no ranking loss."}


def tiny_fit_gate(records):
    conditional = [{**r, "prediction": r.get("conditional_prediction", r["prediction"])} for r in records]
    timing = timing_contrasts(conditional)
    correct = sum(r["label"] == r["prediction"] for r in conditional)
    false_cd = sum(r["label"] in "CD" and r["prediction"] == "A" for r in conditional)
    return {"passed": len(records) == 16 and correct >= 15 and
                      timing["changed_safety_pairs"] == timing["changed_pair_both_classes_correct"] == 2 and false_cd == 0,
            "examples": len(records), "correct": correct, "required_correct": 15,
            "strict_pairs": timing["changed_safety_pairs"],
            "strict_pairs_both_classes_correct": timing["changed_pair_both_classes_correct"],
            "C_D_false_safe": false_cd,
            "note": "Training-only fit check; passing permits full-data training, not policy execution."}


def balanced_updates(train, previous, steps, seed):
    """Restore observed safe controls while learning duration-free prerequisites."""
    if len(previous) != len(train) or any(
        any(a[k] != b[k] for k in ("seed", "scenario", "plan", "label")) for a, b in zip(train, previous)
    ):
        raise ValueError("hard reference must match the training set, in order")
    rng = random.Random(seed)
    by_class = {c: [i for i, r in enumerate(train) if r["label"] == c] for c in "ABCDE"}
    roles, matched = {}, {}
    for i, r in enumerate(train):
        if r["plan"]["kind"] != "local":
            continue
        key = start_key(r)
        if key in roles and (train[roles[key]]["label"] == "E") != (r["label"] == "E"):
            raise ValueError("one observed role has conflicting prerequisite labels")
        roles.setdefault(key, i)
        plan = r["plan"]
        if plan["local_relevel_steps"] in (40, 60):
            fixed = {k: v for k, v in plan.items() if k != "local_relevel_steps"}
            matched.setdefault((key, json.dumps(fixed, sort_keys=True)), {})[plan["local_relevel_steps"]] = i
    strict, safe_controls = [], []
    for group in matched.values():
        if set(group) == {40, 60}:
            a, b = group[40], group[60]
            if {train[a]["label"], train[b]["label"]} == {"A", "B"}:
                strict.append((a, b) if train[a]["label"] == "A" else (b, a))
    safe40 = [i for i in by_class["A"] if train[i]["plan"]["kind"] == "local"
              and train[i]["plan"]["local_relevel_steps"] == 40]
    for a in safe40:
        for b in by_class["A"]:
            if train[b]["plan"]["kind"] == "local" and start_key(train[a]) == start_key(train[b]) and \
                    train[b]["plan"]["local_relevel_steps"] != 40:
                safe_controls.append((a, b))
    hard40 = [i for i in safe40 if previous[i]["prediction"] != "A"]
    start_pairs = []
    for key, i in roles.items():
        if train[i]["label"] == "E":
            continue
        for other, j in roles.items():
            if train[j]["label"] == "E" and train[i]["image"] == train[j]["image"] and train[i]["robot"] == train[j]["robot"]:
                start_pairs.append((i, j))
    if not strict or not safe_controls or not start_pairs or not safe40:
        raise ValueError("balanced training needs observed timing flips, safe controls and both start roles")
    # 8 paired and 8 single-example updates per 16 steps: 768 presentations
    # at 512 steps, matching the preceding targeted run. A/A controls have no
    # ranking loss; their other phase durations may differ and are NOT called
    # strict timing controls. No unexecuted outcome is invented.
    block = ["strict_timing"]*2 + ["safe_control"]*2 + ["safe_40"]*2 + ["start_roles"]*4 + \
            ["incomplete"]*2 + ["process"]*2 + ["background"]*2
    pools = {"strict_timing": strict, "safe_control": safe_controls, "start_roles": start_pairs,
             "safe_40": [(i,) for i in (hard40 or safe40)], "incomplete": [(i,) for i in by_class["C"]]}
    queues, used, updates = {}, Counter(), []

    def cycle(kind, pool):
        if not queues.get(kind):
            queues[kind] = list(pool)
            rng.shuffle(queues[kind])
        return list(queues[kind].pop())

    while len(updates) < steps:
        order = block.copy()
        rng.shuffle(order)
        for kind in order:
            if len(updates) == steps:
                break
            if kind in pools:
                indices = cycle(kind, pools[kind])
            else:
                c = "BD"[used[kind] % 2] if kind == "process" else "ABCD"[used[kind] % 4]
                indices = cycle(kind+c, [(i,) for i in by_class[c]])
            used[kind] += 1
            task = "start" if kind == "start_roles" else "outcome"
            updates.append({"kind": kind, "task": task, "indices": indices,
                            "labels": [task_label(train[i], task) for i in indices]})
    strict_safe_controls = sum(all(train[a]["plan"][k] == train[b]["plan"][k]
                                   for k in train[a]["plan"] if k != "local_relevel_steps") for a, b in safe_controls)
    return updates, {"update_counts": dict(Counter(x["kind"] for x in updates)),
                     "presentations": sum(len(x["indices"]) for x in updates),
                     "answer_presentations": dict(Counter(c for x in updates for c in x["labels"])),
                     "distinct_strict_timing_pairs": len(strict), "distinct_safe_control_pairs": len(safe_controls),
                     "strict_safe_control_pairs": strict_safe_controls,
                     "safe_40_examples": len(safe40), "hard_safe_40_examples": len(hard40),
                     "unique_start_roles": len(roles), "start_role_pairs": len(start_pairs),
                     "safe_control_note": "Observed same-state safe actions; other phase durations may differ. Not strict timing controls."}


def conditional_completion_loss(logits, ids, label):
    import torch
    import torch.nn.functional as F
    v = logits[ids[:4]]
    completed, incomplete = torch.logsumexp(v[:2], 0), torch.logsumexp(v[2:], 0)
    return F.softplus(incomplete-completed) if label in "AB" else F.softplus(completed-incomplete)


def conditional_process_loss(logits, ids, label):
    """Process safety is the A/C versus B/D marginal of the SAME joint logits."""
    import torch
    import torch.nn.functional as F
    if label not in 'ABCD':
        raise ValueError('process outcome is undefined for an invalid start')
    v = logits[ids[:4]]
    safe, violation = torch.logsumexp(v[[0, 2]], 0), torch.logsumexp(v[[1, 3]], 0)
    return F.softplus(violation - safe) if label in 'AC' else F.softplus(safe - violation)


def family_pair_loss(values, ids, labels):
    """Rank actual joint-admission and process flips, leaving A/A unordered."""
    import torch
    import torch.nn.functional as F
    native = [v[ids[:4]] for v in values]
    joint = [v[0] - torch.logsumexp(v[1:], 0) for v in native]
    process = [torch.logsumexp(v[[0, 2]], 0) - torch.logsumexp(v[[1, 3]], 0) for v in native]
    terms, counts = [], {'joint': 0, 'process': 0}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            for name, scores, positive in [('joint', joint, lambda c: c == 'A'),
                                           ('process', process, lambda c: c in 'AC')]:
                if positive(labels[i]) != positive(labels[j]):
                    a, b = (i, j) if positive(labels[i]) else (j, i)
                    terms.append(F.softplus(-(scores[a] - scores[b])))
                    counts[name] += 1
    return (torch.stack(terms).mean() if terms else values[0].new_zeros(())), counts


def joint_marginal_metrics(records):
    rows = [r for r in records if r['label'] in 'ABCD']
    violation = lambda r: r['conditional_probabilities'][1] + r['conditional_probabilities'][3] >= .5
    complete = lambda r: r['conditional_probabilities'][0] + r['conditional_probabilities'][1] >= .5
    return {'examples': len(rows),
        'process_correct': sum(violation(r) == (r['label'] in 'BD') for r in rows),
        'process_missed_violations': sum(not violation(r) for r in rows if r['label'] in 'BD'),
        'process_false_alarms': sum(violation(r) for r in rows if r['label'] in 'AC'),
        'completion_correct': sum(complete(r) == (r['label'] in 'AB') for r in rows),
        'note': 'Two binary marginals from the same conditional A-D distribution;0.5 is diagnostic class argmax, not a new admission threshold.'}


def admission_boundary_loss(logits, ids, label, start_score, threshold, margin=1.):
    """Align the native joint A score to an unchanged policy threshold."""
    import math
    import torch
    import torch.nn.functional as F
    v = logits[ids[:4]]
    normalizer = torch.logsumexp(v, dim=0)
    log_q = math.log(start_score)
    log_joint = log_q + v[0] - normalizer
    # 1 - q*P(A) = (1-q) + q*P(B/C/D). Avoid subtracting a
    # rounded-to-one softmax, which loses gradients at this high threshold.
    log_not_q = math.log1p(-start_score) if start_score < 1. else -math.inf
    log_not_joint = torch.logaddexp(v.new_tensor(log_not_q),
        log_q + torch.logsumexp(v[1:], dim=0) - normalizer)
    score = log_joint - log_not_joint
    boundary = math.log(threshold) - math.log1p(-threshold)
    return F.softplus(boundary + margin - score) if label == 'A' else F.softplus(score - boundary + margin)


def family_set_admission_loss(values, ids, labels, start_scores, plans, threshold, margin=1.):
    """Reject every executed non-A; admit the cheapest actually safe option.

    All A candidates retain their ordinary classification supervision. This
    boundary term does not require every safe alternative to pass at once.
    """
    import torch
    indices=[i for i,label in enumerate(labels) if label!='A']
    safe=[i for i,label in enumerate(labels) if label=='A']
    if safe:indices.append(min(safe,key=lambda i:plans[i]['local_relevel_steps']))
    terms=[admission_boundary_loss(values[i],ids,labels[i],start_scores[i],threshold,margin) for i in indices]
    return torch.stack(terms).mean()


def fixed_admission_metrics(records, threshold):
    """Report the frozen policy boundary separately from native class argmax."""
    positive = [r for r in records if r['label'] == 'A']
    negative = [r for r in records if r['label'] != 'A']
    accepted = lambda r: r['class_probabilities'][0] >= threshold
    groups = {}
    for row in records:
        groups.setdefault(observed_state_key(row), []).append(row)
    no_safe = [rows for rows in groups.values() if not any(r['label'] == 'A' for r in rows)]
    native = lambda r: r.get('conditional_prediction', r['prediction'])
    return {'threshold': threshold, 'positive_examples': len(positive), 'negative_examples': len(negative),
        'safe_admitted': sum(accepted(r) for r in positive),
        'false_admitted': sum(accepted(r) for r in negative),
        'non_A_rejected': sum(not accepted(r) for r in negative),
        'native_correct': sum(native(r) == r['label'] for r in records),
        'native_A_vs_non_A_correct': sum((native(r) == 'A') == (r['label'] == 'A') for r in records),
        'native_confusion': {c: dict(Counter(native(r) for r in records if r['label'] == c)) for c in 'ABCDE'},
        'no_safe_families': len(no_safe),
        'no_safe_families_rejected': sum(not any(accepted(r) for r in rows) for rows in no_safe),
        'passed': bool(records) and all(accepted(r) for r in positive) and not any(accepted(r) for r in negative),
        'note': 'Fixed operating point, not a threshold search. Native classification and actual admission are distinct.'}


def timing_contrasts(records):
    """Match local plans that differ only in 40 versus 60 relevel steps."""
    groups = {}
    for r in records:
        plan = r["plan"]
        if plan["kind"] != "local" or plan.get("local_relevel_steps") not in (40, 60):
            continue
        unchanged = {k: v for k, v in plan.items() if k != "local_relevel_steps"}
        key = (*observed_state_key(r),json.dumps(unchanged, sort_keys=True))
        groups.setdefault(key, {})[plan["local_relevel_steps"]] = r
    contrasts = []
    for (seed, scenario, image, _, _), group in sorted(groups.items()):
        if set(group) != {40, 60}:
            continue
        a, b = group[40], group[60]
        pa, pb = a["class_probabilities"][0], b["class_probabilities"][0]
        changed_safety = (a["label"] == "A") != (b["label"] == "A")
        ordered = ((pb > pa) if b["label"] == "A" else (pa > pb)) if changed_safety else None
        contrasts.append({"seed": seed, "scenario": scenario, "image":image,"repair_arm": a["plan"]["repair_arm"],
                          "label_40": a["label"], "label_60": b["label"],
                          "prediction_40": a["prediction"], "prediction_60": b["prediction"],
                          "p_safe_40": pa, "p_safe_60": pb, "p_safe_delta_60_minus_40": pb-pa,
                          "changed_safety": changed_safety, "safe_order_correct": ordered,
                          "both_joint_classes_correct": a["prediction"] == a["label"] and b["prediction"] == b["label"],
                          "both_safety_decisions_correct": (a["prediction"] == "A") == (a["label"] == "A") and
                                                           (b["prediction"] == "A") == (b["label"] == "A")})
    changed = [r for r in contrasts if r["changed_safety"]]
    return {"matched_pairs": len(contrasts), "changed_safety_pairs": len(changed),
            "changed_pair_order_correct": sum(r["safe_order_correct"] for r in changed),
            "changed_pair_both_safety_correct": sum(r["both_safety_decisions_correct"] for r in changed),
            "changed_pair_both_classes_correct": sum(r["both_joint_classes_correct"] for r in changed),
            "cases": contrasts}


def prediction_metrics(records):
    accepted = [r for r in records if r["prediction"] == "A"]
    positives = [r for r in records if r["label"] == "A"]
    negatives = [r for r in records if r["label"] != "A"]
    p = lambda r: r["class_probabilities"][0]
    groups = {}
    for r in records:
        groups.setdefault(observed_state_key(r), []).append(r)
    pairs = [(a, b) for group in groups.values() for a in group if a["label"] == "A"
             for b in group if b["label"] != "A"]
    local = [(a, b) for a, b in pairs if a["plan"]["kind"] == b["plan"]["kind"] == "local"]
    completed_unsafe = [(a, b) for a, b in local if b["label"] == "B"]
    ordering = lambda xs: {"correct": sum(p(a) > p(b) for a, b in xs), "pairs": len(xs),
                           "accuracy": sum(p(a) > p(b) for a, b in xs)/len(xs) if xs else None}
    auc = (sum((p(a) > p(b)) + .5*(p(a) == p(b)) for a in positives for b in negatives)
           / (len(positives)*len(negatives))) if positives and negatives else None
    zero_error = []
    for threshold in sorted({p(r) for r in records}):
        chosen = [r for r in records if p(r) >= threshold]
        if all(r["label"] == "A" for r in chosen):
            zero_error.append({"threshold": threshold, "accepted": len(chosen)})
    return {"examples": len(records), "accuracy": sum(r["label"] == r["prediction"] for r in records)/len(records),
            "safe_accepts": len(accepted), "false_safe_accepts": sum(r["label"] != "A" for r in accepted),
            "safe_precision": sum(r["label"] == "A" for r in accepted)/len(accepted) if accepted else None,
            "safe_recall": sum(r["prediction"] == "A" for r in positives)/len(positives) if positives else None,
            "safe_probability_auroc": auc,
            "mean_class_token_mass": float(np.mean([r["class_token_mass"] for r in records])),
            "predicted_counts": dict(Counter(r["prediction"] for r in records)),
            "confusion": {c: dict(Counter(r["prediction"] for r in records if r["label"] == c)) for c in "ABCDE"},
            "false_safe_by_class": {c: {"examples": sum(r["label"] == c for r in records),
                                         "accepted": sum(r["label"] == c and r["prediction"] == "A" for r in records)}
                                    for c in "BCDE"},
            "process_violations": {"examples": sum(r["label"] in ("B", "D") for r in records),
                                   "accepted": sum(r["label"] in ("B", "D") and r["prediction"] == "A" for r in records)},
            "within_state_ranking": ordering(pairs), "within_state_local_ranking": ordering(local),
            "within_state_local_A_vs_B": ordering(completed_unsafe),
            "timing_40_60": timing_contrasts(records),
            "largest_empirical_zero_error_acceptance": max(zero_error, key=lambda r: r["accepted"]) if zero_error else None,
            "note": "Uncalibrated restricted-token argmax; training/development diagnostic only. Zero-error acceptance is post-hoc, not a selected operating point."}


def targeted_updates(train, previous, steps, seed):
    """Fixed training-only mix: exact timing pairs, C/E pairs and hard singles."""
    if len(previous) != len(train) or any(
        any(a[k] != b[k] for k in ("seed", "scenario", "plan", "label"))
        for a, b in zip(train, previous)
    ):
        raise ValueError("hard reference must match the training set, in order")
    rng = random.Random(seed)
    by_class = {c: [i for i, r in enumerate(train) if r["label"] == c] for c in "ABCDE"}
    if any(not by_class[c] for c in "ABCDE"):
        raise ValueError("the targeted diagnostic requires all five existing training classes")
    hard = {c: [i for i in by_class[c] if previous[i]["prediction"] == "A"] for c in "CE"}
    good = {}
    for i in by_class["A"]:
        r = train[i]
        if r["plan"]["kind"] == "local":
            good.setdefault((r["seed"], r["scenario"]), []).append(i)
    matched = {}
    for i, r in enumerate(train):
        plan = r["plan"]
        if plan["kind"] == "local" and plan.get("local_relevel_steps") in (40, 60):
            fixed = {k: v for k, v in plan.items() if k != "local_relevel_steps"}
            key = r["seed"], r["scenario"], json.dumps(fixed, sort_keys=True)
            matched.setdefault(key, {})[plan["local_relevel_steps"]] = i
    strict = []
    for group in matched.values():
        if set(group) != {40, 60}:
            continue
        a, b = group[40], group[60]
        if (train[a]["label"] == "A") == (train[b]["label"] == "A"):
            continue
        if train[a]["image"] != train[b]["image"] or train[a]["robot"] != train[b]["robot"]:
            raise ValueError("strict timing pair observations differ")
        strict.append((a, b) if train[a]["label"] == "A" else (b, a))
    if not strict:
        raise ValueError("no strictly matched training timing contrasts")
    # Eight paired and eight single-example updates per 16 steps. This keeps
    # the number of presentations close to the original 50% pair sampler.
    block = (["strict_timing"]*4 + ["feasibility_pair"]*2 + ["completion_pair"]*2 +
             ["hard_E"]*3 + ["hard_C"]*3 + ["background"]*2)
    updates, strict_used = [], 0

    def negative(c, require_local_positive=False):
        pool = hard[c] if hard[c] and rng.random() < .75 else by_class[c]
        if require_local_positive:
            pool = [i for i in pool if (train[i]["seed"], train[i]["scenario"]) in good]
        if not pool:
            raise ValueError("no same-state positive for the targeted negative")
        return rng.choice(pool)

    while len(updates) < steps:
        order = block.copy()
        rng.shuffle(order)
        for kind in order:
            if len(updates) == steps:
                break
            if kind == "strict_timing":
                indices = list(strict[strict_used % len(strict)])
                strict_used += 1
            elif kind in ("feasibility_pair", "completion_pair"):
                n = negative("E" if kind == "feasibility_pair" else "C", True)
                r = train[n]
                indices = [rng.choice(good[r["seed"], r["scenario"]]), n]
            elif kind in ("hard_E", "hard_C"):
                indices = [negative(kind[-1])]
            else:
                indices = [rng.choice(by_class[rng.choice("ABD")])]
            updates.append({"kind": kind, "indices": indices,
                            "labels": [train[i]["label"] for i in indices]})
    return updates, {"update_counts": dict(Counter(x["kind"] for x in updates)),
                     "presentations": sum(len(x["indices"]) for x in updates),
                     "class_presentations": dict(Counter(c for x in updates for c in x["labels"])),
                     "strict_pairs": [{"positive_index": a, "negative_index": b,
                                       "seed": train[a]["seed"], "scenario": train[a]["scenario"]} for a, b in strict],
                     "hard_reference_counts": {c: len(hard[c]) for c in hard},
                     "hard_negative_probability": .75}


def auxiliary_task_loss(logits, ids, label):
    """Native-token marginal losses for start feasibility and completion."""
    import torch
    import torch.nn.functional as F
    v = logits[ids]
    valid = torch.logsumexp(v[:4], dim=0)
    feasibility = F.softplus(valid-v[4]) if label == "E" else F.softplus(v[4]-valid)
    if label == "E":
        return feasibility  # Completion is undefined for an invalid start.
    completed = torch.logsumexp(v[:2], dim=0)
    incomplete = torch.logsumexp(v[2:4], dim=0)
    completion = F.softplus(incomplete-completed) if label in "AB" else F.softplus(completed-incomplete)
    return feasibility+completion


def run(args):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import LoraConfig, PeftModel, get_peft_model

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("training run exists; do not overwrite")
    read = lambda name: [json.loads(x) for x in (Path(args.data_dir) / name).read_text().splitlines()]
    train = read("train.jsonl")
    dev = [] if args.train_only else read("development.jsonl")
    data_names = ["train.jsonl"] if args.train_only else ["train.jsonl", "development.jsonl"]
    if {r["seed"] for r in train} & {r["seed"] for r in dev}:
        raise ValueError("initialization seeds overlap")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    updates, sampling_design = None, None
    replay = args.sampling == "outcome-replay"
    state_balanced=args.sampling=='outcome-state'
    if args.strict_timing_coverage and not state_balanced:
        raise ValueError('strict timing coverage requires outcome-state sampling')
    if args.risk_quota_replay and (not state_balanced or not args.strict_timing_coverage or
            not args.hard_reference or args.admission_threshold is None):
        raise ValueError('risk quota replay needs strict outcome-state sampling, a training reference and the frozen threshold')
    family_training = args.sampling == 'outcome-family'
    isolated = args.sampling in ("outcome-tiny", "outcome-full", "outcome-replay",'outcome-state','outcome-family')
    process_weight = getattr(args, 'process_loss_weight', 0.)
    shared_family=args.temporal_mode=='family_stream'
    if shared_family and not family_training:raise ValueError('family_stream requires outcome-family sampling')
    if args.family_boundary=='set' and not shared_family:raise ValueError('set boundary requires shared family readouts')
    if process_weight < 0 or process_weight and (not isolated or args.temporal_mode not in ('stage_joint','family_stream')):
        raise ValueError('process marginal training requires nonnegative weight and isolated stage_joint outcomes')
    if family_training and args.temporal_mode not in ('stage_joint','family_stream'):
        raise ValueError('duration-family training uses the optional stage_joint input mode')
    support_time_weight = getattr(args, 'support_time_loss_weight', 0.)
    if support_time_weight < 0 or support_time_weight and (not isolated or args.temporal_mode != 'stage_late'):
        raise ValueError('support-time auxiliary training needs nonnegative weight and isolated stage_late outcomes')
    if support_time_weight and any('support_time' not in r for r in train):
        raise ValueError('support-time training targets are missing')
    tiny = args.sampling == "outcome-tiny"
    factorized = args.sampling == "balanced" or isolated
    if args.admission_loss_weight and (not isolated or args.admission_threshold is None):
        raise ValueError('admission boundary training requires isolated outcomes and a frozen threshold')
    if args.sampling in ("targeted", "balanced"):
        if not args.hard_reference:
            raise ValueError("targeted sampling needs the previous training predictions")
        previous = [json.loads(x) for x in Path(args.hard_reference).read_text().splitlines()]
        sampler = balanced_updates if factorized else targeted_updates
        updates, sampling_design = sampler(train, previous, args.steps, args.seed)
    if isolated:
        if not args.init_adapter or not args.start_adapter:
            raise ValueError("isolated outcomes require initial outcome and frozen start adapters")
        if family_training:
            from .vlm_duration_temporal import family_updates
            updates, sampling_design = family_updates(train, args.steps, args.seed,balanced_replay=args.family_replay_balanced,
                stratify_safe_sets=args.family_stratify_safe_sets)
        elif state_balanced:
            from .vlm_duration_temporal import state_updates
            updates,sampling_design=state_updates(train,args.steps,args.seed,strict_timing=args.strict_timing_coverage)
            if args.risk_quota_replay:
                from .vlm_duration_temporal import risk_quota_updates
                previous=[json.loads(line) for line in Path(args.hard_reference).read_text().splitlines()]
                updates,quota_design=risk_quota_updates(train,previous,updates,args.seed,args.admission_threshold)
                sampling_design.update(quota_design)
        elif replay:
            if not args.hard_reference:
                raise ValueError("risk replay requires the preceding training predictions")
            previous = [json.loads(x) for x in Path(args.hard_reference).read_text().splitlines()]
            updates, sampling_design = risk_replay_updates(train, previous, args.steps, args.seed,args.risk_threshold)
        else:
            updates, sampling_design = outcome_updates(train, args.steps, args.seed, tiny=tiny)
    if updates is not None:
        (output / "sampling-plan.jsonl").write_text("".join(json.dumps(x)+"\n" for x in updates))
    auxiliary_weight = .5 if factorized or args.sampling == "targeted" else 0.
    config = {"model_path": args.model_path, "data_dir": str(Path(args.data_dir).resolve()),
              "purpose": "fixed-budget fit and candidate-ranking diagnostic; no policy deployment or held-out test",
              "steps": args.steps, "rank": 16, "alpha": 32, "learning_rate": args.learning_rate,
              "pair_weight": .1, "pair_probability": .5, "gradient_checkpointing": True,
              "dtype": "bfloat16", "seed": args.seed, "input_fields": ["image", "robot", "plan"],
              "metadata_excluded_from_prompt": ["seed", "scenario", "label"],
              "train_examples": len(train), "development_examples": len(dev),
              "train_class_counts": dict(Counter(r["label"] for r in train)),
              "train_seeds": sorted({r["seed"] for r in train}), "development_seeds": sorted({r["seed"] for r in dev}),
              "gpu": torch.cuda.get_device_name(0), "initial_memory_free_total": list(torch.cuda.mem_get_info(0)),
              "torch": torch.__version__, "instruction": INSTRUCTION,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "early_stopping": False, "checkpoint_selection": "final optimizer step only",
              "evaluation_splits": ["train"] if args.train_only else ["train", "development"],
              "data_sha256": {name: hashlib.sha256((Path(args.data_dir)/name).read_bytes()).hexdigest()
                              for name in data_names},
              "sampling": args.sampling, "sampling_design": sampling_design,
              "temporal_mode":args.temporal_mode,
              "family_boundary":args.family_boundary,
              "strict_timing_coverage":args.strict_timing_coverage,
              "risk_quota_replay":args.risk_quota_replay,
              "admission_loss_weight":args.admission_loss_weight, "admission_logit_margin":args.admission_margin,
              "train_only":args.train_only, "fixed_admission_threshold":args.admission_threshold,
              "final_checkpoint_evaluation_only":args.final_eval_only,
              "rank_cross_wait_pairs":args.temporal_mode=='paired',
              "auxiliary_task_weight": auxiliary_weight,
              "auxiliary_tasks": ["start_feasibility_E_vs_ABCD", "completion_AB_vs_CD_for_valid_starts"] if auxiliary_weight else [],
              "hard_reference": str(Path(args.hard_reference).resolve()) if args.hard_reference else None,
              "hard_reference_sha256": hashlib.sha256(Path(args.hard_reference).read_bytes()).hexdigest() if args.hard_reference else None}
    if isolated:
        config.update(init_adapter=str(Path(args.init_adapter).resolve()),
                      frozen_start_adapter=str(Path(args.start_adapter).resolve()),
                      pair_probability=None,
                      evaluation_splits=["tiny_train"] if tiny else (["train"] if args.train_only else ["train", "development"]),
                      tiny_gate={"minimum_correct": 15, "examples": 16, "both_B_A_pairs_correct": True,
                                 "C_D_false_safe": 0} if tiny else None)
    if factorized:
        config.update(start_instruction=START_INSTRUCTION, outcome_instruction=OUTCOME_INSTRUCTION,
                      prediction_rule="local: q(start valid)*P(A-D|start valid), E=1-q; other plans: P(A-D)",
                      start_input_fields=["image", "robot", "repair_arm", "keeper_arm"],
                      auxiliary_tasks=["completion_AB_vs_CD_for_conditional_outcomes"])
    if args.temporal_mode=='paired':
        from .vlm_duration_temporal import TEMPORAL_INSTRUCTION
        config.update(outcome_instruction=TEMPORAL_INSTRUCTION,
            input_fields=['before_image','image','before_robot','robot','wait_steps','plan'],
            outcome_rgb_frames=8,start_rgb_frames=4)
    if args.temporal_mode in ('stage_early','stage_late','stage_joint'):
        config.update(outcome_rgb_frames=8,start_rgb_frames=4,
            stage_observation=args.temporal_mode,whole_episode_guards=True,
            input_fields=(['before_image','image','before_robot','robot','early_history','plan']
                if args.temporal_mode=='stage_early' else
                ['image','after_prefix_image','robot','after_prefix_robot','late_history','plan']))
    if args.temporal_mode in ('stage_joint','family_stream'):
        config.update(process_loss_weight=process_weight,
            process_prediction_rule='P(violation)=P(B)+P(D); completion=P(A)+P(B); admission=q(start)*P(A)',
            nominal_features='Derived only from h1/h2 robot observations and the known candidate plan; no future states or contact truth',
            family_pairing='Average actual A/non-A joint flips and A/C versus B/D process flips; no A/A ordering',
            support_time_inference=False)
    if shared_family:
        from .vlm_duration_family import INSTRUCTION as FAMILY_INSTRUCTION
        config.update(outcome_rgb_frames=4,start_rgb_frames=4,
            outcome_instruction=FAMILY_INSTRUCTION,
            input_fields=['before_image','before_robot','robot','entry_seconds_after_h0','lookahead_robot',
                'lookahead_previous_robot','lookahead_delta_seconds','remaining_close_steps','plan'],
            prerequisite_phase='h0 only; never reused as current h1/h2 grasp evidence',
            family_readouts=[60,40,30],candidate_supervision_budget=sampling_design['presentations'],
            readout_slots_per_optimizer_step=3,
            language_forward_calls_per_optimizer_step=2,
            checkpoint_scope='stateless linear/norm/MLP modules; KV updates outside checkpoints; full prefix gradients',
            set_boundary_rule='all non-A and cheapest executed A; other A retain CE' if args.family_boundary=='set' else 'per candidate')
    if support_time_weight:
        from .vlm_duration_support_time import SUPPORT_TIME_CODES, SUPPORT_TIME_INSTRUCTION
        config.update(support_time_loss_weight=support_time_weight,
            support_time_instruction=SUPPORT_TIME_INSTRUCTION, support_time_codes=SUPPORT_TIME_CODES,
            support_time_train_counts=dict(Counter(r['support_time']['label'] for r in train)),
            support_time_inference=False, support_time_forward='Separate prompt without any true outcome or event in its input',
            auxiliary_rng='Fork/restore Torch RNG; primary dropout sequence and sampling stay unchanged',
            budget_note='Same optimizer steps and primary presentations; additional auxiliary forwards/backwards cost extra compute')
        config['metadata_excluded_from_prompt'].append('support_time')
    (output / "protocol.json").write_text(json.dumps(config, indent=2)+"\n")
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    tokens = [processor.tokenizer.encode(c, add_special_tokens=False) for c in "ABCDE"]
    if any(len(x) != 1 for x in tokens):
        raise ValueError("class answers must be single tokens")
    ids = [x[0] for x in tokens]
    support_time_ids = []
    if support_time_weight:
        time_tokens = [processor.tokenizer.encode(c, add_special_tokens=False) for c in SUPPORT_TIME_CODES]
        if any(len(x) != 1 for x in time_tokens):
            raise ValueError('support-time answers must be single native tokens')
        support_time_ids = [x[0] for x in time_tokens]
    started = time.monotonic()
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True, device_map={"": 0})
    model.config.use_cache = False
    if isolated:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        model.load_adapter(args.start_adapter, adapter_name="start", is_trainable=False)
        model.set_adapter("default")
    else:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none", task_type="CAUSAL_LM"))
    if shared_family:model.gradient_checkpointing_disable()
    else:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if shared_family:
        from .vlm_duration_family import checkpoint_stateless_parts
        checkpoint_stateless_parts(model)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if any("visual" in n for n, p in trainable):
        raise ValueError("vision trunk unexpectedly trainable")
    if isolated and any(".start." in n for n, p in trainable):
        raise ValueError("start adapter unexpectedly trainable")
    frozen_start = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                    if isolated and ".start." in n}
    config["trainable_parameters"] = sum(p.numel() for n, p in trainable)
    config["model_load_seconds"] = time.monotonic()-started
    (output / "protocol.json").write_text(json.dumps(config, indent=2)+"\n")
    print(json.dumps({"loaded": True, "trainable_parameters": config["trainable_parameters"],
                      "model_load_seconds": config["model_load_seconds"]}), flush=True)

    def inputs(example, task="joint"):
        if args.temporal_mode!='legacy':
            from .vlm_duration_temporal import input_spec
            paths,prompt=input_spec(example,task,args.temporal_mode)
        else:paths,prompt=[example['image']],task_prompt_for(example,task)
        images=[]
        for path in paths:
            with np.load(path) as archive:images.extend(Image.fromarray(x.copy()) for x in archive['rgb'])
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        text = processor.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)
        return processor(text=[text], images=images, return_tensors="pt").to("cuda:0")

    family_eval_cache={}
    def prerequisite_key(example):
        if shared_family:
            from .vlm_duration_family import entry_example
            return start_key(entry_example(example))
        return start_key(example)

    def family_logits(example):
        from .vlm_duration_family import stream_logits
        if model.active_adapter!='default':model.set_adapter('default',inference_mode=not model.training)
        return stream_logits(model,processor,example)

    def logits(example, task="joint"):
        if isolated:
            adapter = "start" if task == "start" else "default"
            if model.active_adapter != adapter:
                model.set_adapter(adapter, inference_mode=task == "start")
        if shared_family and task!='start':
            from .vlm_duration_family import family_key
            key=family_key(example)
            if key not in family_eval_cache:
                family_eval_cache.clear();family_eval_cache[key]=family_logits(example)
            return family_eval_cache[key][example['plan']['local_relevel_steps']]
        return model(**inputs(example, task), logits_to_keep=1, use_cache=False).logits[0, -1].float()

    def evaluate(name, examples=None):
        model.eval()
        family_eval_cache.clear()
        records, starts = [], {}
        with torch.inference_mode():
            for example in (dev if examples is None else examples):
                v = logits(example, "outcome" if factorized else "joint")
                extra = {}
                if factorized:
                    conditional = torch.softmax(v[ids[:4]], dim=0).cpu().tolist()
                    q = 1.
                    if example["plan"]["kind"] == "local":
                        key = prerequisite_key(example)
                        if key not in starts:
                            u = logits(example, "start")
                            s = torch.softmax(u[[ids[0], ids[4]]], dim=0).cpu().tolist()
                            starts[key] = {"seed": example["seed"], "scenario": example["scenario"],
                                           "repair_arm": example["plan"]["repair_arm"],
                                           "keeper_arm": example["plan"]["keeper_arm"],
                                           "label": task_label(example, "start"),
                                           "prediction": "A" if s[0] >= s[1] else "E", "p_valid": s[0]}
                        q = starts[key]["p_valid"]
                    p = [q*x for x in conditional]+[1-q]
                    extra = {"start_valid_score": q, "conditional_probabilities": conditional,
                             "conditional_prediction": "ABCD"[int(np.argmax(conditional))]}
                    mass = float(torch.softmax(v, dim=0)[ids[:4]].sum())
                else:
                    p = torch.softmax(v[ids], dim=0).cpu().tolist()
                    mass = float(torch.softmax(v, dim=0)[ids].sum())
                prediction = "ABCDE"[int(np.argmax(p))]
                records.append({"seed": example["seed"], "scenario": example["scenario"],
                                "image":example['image'],"robot":example['robot'],
                                "plan": example["plan"], "label": example["label"],
                                "prediction": prediction, "class_probabilities": p, "class_token_mass": mass, **extra})
                if 'before_image' in example:
                    records[-1].update(before_image=example['before_image'],wait_steps=example['wait_steps'],source=example['source'])
                if 'after_prefix_image' in example:
                    records[-1].update(after_prefix_image=example['after_prefix_image'],
                        post_prefix_wait_steps=example['post_prefix_wait_steps'])
                if shared_family:records[-1].update(prerequisite_phase='h0',before_robot=example['before_robot'])
        report = prediction_metrics(records)
        if args.temporal_mode in ('stage_joint','family_stream'):
            report['joint_marginals'] = joint_marginal_metrics(records)
        if args.admission_threshold is not None:
            report['fixed_admission'] = fixed_admission_metrics(records, args.admission_threshold)
        if isolated:
            model.set_adapter("default")
        if tiny:
            report["tiny_fit_gate"] = tiny_fit_gate(records)
        if factorized:
            start_rows = list(starts.values())
            report["start_feasibility"] = {"unique_roles": len(start_rows),
                "accuracy": sum(r["label"] == r["prediction"] for r in start_rows)/len(start_rows),
                "valid_roles": sum(r["label"] == "A" for r in start_rows),
                "invalid_roles": sum(r["label"] == "E" for r in start_rows),
                "false_invalid": sum(r["label"] == "A" and r["prediction"] == "E" for r in start_rows),
                "false_valid": sum(r["label"] == "E" and r["prediction"] == "A" for r in start_rows),
                "note": "One prediction per image/robot/roles, shared by all durations; no true label is used for inference."}
            (output/(name+"-start-predictions.jsonl")).write_text("".join(json.dumps(r)+"\n" for r in start_rows))
        (output / (name+".json")).write_text(json.dumps(report, indent=2)+"\n")
        (output / (name+"-predictions.jsonl")).write_text("".join(json.dumps(r)+"\n" for r in records))
        print(json.dumps({"evaluation": name, **{k: report[k] for k in
              ("examples", "accuracy", "false_safe_accepts", "safe_precision", "safe_recall", "safe_probability_auroc")}}), flush=True)
        return report

    def evaluate_support_time(name, examples):
        from .vlm_duration_support_time import support_time_metrics
        model.eval()
        records = []
        with torch.inference_mode():
            for example in examples:
                v = logits(example, 'support_time')
                p = torch.softmax(v[support_time_ids], dim=0).cpu().tolist()
                records.append({'seed': example['seed'], 'scenario': example['scenario'],
                    'post_prefix_wait_steps': example['post_prefix_wait_steps'],
                    'local_relevel_steps': example['plan']['local_relevel_steps'],
                    'label': task_label(example, 'support_time'),
                    'prediction': SUPPORT_TIME_CODES[int(np.argmax(p))], 'class_probabilities': p,
                    'target': example['support_time']})
        report = support_time_metrics(records)
        (output / (name + '-support-time.json')).write_text(json.dumps(report, indent=2) + '\n')
        (output / (name + '-support-time-predictions.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in records))
        return report

    fit_train = [train[i] for i in sampling_design["cohort_indices"]] if tiny else train
    frozen_start_scores = {}
    if args.final_eval_only:
        # The boundary loss still needs the unchanged prerequisite probability.
        # Cache only this frozen head on TRAIN observations; do not evaluate
        # initial outcome predictions or development data before training.
        if args.admission_loss_weight:
            model.eval()
            with torch.inference_mode():
                for example in fit_train:
                    key = prerequisite_key(example)
                    if key not in frozen_start_scores:
                        v = logits(example, 'start')
                        frozen_start_scores[key] = float(torch.softmax(v[[ids[0], ids[4]]], dim=0)[0])
            model.set_adapter('default')
            (output/'frozen-start-scores.json').write_text(json.dumps(frozen_start_scores, indent=2)+'\n')
    else:
        evaluate("train-before", fit_train)
        if args.admission_loss_weight:
            before_rows = [json.loads(line) for line in (output/'train-before-predictions.jsonl').read_text().splitlines()]
            frozen_start_scores = {prerequisite_key(r): r['start_valid_score'] for r in before_rows}
    if not tiny and not replay and not args.train_only and not args.final_eval_only:
        evaluate("before")
    grouped = {}
    for r in train:
        grouped.setdefault((r["seed"], r["scenario"]), []).append(r)
    pairs = [(a, b) for group in grouped.values() for a in group if a["label"] == "A"
             for b in group if b["label"] != "A"]
    by_class = {c: [r for r in train if r["label"] == c] for c in "ABCDE"}
    present = [c for c in by_class if by_class[c]]
    optimizer = torch.optim.AdamW([p for n, p in trainable], lr=args.learning_rate, weight_decay=.01)
    training_started = time.monotonic()
    with (output / "training.jsonl").open("w") as stream:
        for step in range(args.steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            t = time.monotonic()
            if updates is None:
                paired = bool(pairs and random.random() < .5)
                examples = random.choice(pairs) if paired else [random.choice(by_class[random.choice(present)])]
                sample_kind, indices, task = "baseline", None, "joint"
            else:
                update = updates[step]
                indices, sample_kind = update["indices"], update["kind"]
                examples = [train[i] for i in indices]
                paired = len(examples) == 2
                task = update.get("task", "joint")
            labels = [task_label(r, task) for r in examples]
            if shared_family:
                shared_values=family_logits(examples[0])
                values=[shared_values[r['plan']['local_relevel_steps']] for r in examples]
            else:values = [logits(r, task) for r in examples]
            ce_loss = sum(F.cross_entropy(v[None], torch.tensor([ids["ABCDE".index(label)]], device=v.device))
                          for v, label in zip(values, labels))/len(values)
            aux_loss = ce_loss.new_zeros(())
            if auxiliary_weight and task != "start":
                auxiliary = conditional_completion_loss if factorized else auxiliary_task_loss
                aux_loss = sum(auxiliary(v, ids, label) for v, label in zip(values, labels))/len(values)
            process_loss = ce_loss.new_zeros(())
            if process_weight:
                process_loss = sum(conditional_process_loss(v, ids, label) for v, label in zip(values, labels))/len(values)
            pair_loss = ce_loss.new_zeros(())
            family_pairs = {'joint': 0, 'process': 0}
            ranked_pair = family_training and len(examples) == 3 or paired and (not factorized or sample_kind in ("strict_timing", "new_B_pair", "process_pair", "C_pair",'action_contrast')
                                     or sample_kind=='wait_contrast' and args.temporal_mode=='paired')
            if family_training and ranked_pair:
                pair_loss, family_pairs = family_pair_loss(values, ids, labels)
            elif ranked_pair:
                negative_ids = ids[1:4] if factorized else ids[1:]
                scores = [v[ids[0]] - torch.logsumexp(v[negative_ids], dim=0) for v in values]
                pair_loss = F.softplus(-(scores[0]-scores[1]))
            boundary_loss = ce_loss.new_zeros(())
            if args.admission_loss_weight:
                if shared_family and args.family_boundary=='set' and len(examples)==3:
                    boundary_loss=family_set_admission_loss(values,ids,labels,
                        [frozen_start_scores[prerequisite_key(r)] for r in examples],
                        [r['plan'] for r in examples],args.admission_threshold,args.admission_margin)
                elif not shared_family or args.family_boundary!='set':
                    boundary_loss = sum(admission_boundary_loss(v, ids, label, frozen_start_scores[prerequisite_key(r)],
                    args.admission_threshold, args.admission_margin)
                    for v, label, r in zip(values, labels, examples))/len(values)
            loss = ce_loss+auxiliary_weight*aux_loss+process_weight*process_loss+.1*pair_loss+args.admission_loss_weight*boundary_loss
            loss.backward()
            time_loss_value = 0.
            if support_time_weight:
                # Independent prompt, so a true A-D answer is never a conditioning input.
                # Backward each auxiliary example separately to retain the existing memory budget.
                with torch.random.fork_rng(devices=[0]):
                    for example in examples:
                        v_time = logits(example, 'support_time')
                        target_id = support_time_ids[SUPPORT_TIME_CODES.index(task_label(example, 'support_time'))]
                        time_loss = F.cross_entropy(v_time[None], torch.tensor([target_id], device=v_time.device))
                        (support_time_weight * time_loss / len(examples)).backward()
                        time_loss_value += float(time_loss.detach()) / len(examples)
                        del v_time, time_loss
            gradient_norm = torch.nn.utils.clip_grad_norm_([p for n, p in trainable], 1.)
            optimizer.step()
            torch.cuda.synchronize()
            record = {"step": step+1, "loss": float(loss.detach()) + support_time_weight * time_loss_value, "paired": paired,
                      "sample_kind": sample_kind, "example_indices": indices,
                      "labels": labels, "task": task, "ranked_pair": ranked_pair,
                      "classification_loss": float(ce_loss.detach()), "auxiliary_task_loss": float(aux_loss.detach()),
                      "pair_loss": float(pair_loss.detach()),
                      "admission_boundary_loss": float(boundary_loss.detach()),
                      "gradient_norm_before_clip": float(gradient_norm),
                      "seconds": time.monotonic()-t, "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated()}
            if support_time_weight:
                record.update(support_time_loss=time_loss_value,
                    support_time_labels=[task_label(r, 'support_time') for r in examples])
            if args.temporal_mode in ('stage_joint','family_stream'):
                record.update(process_marginal_loss=float(process_loss.detach()), family_pairs=family_pairs,
                    family_update=len(examples) == 3)
            if shared_family:record.update(outcome_forward_calls=2,readout_slots=3,
                prefix_gradients=True,first_prefix_gradient_norm=getattr(model,'family_prefix_gradient_norm',None),
                checkpoint_scope='stateless modules; full prefix KV gradient')
            stream.write(json.dumps(record)+"\n")
            stream.flush()
            print(json.dumps(record), flush=True)
            del values, loss, ce_loss, aux_loss, pair_loss, boundary_loss, process_loss
            if shared_family:del shared_values
    checkpoint = output / "adapter"
    if isolated:
        # Keep newly saved adapters loadable when the frozen base cache moved.
        model.peft_config['default'].base_model_name_or_path=str(Path(args.model_path).resolve())
    model.save_pretrained(checkpoint, selected_adapters=["default"] if isolated else None)
    training_seconds = time.monotonic()-training_started
    training_after = evaluate("train-after", fit_train)
    after = None if tiny or args.train_only else evaluate("after")
    support_time_reports = None
    if support_time_weight:
        support_time_reports = {'train': evaluate_support_time('train-after', fit_train)}
        if not tiny and not args.train_only:
            support_time_reports['development'] = evaluate_support_time('after', dev)
    start_unchanged = all(torch.equal(p.detach().cpu(), frozen_start[n]) for n, p in model.named_parameters() if n in frozen_start)
    if isolated and not start_unchanged:
        raise ValueError("frozen start adapter changed during outcome training")
    result = {"completed_steps": args.steps, "adapter": str(checkpoint.resolve()),
              "training_seconds": training_seconds,
              "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
              "training_fit": training_after, "development": after,
              "frozen_start_unchanged": start_unchanged if isolated else None,
              "tiny_fit_gate": training_after.get("tiny_fit_gate"),
              "data_unchanged": all(hashlib.sha256((Path(args.data_dir)/name).read_bytes()).hexdigest() == digest
                                    for name, digest in config["data_sha256"].items()),
              "admission": "not evaluated; no closed-loop deployment in this readiness run"}
    if support_time_weight:
        result['support_time'] = support_time_reports
    (output / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--sampling", choices=("baseline", "targeted", "balanced", "outcome-tiny", "outcome-full", "outcome-replay",'outcome-state','outcome-family'), default="baseline")
    parser.add_argument('--temporal-mode',choices=('legacy','static','paired','stage_early','stage_late','stage_joint','family_stream'),default='legacy')
    parser.add_argument("--hard-reference")
    parser.add_argument("--risk-threshold",type=float,help='Frozen working point for training-only hard-example replay')
    parser.add_argument("--init-adapter")
    parser.add_argument("--start-adapter")
    parser.add_argument('--train-only', action='store_true', help='Do not read or evaluate any development examples')
    parser.add_argument('--admission-threshold', type=float, help='Frozen threshold for reporting; not optimized')
    parser.add_argument('--strict-timing-coverage', action='store_true', help='Cycle all observed strict 40/60 flips in existing action-pair slots')
    parser.add_argument('--admission-loss-weight', type=float, default=0., help='Fixed weight for the optional joint-score boundary loss')
    parser.add_argument('--admission-margin', type=float, default=1., help='Fixed margin in joint A log-odds')
    parser.add_argument('--final-eval-only', action='store_true', help='Evaluate outcomes only after the final optimizer step; cache the frozen start head on train if needed')
    parser.add_argument('--risk-quota-replay', action='store_true', help='Replace singles with fixed training-only risk quotas while preserving all strict/wait/control pairs')
    parser.add_argument('--support-time-loss-weight', type=float, default=0., help='Training-only first future keeper-grasp loss time-bin supervision; keep native A-D inference unchanged')
    parser.add_argument('--process-loss-weight', type=float, default=0., help='Process A/C versus B/D loss on the same native A-D logits used for admission')
    parser.add_argument('--family-replay-balanced', action='store_true', help='Equal old/new family and single presentations using training-only replay_source metadata')
    parser.add_argument('--family-stratify-safe-sets',action='store_true',help='Cycle observed safety-set patterns within each old/new pool')
    parser.add_argument('--family-boundary',choices=('per-candidate','set'),default='per-candidate')
    run(parser.parse_args())


if __name__ == "__main__":
    main()
