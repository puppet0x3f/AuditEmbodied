from auditembodied.vlm_duration_data import outcome, plan
from auditembodied.vlm_duration_lora import (auxiliary_task_loss, prediction_metrics, prompt_for,
                                           balanced_updates, conditional_completion_loss, start_key,
                                           outcome_updates, start_prompt_for, targeted_updates, task_prompt_for,
                                           risk_replay_updates, timing_contrasts, tiny_fit_gate, fixed_admission_metrics,
                                           admission_boundary_loss)


def test_joint_outcome_keeps_completion_and_process_safety_distinct():
    row = {"candidate": "R_left", "post_fault": {"holds": [False, True]}}
    for success, violation, expected in [(True, False, "A"), (True, True, "B"),
                                         (False, False, "C"), (False, True, "D")]:
        row.update(success=success, protection_violation=violation)
        assert outcome(row) == expected
    row["post_fault"]["holds"][1] = False
    assert outcome(row) == "E"


def test_prompt_uses_plan_timing_without_seed_fault_or_outcome_metadata():
    candidate = plan({"candidate": "R_right", "local_steps": {"local_relevel_steps": 60}})
    assert candidate["repair_arm"] == "right" and candidate["keeper_arm"] == "left"
    assert candidate["local_relevel_steps"] == 60
    text = prompt_for({"robot": {"positions": [[0, 0, 1], [0, 1, 1]]}, "plan": candidate,
                       "seed": "PRIVATE_SEED", "scenario": "PRIVATE_FAULT", "label": "PRIVATE_OUTCOME"})
    assert "60" in text
    assert all(v not in text for v in ("PRIVATE_SEED", "PRIVATE_FAULT", "PRIVATE_OUTCOME"))


def predicted(relevel, label, score, approach=16):
    p = plan({"candidate": "R_left", "local_steps": {"local_relevel_steps": relevel, "local_approach_steps": approach}})
    return {"seed": 5, "scenario": "left_loss", "plan": p, "label": label,
            "prediction": "A" if score > .5 else "B", "class_probabilities": [score, 1-score, 0., 0., 0.],
            "class_token_mass": 1.}


def test_40_60_contrasts_require_every_other_plan_field_to_match():
    rows = [predicted(40, "B", .1), predicted(60, "A", .9), predicted(60, "C", .2, approach=24)]
    result = timing_contrasts(rows)
    assert result["matched_pairs"] == result["changed_safety_pairs"] == 1
    assert result["changed_pair_both_safety_correct"] == result["changed_pair_order_correct"] == 1
    assert result["cases"][0]["p_safe_delta_60_minus_40"] == .8


def test_outcome_accuracy_does_not_hide_failed_timing_ordering():
    result = prediction_metrics([predicted(40, "B", .7), predicted(60, "A", .6)])
    assert result["false_safe_accepts"] == 1
    assert result["safe_probability_auroc"] == 0.
    assert result["within_state_local_ranking"]["correct"] == 0
    assert result["within_state_local_A_vs_B"]["pairs"] == 1
    assert result["timing_40_60"]["changed_pair_both_safety_correct"] == 0


def test_fixed_admission_distinguishes_correct_classes_from_usable_coverage():
    a = {**predicted(60, 'A', .99), 'image': 'recoverable.npz', 'robot': {}}
    b = {**predicted(40, 'B', .01), 'image': 'no-safe.npz', 'robot': {}}
    result = fixed_admission_metrics([a, b], .9996)
    assert result['native_A_vs_non_A_correct'] == result['native_correct'] == 2
    assert result['safe_admitted'] == 0 and result['non_A_rejected'] == 1
    assert result['no_safe_families_rejected'] == result['no_safe_families'] == 1
    assert not result['passed']
    a['class_probabilities'][0] = .9998
    assert fixed_admission_metrics([a, b], .9996)['passed']


def test_admission_boundary_loss_uses_joint_score_and_retains_high_confidence_gradients():
    import torch
    for label, direction in [('A', -1), ('B', 1), ('C', 1), ('D', 1)]:
        logits = torch.tensor([8., 0., 0., 0., 20.], requires_grad=True)
        loss = admission_boundary_loss(logits, list(range(5)), label, .99999, .999667229515218)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
        assert direction * logits.grad[0] > 0 and logits.grad[4] == 0
    logits = torch.tensor([100., 0., 0., 0., 0.], requires_grad=True)
    loss = admission_boundary_loss(logits, list(range(5)), 'B', 1., .999667229515218)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all() and logits.grad[0] > .9


def test_targeted_mix_keeps_budget_and_strict_observation_matching():
    def sample(label, steps=40, kind="local"):
        p = plan({"candidate": "R_left", "local_steps": {"local_relevel_steps": steps}})
        p["kind"] = kind
        return {"seed": 5, "scenario": "left_loss", "plan": p, "image": "same.npz", "robot": {}, "label": label}
    rows = [sample("B"), sample("A", 60), sample("C", 30), sample("E", 40, "wrong_side"), sample("D", 20)]
    previous = [{**r, "prediction": "A"} for r in rows]
    updates, design = targeted_updates(rows, previous, 512, 17)
    assert len(updates) == 512 and design["presentations"] == 768
    assert design["update_counts"]["strict_timing"] == 128
    assert sum(len(r["indices"]) == 2 for r in updates) == 256
    assert all(r["indices"] == [1, 0] for r in updates if r["kind"] == "strict_timing")
    assert all(r["labels"] == ["E"] for r in updates if r["kind"] == "hard_E")
    assert all(r["labels"] == ["C"] for r in updates if r["kind"] == "hard_C")


def test_auxiliary_losses_push_false_acceptance_down_without_new_head():
    import torch
    for label in ("C", "E"):
        logits = torch.tensor([3., 0., 0., 0., 0.], requires_grad=True)
        loss = auxiliary_task_loss(logits, list(range(5)), label)
        loss.backward()
        assert logits.grad[0] > 0
        assert logits.grad[2 if label == "C" else 4] < 0


def test_start_question_is_identical_across_future_plan_durations():
    a = {**predicted(40, "B", .1), "image": "observation.npz", "robot": {"fingers": [[.04, -.04], [.01, -.01]]}}
    b = {**a, "plan": {**a["plan"], "local_relevel_steps": 60, "local_close_steps": 99,
                       "instruction": "PRIVATE_FUTURE_PLAN"}}
    assert start_key(a) == start_key(b)
    assert start_prompt_for(a) == start_prompt_for(b)
    assert "PRIVATE_FUTURE_PLAN" not in start_prompt_for(b)
    assert "required keeper grasp absent at the start" not in task_prompt_for(a, "outcome")
    assert "Answer exactly one letter A, B, C or D." in task_prompt_for(a, "outcome")


def test_balanced_sampler_restores_observed_safe_controls_without_invalid_outcome_labels():
    def sample(seed, label, steps, wrong=False):
        p = plan({"candidate": "R_right" if wrong else "R_left",
                  "local_steps": {"local_relevel_steps": steps}})
        return {"seed": seed, "scenario": "left_loss", "plan": p, "image": f"{seed}.npz", "robot": {}, "label": label}
    rows = [sample(5, "B", 40), sample(5, "A", 60), sample(5, "D", 30), sample(5, "E", 40, True),
            sample(6, "A", 40), sample(6, "A", 30), sample(6, "C", 60), sample(6, "E", 40, True)]
    previous = [{**r, "prediction": "E"} for r in rows]
    updates, design = balanced_updates(rows, previous, 512, 17)
    assert len(updates) == 512 and design["presentations"] == 768
    assert design["update_counts"]["strict_timing"] == 64
    assert design["update_counts"]["safe_control"] == 64
    assert design["update_counts"]["start_roles"] == 128
    assert all(r["labels"] == ["A", "B"] for r in updates if r["kind"] == "strict_timing")
    assert all(r["labels"] == ["A", "A"] for r in updates if r["kind"] == "safe_control")
    assert all(r["labels"] == ["A", "E"] for r in updates if r["task"] == "start")
    assert all("E" not in r["labels"] for r in updates if r["task"] == "outcome")


def test_conditional_completion_loss_separates_incomplete_from_complete_without_start_logit():
    import torch
    logits = torch.tensor([0., 3., 0., 0., 8.], requires_grad=True)
    conditional_completion_loss(logits, list(range(5)), "D").backward()
    assert logits.grad[1] > 0 and logits.grad[3] < 0
    assert logits.grad[4] == 0


def tiny_outcome_fixture():
    rows = []
    def add(seed, scenario, arm, label, relevel, approach=16, close=22):
        p = plan({"candidate": arm, "local_steps": {"local_relevel_steps": relevel,
                  "local_approach_steps": approach, "local_close_steps": close}})
        rows.append({"seed": seed, "scenario": scenario, "image": f"{seed}-{scenario}.npz",
                     "robot": {}, "plan": p, "label": label})
    for scenario, arm in [("left_loss", "R_left"), ("right_loss", "R_right")]:
        for label, steps, approach, close in [("B", 40, 16, 22), ("A", 60, 16, 22),
                                              ("D", 30, 12, 17), ("D", 30, 16, 22)]:
            add(5, scenario, arm, label, steps, approach, close)
    for seed in range(4):
        add(seed, "left_loss", "R_left", "A", 40)
        add(seed, "left_loss", "R_left", "C", 60, 24, 33)
    return rows


def test_outcome_only_cycles_cover_tiny_cohort_without_start_updates():
    from collections import Counter
    rows = tiny_outcome_fixture()
    updates, design = outcome_updates(rows, 256, 17, tiny=True)
    assert design["cohort_examples"] == 16 and len(updates) == 256
    assert design["cohort_classes"] == {"A": 6, "B": 2, "C": 4, "D": 4}
    assert all(r["task"] == "outcome" for r in updates)
    assert all(r["labels"] == ["A", "B"] for r in updates if r["kind"] == "strict_timing")
    counts = Counter(i for r in updates for i in r["indices"])
    assert len(counts) == 16 and max(counts.values())-min(counts.values()) <= 1
    extra = {**rows[0], "label": "E"}
    _, full = outcome_updates(rows+[extra], 512, 17)
    assert full["cohort_indices"] == list(range(16))


def test_tiny_gate_requires_critical_classes_and_no_incomplete_false_accepts():
    rows = [{**r, "prediction": r["label"], "class_probabilities": [.9 if r["label"] == "A" else .1, .1, 0., 0., 0.]}
            for r in tiny_outcome_fixture()]
    assert tiny_fit_gate(rows)["passed"]
    bad_cd = [{**r, "prediction": "A" if i == 2 else r["prediction"]} for i, r in enumerate(rows)]
    assert tiny_fit_gate(bad_cd)["correct"] == 15 and not tiny_fit_gate(bad_cd)["passed"]
    bad_pair = [{**r, "prediction": "D" if i == 0 else r["prediction"]} for i, r in enumerate(rows)]
    assert tiny_fit_gate(bad_pair)["correct"] == 15 and not tiny_fit_gate(bad_pair)["passed"]


def test_risk_replay_uses_training_errors_and_keeps_observed_safe_controls():
    import pytest
    rows = tiny_outcome_fixture()
    rows.append({**rows[8], "plan": {**rows[8]["plan"], "local_relevel_steps": 60}})
    previous = [{**r, "prediction": "A" if i in [0,4,9,11] else "C" if i == 8 else r["label"]}
                for i, r in enumerate(rows)]
    updates, design = risk_replay_updates(rows, previous, 512, 17)
    assert len(updates) == 512 and design["presentations"] == 800
    assert design["update_counts"] == {"new_B_pair":96,"process_pair":32,"C_pair":64,
                                       "AA_pair":96,"hard_A":64,"background":160}
    assert design["hard_B_indices"] == [0,4] and design["hard_C_indices"] == [9,11]
    assert set(design["candidate_presentations"]) == set(range(len(rows)))
    for r in updates:
        assert r["task"] == "outcome" and "E" not in r["labels"]
        if len(r["indices"]) == 2:
            a,b = [rows[i] for i in r["indices"]]
            assert start_key(a) == start_key(b)
            assert r["labels"][0] == "A"
            assert r["labels"][1] == "A" if r["kind"] == "AA_pair" else r["labels"][1] in "BCD"
    mismatched = [{**previous[0], "seed":20}]+previous[1:]
    with pytest.raises(ValueError,match="training set"):
        risk_replay_updates(rows,mismatched,512,17)


def test_waiting_observation_is_not_paired_with_same_seed_before_waiting():
    rows=[{**predicted(40,'A',.9),'image':'before.npz'},
          {**predicted(60,'A',.9),'image':'before.npz'},
          {**predicted(40,'B',.1),'image':'after.npz'},
          {**predicted(60,'A',.9),'image':'after.npz'}]
    metrics=prediction_metrics(rows)
    assert metrics['timing_40_60']['matched_pairs']==2
    assert metrics['timing_40_60']['changed_safety_pairs']==1
    assert metrics['within_state_local_A_vs_B']['pairs']==1


def test_unrecoverable_observations_are_replayed_without_fabricated_safe_pairs():
    rows=tiny_outcome_fixture()
    rows.append({**rows[8],'plan':{**rows[8]['plan'],'local_relevel_steps':60}})
    rows.append({**rows[0],'image':'unrecoverable.npz','label':'B'})
    rows.append({**rows[0],'image':'unrecoverable.npz','label':'C',
                 'plan':{**rows[0]['plan'],'local_relevel_steps':60}})
    previous=[{**r,'prediction':'A' if r['label'] in 'BC' else r['label'],
               'class_probabilities':[.999 if r['label'] in 'BC' else .01,.001,0.,0.,0.]}
              for r in rows]
    updates,design=risk_replay_updates(rows,previous,512,17,threshold=.99)
    last={len(rows)-2,len(rows)-1}
    assert set(design['unpaired_risk_indices'])==last
    assert len(updates)==512 and design['update_counts']['background']==192
    assert set(design['candidate_presentations'])==set(range(len(rows)))
    assert all(len(u['indices'])==1 for u in updates if last.intersection(u['indices']))
