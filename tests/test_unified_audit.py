from auditembodied.unified_audit import aggregate, episode_audit, paired_cost


def lift_row(unsafe=False, incomplete_hold=False):
    frame = lambda ready, holds: {"phase": "final_hold", "stage": "final_stable_hold",
                                   "evaluator": {"ready": ready, "holds": holds, "table_contact": False}}
    trace = []
    if unsafe:
        trace += [{**frame(False, [False, False]), "phase": "recovery", "stage": "local_relevel"} for _ in range(3)]
    trace += [frame(not incomplete_hold or i == 19, [True, True]) for i in range(20)]
    return {"seed": 0, "scenario": "left_loss", "candidate": "R_left", "method": "proprio", "trace": trace,
            "success": not unsafe and not incomplete_hold, "task_complete": not incomplete_hold,
            "protection_violation": unsafe, "protection": {"max_unsupported_seconds": .15 if unsafe else 0.},
            "post_fault": {"holds": [False, True]}, "hold_seconds": 1., "failure_reason": None,
            "contact_anchor_count": 0, "state_edit_count_after_initialization": 0}


def test_terminal_success_is_separate_from_whole_trajectory_safety():
    row = episode_audit(lift_row(unsafe=True), "lift")
    assert row["terminal_goal_met"] and not row["safe_success"]
    assert row["terminal_met_without_safe_completion"]
    assert row["local_keeper_interrupted_after_valid_start"]
    assert row["unsupported_steps"] == 3
    summary = aggregate([row])
    assert summary["terminal_goal_met"] == 1 and summary["safe_success"] == 0


def test_last_ready_frame_does_not_replace_one_second_hold():
    row = episode_audit(lift_row(incomplete_hold=True), "lift")
    assert not row["terminal_goal_met"]


def test_paired_cost_uses_only_common_safe_scenes_and_seed_clusters():
    rows = []
    for seed in (0, 1):
        for scene in ("a", "b", "c"):
            for method in ("policy", "global"):
                failed = seed == 0 and scene == "a" and method == "policy"
                cost = 1. if method == "policy" else 2.
                rows.append({"seed": seed, "scenario": scene, "method": method, "safe_success": not failed,
                             "recovery_seconds": 0. if failed else cost, "recovery_path_m": cost / 10.})
    result = paired_cost(rows, "policy", "global", 0, draws=100)
    assert result["paired_safe_scenes"] == 5
    assert result["recovery_seconds"]["reduction_fraction"] == .5
    assert result["recovery_seconds"]["ci95"] == [.5, .5]
