from dataclasses import asdict
import copy

from auditembodied.duration_pilot import LOCAL_FIELDS, SCALES, analyze, metrics, retimed_config
from auditembodied.l4_lift import LiftConfig


def row(seed, scale, keeper=True, success=True, stage="pilot", candidate="R_left", start=True):
    r = dict(seed=seed, scenario="left_loss", candidate=candidate, time_scale=scale,
             experiment_stage=stage, local_steps={}, wall_seconds=0., success=success,
             protection_violation=not keeper, fault_class_valid=True,
             post_fault={"holds": [False, start], "supports": [False, start]},
             recovery_seconds=scale, recovery_path_m=.1, failure_reason=None,
             protection={"max_unsupported_seconds": 0., "deep_interarm_contact_steps": 0},
             blocked_calls=[], contact_anchor_count=0, state_edit_count_after_initialization=0,
             external_wrench_at_end=0.)
    r.update({k: {"time": 1., "qpos": [0., 1.]} for k in
              ("initialization_state", "preclose_state", "initialization_end_state", "prefix_state")})
    r["trace"] = [{"phase": "fault", "time": 1.},
                  {"phase": "recovery", "stage": "local_arc_close", "time": 1.05,
                   "evaluator": {"holds": [True, keeper], "supports": [True, keeper]}}]
    return r


def group(seed, flip=False, stage="pilot"):
    result = [row(seed, s, keeper=not (flip and s == 1.), stage=stage) for s in SCALES]
    result.append(row(seed, 1., stage=stage, candidate="R_global"))
    return result


def test_retiming_changes_only_local_step_counts():
    base = asdict(LiftConfig())
    fast = asdict(retimed_config(.75))
    assert {k for k in base if base[k] != fast[k]} == set(LOCAL_FIELDS)
    assert (fast["local_approach_steps"], fast["local_close_steps"], fast["local_relevel_steps"]) == (12, 17, 30)
    assert asdict(retimed_config(1.)) == base
    isolated = asdict(retimed_config(1.5, "relevel_only"))
    assert {k for k in base if base[k] != isolated[k]} == {"local_relevel_steps"}
    assert isolated["local_relevel_steps"] == 60


def test_missing_keeper_is_not_counted_as_new_loss():
    m = metrics(row(0, 1., keeper=False, start=False))
    assert not m["keeper_valid_at_start"]
    assert m["keeper_continuous"] is None
    assert m["first_keeper_loss_seconds"] is None


def test_signal_requires_a_safely_completed_variant_and_fresh_seed_confirmation():
    pilot = group(0, flip=True)
    assert analyze(pilot)["decision"] == "GO_CONFIRMATION"
    confirmed = analyze(pilot + group(20, flip=True, stage="confirmation"))
    assert confirmed["decision"] == "GO_LIMITED_LORA_PILOT"
    assert confirmed["usable_flip_seeds"] == [0, 20]
    for r in pilot:
        r["success"] = False
    assert analyze(pilot)["decision"] == "NO_GO_LORA_TIMING"


def test_pairing_checks_full_prefix_and_all_required_candidates():
    rows = group(0, flip=True)
    changed = copy.deepcopy(rows)
    changed[0]["trace"][0]["time"] = .9
    assert analyze(changed)["decision"] == "INVALID"
    assert analyze(rows[:-1])["decision"] == "INVALID"
    assert analyze(group(0))["decision"] == "NO_GO_LORA_TIMING"


def test_new_training_collection_requires_three_matched_physical_prefixes():
    from auditembodied.duration_acquire import analyze as acquisition_metrics
    rows = [row(8, scale, keeper=scale == 1.5) for scale in SCALES]
    for r, steps in zip(rows, (30, 40, 60)):
        r["local_steps"] = {"local_approach_steps": 16, "local_close_steps": 22, "local_relevel_steps": steps}
        r["trace"] = [{"phase": "fault", "time": 1., "stage": "fault_withdraw"},
                      {"phase": "recovery", "time": 1.05, "stage": "local_arc_close",
                       "evaluator": {"holds": [True, True], "supports": [True, True]}},
                      {"phase": "recovery", "time": 1.1, "stage": "local_relevel",
                       "evaluator": {"holds": [True, steps == 60], "supports": [True, steps == 60]}}]
    rows[0]["capture"] = {"prefix_state": rows[0]["prefix_state"]}
    report = acquisition_metrics(rows)
    assert report["valid"] and report["episodes"] == 3
    assert report["classes"] == {"B": 2, "A": 1}
    assert len(report["changed_40_60"]) == 1
    changed = copy.deepcopy(rows)
    changed[1]["trace"][1]["time"] += .01
    bad = acquisition_metrics(changed)
    assert not bad["valid"]
    assert [8, "left_loss", "prefix_before_relevel"] in bad["prefix_mismatches"]
