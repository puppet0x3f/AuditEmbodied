from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest

import auditembodied.l3_visual_test as experiment
from auditembodied.visual_recovery import MultiSupportVerifier


def test_proprio_observer_never_acquires_rgb(monkeypatch):
    observer = experiment.DecisionObserver("robot")
    observer.reference_gaps = np.array([.018, .02])
    observer._robot_rgb = Mock(side_effect=AssertionError("RGB must not be read"))
    monkeypatch.setattr(experiment, "_proprio", lambda env: np.arange(20))
    fingers = np.array([[.018, -.018], [.021, -.019]])
    monkeypatch.setattr(experiment, "actual_gripper_positions", lambda env: fingers)
    monkeypatch.setattr(experiment, "actual_eef_rotations", lambda env: np.tile(np.eye(3), (2,1,1)))
    observed = observer.observe(SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(time=1.))))
    assert observed["rgb"].size == 0 and not observer.rgb
    assert observed["robot"].shape == (46,)
    np.testing.assert_array_equal(observed["robot"][:20], np.arange(20))
    np.testing.assert_allclose(observed["robot"][20:24], fingers.reshape(-1))
    assert observer.policy_observations == 1


def test_proprio_checkpoint_ignores_even_unreadable_image_port():
    class ForbiddenImage:
        def __array__(self, *args, **kwargs):
            raise AssertionError("image port read")
    model = MultiSupportVerifier(np.zeros(46), np.ones(46), np.zeros((46,3)),
                                 np.zeros(3), np.full(3,.5), "robot")
    np.testing.assert_array_equal(model.predict_proba(ForbiddenImage(), np.zeros((1,46))), [[.5,.5,.5]])


def test_bootstrap_bad_warning_aborts_before_recovery():
    observer = experiment.DecisionObserver("robot")
    warnings = [SimpleNamespace(number=0) for _ in range(int(mujoco.mjtWarning.mjNWARNING))]
    env = SimpleNamespace(raw=SimpleNamespace(modify_observable=Mock()),
                          sim=SimpleNamespace(data=SimpleNamespace(warning=warnings)))
    observer.after_reset(env)
    warnings[int(mujoco.mjtWarning.mjWARN_BADQACC)].number = 1
    with pytest.raises(FloatingPointError,match="bootstrap"):
        observer.register([.018,.018])


def test_uniform_initialization_failure_retained_and_prefix_mismatch_rejected():
    state={"qpos":[0.],"time":.2}
    group={k:{"initialization_state":state.copy(),"initialization_end_state":state.copy(),
               "failure_reason":"initialization_not_supported"} for k in experiment.METHODS}
    assert experiment.paired_prefix(group)["comparable"]
    assert not experiment.paired_prefix(group)["fault_prefix_reached"]
    for r in group.values():r["prefix_state"]={"time":3.35}
    group["visual"]["prefix_state"]["time"] += 1e-12
    assert not experiment.paired_prefix(group)["comparable"]


def test_go_requires_benefit_over_same_logic_proprio_control():
    report={"validity":{"all_cases_comparable":True,"all_sources_frozen":True,"initialized_seed_rate":1.,
            **{k:0 for k in ("numerical_instabilities","execution_errors","contact_anchors","manual_state_edits","forbidden_calls","wrench_leaks")}},
        "by_method":{k:{"episodes":100,"safe_successes":95 if k!="rule" else 80,"protection_violation_rate":0.}
                     for k in experiment.METHODS},
        "paired_costs":{k:{"scenes":95,"recovery_seconds":{"reduction_fraction":.5},
                           "recovery_path_m":{"reduction_fraction":.5}} for k in ("rule","proprio","R3")}}
    criteria={"minimum_initialization_seed_rate":.9,"global_success_margin":.05}
    assert experiment.verdict(report,criteria)["decision"]=="Go"
    report["by_method"]["proprio"]["safe_successes"]=99
    assert experiment.verdict(report,criteria)["decision"]=="No-Go"
    report["by_method"]["proprio"]["safe_successes"]=95
    for key in ("recovery_seconds","recovery_path_m"):
        report["paired_costs"]["proprio"][key]["reduction_fraction"]=0
    assert "independent_visual_benefit" in experiment.verdict(report,criteria)["failed_checks"]


def test_cluster_bootstrap_retains_paired_cost_ratios():
    groups={(s,q):{k:{"success":True,"protection_violation":False,
                 "recovery_seconds":1. if k=="visual" else 2.,
                 "recovery_path_m":.1 if k=="visual" else .2} for k in experiment.METHODS}
            for s in (1,2) for q in range(5)}
    result=experiment.intervals(groups,draws=100)
    assert result["paired_recovery_seconds_reduction_vs_proprio"]["ci95"]==[.5,.5]
    assert result["safe_success_delta_vs_proprio"]["point"]==0
