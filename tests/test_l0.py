from __future__ import annotations

import numpy as np
import pytest

from auditembodied.l0 import (
    L0Config,
    L0Episode,
    OracleVerifier,
    TwoArmL0Env,
    aggregate_metrics,
    run_benchmark,
)


def test_l0_predicate_and_checkpoint_restore() -> None:
    env = TwoArmL0Env(seed=0, config=L0Config(render=False))
    try:
        env.reset()
        predicate = OracleVerifier().evaluate(env)
        assert predicate["giver_holds"] is True
        state = env.checkpoint()
        expected = state.qpos.copy()
        env.place_object_on_table()
        env.restore(state)
        np.testing.assert_allclose(env.sim.data.qpos, expected, atol=1e-10)
    finally:
        env.close()


def test_l0_gate_and_r1_smoke_matrix() -> None:
    results = run_benchmark(
        seeds=[0],
        perturbations=("nominal", "late"),
        conditions=L0Episode.CONDITIONS,
        config=L0Config(render=False),
    )
    metrics = aggregate_metrics(results)
    assert next(r.success for r in results if r.perturbation == "nominal" and r.condition == "no_gate")
    assert next(r.unsupported_release for r in results if r.perturbation == "late" and r.condition == "no_gate")
    assert next(r.success for r in results if r.perturbation == "late" and r.condition == "oracle_gate_r1")
    assert metrics["by_condition"]["oracle_gate_r1"]["recovery_success_rate"] == 1.0


@pytest.mark.parametrize("seed", [0, 14])
def test_continuous_initialization_holds_without_anchors(monkeypatch, seed) -> None:
    env = TwoArmL0Env(seed=seed, config=L0Config(render=False, expert_mode="continuous"))
    try:
        env.reset()
        env.bootstrap_handover()

        def forbidden_edit(*args, **kwargs):
            raise AssertionError("a state edit was attempted during the hold")

        monkeypatch.setattr(env, "_set_hammer_pose", forbidden_edit)
        monkeypatch.setattr(env, "_sync_after_kinematic_edit", forbidden_edit)
        action = np.zeros(14)
        action[6] = action[13] = 1.0
        for _ in range(env.config.control_freq):
            env.step(action)
            assert OracleVerifier().evaluate(env)["receiver_supports"]
        assert env.bootstrap_info["contact_anchor_steps"] == 0
    finally:
        env.close()
