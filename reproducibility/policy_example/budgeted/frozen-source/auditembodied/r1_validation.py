"""Physical R1 validation: receiver loss, regrasp, giver release and 1 s hold."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple
from unittest.mock import patch

import numpy as np

from .continuous import (
    ACTION_DIM, ContinuousActionExpert, ContinuousR1Recovery, TrajectoryConfig,
    TrajectoryEpisode, _apply_rgb_nuisance, _proprio, _record_failure_event,
)
from .l0 import CAMERAS, L0Config, OracleVerifier, TwoArmL0Env, _jsonable


PHYSICAL_PROTOCOL = "auditembodied.physical_receiver_withdrawal.v2"


def run_physical_trial(
    seed: int, perturbation: str = "late", *, render: bool = False,
    config: Optional[TrajectoryConfig] = None,
    gate: Optional[Callable[[Mapping[str, Any], int, bool], bool]] = None,
    validation_only: bool = False,
) -> Tuple[Dict[str, Any], TrajectoryEpisode]:
    """Shared collector/online runner; all actions after initialization are physical.

    A gate receives only the recorded robot stream. Oracle truth is used by
    the expert collection policy when no deployment gate is supplied.
    """
    if perturbation not in ("nominal", "early", "late"):
        raise ValueError(perturbation)
    cfg = config or TrajectoryConfig()
    if cfg.r1_use_contact_anchor or cfg.allow_safety_fallback:
        raise ValueError("physical protocol forbids contact anchors and safety state edits")
    if cfg.capture_stride != 1:
        raise ValueError("physical protocol records every 20 Hz control step")
    env = TwoArmL0Env(int(seed), L0Config(
        render=render, horizon=300, expert_mode="continuous",
        camera_height=cfg.camera_height, camera_width=cfg.camera_width,
        frame_dir=cfg.render_frame_dir,
    ))
    verifier = OracleVerifier()
    expert = ContinuousActionExpert()
    episode = TrajectoryEpisode(int(seed), perturbation)
    rng = np.random.default_rng(cfg.random_seed + int(seed) * 31 + ("nominal", "early", "late").index(perturbation))
    trace = []
    blocked_calls = []
    row: Dict[str, Any] = {
        "seed": int(seed), "perturbation": perturbation, "success": False,
        "failure_reason": None, "release_index": None, "unsupported_release": False,
        "recovery_attempted": False, "observations": 0, "extra_observations": 0,
        "gate_checks": 0, "success_scope": "physical_release_and_1s_hold",
        "unsupported_checks": 0,
    }
    stage = "initialization"

    def capture(ep, current, stage_name, action):
        predicate = verifier.evaluate(current)
        trace.append({
            "stage": stage_name, "time": float(current.sim.data.time),
            "action": None if action is None else np.asarray(action).tolist(),
            **predicate,
        })
        ep.labels.append(int(predicate["receiver_supports"]))
        ep.predicates.append(predicate)
        ep.timestamps.append(float(current.sim.data.time))
        ep.stages.append(stage_name)
        ep.actions.append(np.zeros(ACTION_DIM, dtype=np.float32) if action is None else np.asarray(action, dtype=np.float32).copy())
        if render:
            # step() has already rendered these observations; no second render.
            frames = np.stack([np.asarray(current.obs[f"{camera}_image"], dtype=np.uint8) for camera in CAMERAS])
            ep.rgb.append(_apply_rgb_nuisance(frames, rng, cfg))
            ep.proprio.append(_proprio(current))
        _record_failure_event(ep)

    def check_gate(allow_stale):
        index = len(episode.stages) - 1
        robot_stream = {
            "rgb": np.asarray(episode.rgb), "proprio": np.asarray(episode.proprio),
            "actions": np.asarray(episode.actions), "stages": list(episode.stages),
        }
        accepted = bool(gate(robot_stream, index, allow_stale)) if gate is not None else bool(episode.labels[index])
        row["gate_checks"] += 1
        row["unsupported_checks"] += int(not episode.labels[index])
        # Every policy monitors the same 20 Hz prefix, including the recovery
        # prefix only when it requests R1. Release/hold audit frames cost no
        # verifier observations because they are never consumed by the gate.
        row["observations"] = index + 1
        if allow_stale:
            row["decision_index"] = index
            row["predicted_support_at_decision"] = accepted
            row["true_support_at_decision"] = bool(episode.labels[index])
        row["extra_observations"] = index - row["decision_index"]
        if accepted:
            row["release_index"] = index
            row["true_support_at_release"] = bool(episode.labels[index])
            row["unsupported_release"] = not bool(episode.labels[index])
        return accepted

    def phase(name, targets, grippers, steps):
        nonlocal stage
        stage = name
        expert.execute_phase(
            env, episode, name, targets, contact_mode=None,
            grippers=grippers, gain=0.8, steps=steps, capture=capture,
        )

    def forbidden(name):
        def reject(*args, **kwargs):
            blocked_calls.append({"stage": stage, "method": name})
            raise AssertionError(f"state edit/contact anchor forbidden during {stage}: {name}")
        return reject

    try:
        import mujoco

        env.reset()
        warnings_before = np.array([warning.number for warning in env.sim.data.warning], dtype=int)
        env.bootstrap_handover()
        row["bootstrap"] = env.bootstrap_info
        row["initial_predicate"] = verifier.evaluate(env)
        capture(episode, env, "receiver_grasp", None)
        if not (row["initial_predicate"]["receiver_supports"] and row["initial_predicate"]["giver_holds"]):
            row["failure_reason"] = "initialization_not_supported"
            return row, episode
        # This waypoint is recorded before fault injection, from robot state.
        giver_target = env._eef_pos(0).copy()
        receiver_target = env._eef_pos(1).copy()
        with ExitStack() as guards:
            for name in (
                "_set_hammer_pose", "_move_arm_to_pose", "_move_arm_to_position",
                "_sync_after_kinematic_edit", "_sync_after_reset", "sync_gripper_controller",
                "place_object_on_table", "place_object_in_receiver", "release_giver",
                "receiver_regrasp", "restore", "bootstrap_handover", "reset",
            ):
                guards.enter_context(patch.object(env, name, side_effect=forbidden(name)))
            guards.enter_context(patch.object(
                ContinuousActionExpert, "_set_contact_state", side_effect=forbidden("_set_contact_state")
            ))
            if not validation_only and perturbation != "early":
                phase("continuous_handover", (giver_target, receiver_target), (1.0, 1.0), 12)
            if perturbation != "nominal":
                phase("fault_receiver_open", (giver_target, receiver_target), (1.0, -1.0), 12)
                phase(
                    "fault_receiver_withdraw", (giver_target, receiver_target + [0.0, 0.0, 0.04]),
                    (1.0, -1.0), 12,
                )
            else:
                phase("benign_hold", (giver_target, receiver_target), (1.0, 1.0), 24)
            if not validation_only and perturbation == "early":
                phase("post_fault_hold", (giver_target, receiver_target + [0.0, 0.0, 0.04]), (1.0, -1.0), 12)
            disturbed = verifier.evaluate(env)
            row["disturbed_predicate"] = disturbed
            row["failure_created"] = bool(not disturbed["receiver_supports"])
            row["giver_preserved_during_fault"] = all(point["giver_holds"] for point in trace)
            if perturbation != "nominal" and not row["failure_created"]:
                row["failure_reason"] = "receiver_failure_not_created"
                return row, episode
            if not row["giver_preserved_during_fault"]:
                row["failure_reason"] = "giver_lost_object_during_fault"
                return row, episode

            targets = (giver_target, receiver_target if perturbation == "nominal" else receiver_target + [0.0, 0.0, 0.04])
            phase("gate_decision", targets, (1.0, 1.0 if perturbation == "nominal" else -1.0), 1)
            if not check_gate(True):
                stage = "recovery"
                row["recovery_attempted"] = True
                recovery = ContinuousR1Recovery(use_contact_anchor=False, approach_steps=cfg.r1_approach_steps).run(
                    env, expert, episode, capture, steps=cfg.r1_steps, receiver_target=receiver_target
                )
                row["recovery"] = recovery
                phase("recovery_decision", (giver_target, receiver_target), (1.0, 1.0), 1)
                if not check_gate(False):
                    predicate = verifier.evaluate(env)
                    row["failure_reason"] = (
                        "gate_rejected_supported_recovery" if predicate["receiver_supports"]
                        else "receiver_did_not_regrasp" if not predicate["receiver_holds"]
                        else "recovery_relative_speed_exceeded"
                    )
                    return row, episode

            phase("giver_open", (giver_target, receiver_target), (-1.0, 1.0), 14)
            retreat_target = giver_target + [0.0, 0.0, 0.06]
            phase("giver_retreat", (retreat_target, receiver_target), (-1.0, 1.0), 12)
            released = verifier.evaluate(env)
            finger_positions = np.asarray(
                env.sim.data.qpos[env.raw.robots[0]._ref_gripper_joint_pos_indexes["right"]]
            )
            row["giver_finger_positions_at_release"] = finger_positions.tolist()
            row["released_predicate"] = released
            row["giver_released"] = bool(not released["giver_holds"] and np.min(np.abs(finger_positions)) >= 0.03)
            if not row["giver_released"]:
                row["failure_reason"] = "giver_not_fully_released"
                return row, episode
            release_trace = [point for point in trace if point["stage"] in ("giver_open", "giver_retreat")]
            row["support_continuous_during_release"] = all(point["receiver_supports"] for point in release_trace)
            if not row["support_continuous_during_release"]:
                row["failure_reason"] = "receiver_support_lost_during_release"
                return row, episode
            start_time = float(env.sim.data.time)
            hold_initial = verifier.evaluate(env)
            hold_steps = int(math.ceil(env.config.control_freq * 1.0))
            phase("post_release_hold", (retreat_target, receiver_target), (-1.0, 1.0), hold_steps)
            hold_trace = [point for point in trace if point["stage"] == "post_release_hold"]
            row["hold_seconds"] = float(env.sim.data.time) - start_time
            row["post_release_receiver_stable"] = bool(
                hold_initial["receiver_supports"] and all(point["receiver_supports"] for point in hold_trace)
            )
            row["giver_stayed_released"] = bool(
                not hold_initial["giver_holds"] and all(not point["giver_holds"] for point in hold_trace)
            )
            row["post_release_max_relative_speed"] = max(point["relative_speed"] for point in hold_trace)
            row["post_release_min_clearance"] = min(point["clearance"] for point in hold_trace)
            after = np.array([warning.number for warning in env.sim.data.warning], dtype=int)
            row["bad_state_warnings"] = {
                name: int(after[int(getattr(mujoco.mjtWarning, name))] - warnings_before[int(getattr(mujoco.mjtWarning, name))])
                for name in ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC", "mjWARN_BADCTRL")
            }
            row["success"] = bool(
                row["post_release_receiver_stable"] and row["giver_stayed_released"]
                and row["hold_seconds"] >= 1.0 - 1e-9
                and np.all(np.isfinite(env.sim.data.qpos)) and np.all(np.isfinite(env.sim.data.qvel))
                and not any(row["bad_state_warnings"].values())
            )
            if not row["success"]:
                row["failure_reason"] = (
                    "numerical_instability" if any(row["bad_state_warnings"].values())
                    else "giver_contact_returned" if not row["giver_stayed_released"]
                    else "receiver_support_lost_during_1s_hold"
                )
            if "recovery" in row:
                row["recovery"]["handover_success"] = row["success"]
        return row, episode
    except Exception as exc:
        row["failure_reason"] = "forbidden_state_edit" if blocked_calls else "execution_error"
        row["error"] = repr(exc)
        row["failed_stage"] = stage
        if render and not episode.rgb:
            # Preserve a synchronized terminal observation for failed initial
            # IK as well; it must not disappear from the 50-seed dataset.
            env.obs = env.raw._get_observations(force_update=True)
            capture(episode, env, "initialization_failed", None)
        return row, episode
    finally:
        row["contact_anchor_count"] = len(episode.contact_anchors)
        row["state_edit_count_after_initialization"] = len(episode.state_edits)
        row["blocked_calls"] = blocked_calls
        row["recovered"] = bool(row["recovery_attempted"] and row["success"])
        row["r1_success"] = row["recovered"]
        if "recovery" in row:
            row["recovery"]["handover_success"] = bool(row["success"])
        row["trace"] = trace
        episode.metadata.update({
            "protocol": PHYSICAL_PROTOCOL, "expert": expert.name,
            "r1_use_contact_anchor": False, "allow_safety_fallback": False,
            "state_edit_count": 0, "contact_anchor_count": 0, "safety_fallback_count": 0,
            "state_edit_scope": "after initialization",
            "initialization": "separated_pose_ik_then_physical_close",
            "camera_names": list(CAMERAS), "proprio_dim": 20, "action_dim": ACTION_DIM,
            "nuisance": {key: getattr(cfg, key) for key in (
                "unrelated_changes", "occlusion_probability", "camera_jitter_pixels", "appearance_jitter",
            )},
        })
        episode.evaluator_metadata.update({key: _jsonable(value) for key, value in row.items() if key != "trace"})
        if "recovery" in row:
            episode.evaluator_metadata["r1"] = _jsonable(row["recovery"])
        env.close()


def validate_full_r1_trial(seed: int) -> Dict[str, Any]:
    row, _ = run_physical_trial(seed, validation_only=True)
    return row


def validate_full_r1(
    seeds: Iterable[int] = range(10), output: str | Path = "outputs/r1-full-validation.json"
) -> Dict[str, Any]:
    rows = [validate_full_r1_trial(int(seed)) for seed in seeds]
    successes = sum(row["success"] for row in rows)
    report = {
        "schema": "auditembodied.r1.full_physical.v1",
        "success_definition": "receiver support throughout giver opening/retreat and 1 s after complete release",
        "protocol": {
            "fault": "receiver open 12 steps then withdraw upward 4 cm for 12 steps; giver holds",
            "recovery": "registered pre-fault robot waypoint; open align 16 steps, close/settle 22 steps",
            "release": "giver opens 14 steps, retreats upward 6 cm for 12 steps",
            "hold_seconds": 1.0, "control_freq": 20,
            "external_force": [0.0, 0.0, 0.0],
            "contact_anchors_allowed": False, "post_initialization_state_edits_allowed": False,
            "target_source": "registered robot EEF position, not object truth",
        },
        "seeds": [row["seed"] for row in rows], "episodes": len(rows),
        "successes": successes, "success_rate": successes / max(1, len(rows)),
        "failure_reasons": dict(Counter(row["failure_reason"] for row in rows if not row["success"])),
        "contact_anchor_count": sum(row["contact_anchor_count"] for row in rows),
        "state_edit_count_after_initialization": sum(row["state_edit_count_after_initialization"] for row in rows),
        "rows": rows,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", default="outputs/r1-full-validation.json")
    args = parser.parse_args()
    result = validate_full_r1(range(args.seeds), args.output)
    print(json.dumps({key: result[key] for key in ("episodes", "successes", "success_rate", "failure_reasons")}, ensure_ascii=False))
