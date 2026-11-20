"""Training-only, forced-candidate physical recovery matrix for early L3."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import csv
import json
import multiprocessing
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np

from .continuous import ContinuousActionExpert, ContinuousR1Recovery, TrajectoryEpisode, _proprio
from .l0 import L0Config, OracleVerifier, TwoArmL0Env, _jsonable
from .l1 import LogisticVisualVerifier
from .recovery_skills import (ContinuousR2Recovery, ContinuousR3Recovery, PoseRetargetedR1Recovery,
                              actual_eef_rotations, actual_gripper_positions)


SCENARIOS = ("receiver_loss", "giver_loss", "coupled_offset")
SKILLS = ("continue", "R1", "R2", "R3")
RETARGET_SKILLS = ("continue", "R1", "R1_retarget", "R3")
GRASP_SCENARIOS = ("object_sham", "object_rotation_mild", "object_rotation")
GRASP_SKILLS = ("continue", "R1_retarget", "R2", "R3")


@dataclass(frozen=True)
class MatrixConfig:
    fault_steps: int = 12
    withdrawal_m: float = 0.04
    coupled_offset_m: float = 0.05
    coupled_mode: str = "opposed_offset"
    coupled_shared_shift_m: float = 0.12
    r3_placement_steps: int = 40
    r3_pickup_height_adjustment: float = 0.0
    r3_receiver_height_offset: float = -0.025
    r3_search_offsets: tuple = (0.0, -0.02, 0.02, -0.04, 0.04, -0.06, 0.06)
    r1_retarget_height_offset: float = 0.0
    object_force_n: tuple = (0.0, -8.0, 0.0)
    object_torque_nm: tuple = (-0.12, 0.0, 0.0)
    object_mild_torque_nm: tuple = (-0.5, 0.0, 0.0)
    wrench_steps: int = 6
    wrench_settle_steps: int = 4
    wrench_profile: str = "constant"
    rule_gap_tolerance: float = 0.05


def _table_contact(env):
    object_geoms = set(env.raw.hammer.contact_geoms)
    for contact in env.sim.data.contact[:env.sim.data.ncon]:
        a, b = (env.sim.model.geom_id2name(int(i)) for i in (contact.geom1, contact.geom2))
        if (a in object_geoms and b.startswith("table")) or (b in object_geoms and a.startswith("table")):
            return True
    return False


def _grasp_relative_change(before, after):
    """Evaluator-only object displacement in each measured gripper frame."""
    from robosuite.utils import transform_utils as transforms
    poses = []
    for point in (before, after):
        state, truth = point["robot"], point["evaluator"]
        rotations = np.asarray(state["actual_eef_rotations2x3x3"])
        positions = np.asarray(state["proprio20"][:6]).reshape(2, 3)
        object_position = np.asarray(truth["object_position"])
        object_rotation = transforms.quat2mat(np.roll(truth["object_quaternion"], -1))
        poses.append([(R.T @ (object_position - p), R.T @ object_rotation) for p, R in zip(positions, rotations)])
    changes = []
    for (p0, R0), (p1, R1) in zip(*poses):
        dp = p1 - p0
        angle = np.degrees(np.arccos(np.clip((np.trace(R1 @ R0.T) - 1) / 2, -1, 1)))
        changes.append({"translation_m": dp.tolist(), "translation_norm_m": float(np.linalg.norm(dp)),
                        "rotation_deg": float(angle)})
    return changes


def run_trial(seed, scenario, skill, config=None, *, observer=None, recovery_policy=None):
    cfg = config or MatrixConfig()
    if scenario not in (*SCENARIOS, *GRASP_SCENARIOS, "object_slip", "nominal") or skill not in (*SKILLS, "R1_retarget", "rule", "visual"):
        raise ValueError((scenario, skill))
    if skill == "visual" and (observer is None or recovery_policy is None):
        raise ValueError("visual recovery requires robot RGB observations and a policy")
    env = TwoArmL0Env(int(seed), L0Config(render=observer is not None, horizon=2000, expert_mode="continuous"))
    expert, oracle = ContinuousActionExpert(), OracleVerifier()
    episode = TrajectoryEpisode(int(seed), scenario)
    row = {"seed": int(seed), "scenario": scenario, "candidate": skill,
           "success": False, "handover_completed": False, "failure_reason": None}
    trace, blocked = [], []
    phase_category, stage = "initialization", "initialization"
    last_positions = None
    recovery_path, total_path = 0.0, 0.0
    flight_run, max_flight_run = 0, 0
    arm_collision_steps = 0
    warning_before = None
    wrench_leak_steps = 0

    def capture(ep, current, name, action):
        nonlocal stage, last_positions, recovery_path, total_path, flight_run, max_flight_run, arm_collision_steps, wrench_leak_steps
        stage = name
        bad_warnings = {key: int(current.sim.data.warning[int(getattr(mujoco.mjtWarning, key))].number
                                - warning_before[int(getattr(mujoco.mjtWarning, key))])
                        for key in ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC", "mjWARN_BADCTRL")}
        rolled_back = bool(trace and float(current.sim.data.time) < trace[-1]["time"])
        if any(bad_warnings.values()) or rolled_back:
            row["bad_state_warnings"] = bad_warnings
            row["numerical_failure_stage"] = name
            row["time_rollback_detected"] = rolled_back
            raise FloatingPointError(f"simulation became unstable during {name}")
        positions = np.stack([current._eef_pos(i) for i in range(2)])
        distance = 0.0 if last_positions is None else float(np.linalg.norm(positions - last_positions, axis=1).sum())
        last_positions = positions
        if phase_category in ("recovery", "release", "hold"):
            wrench_leak_steps += int(np.any(current.sim.data.xfrc_applied))
            total_path += distance
        if phase_category == "recovery":
            recovery_path += distance
        predicate = oracle.evaluate(current)
        table = _table_contact(current)
        if phase_category in ("recovery", "release", "hold"):
            flight_run = flight_run + 1 if not (predicate["giver_holds"] or predicate["receiver_holds"] or table) else 0
            max_flight_run = max(max_flight_run, flight_run)
            for contact in current.sim.data.contact[:current.sim.data.ncon]:
                a, b = (current.sim.model.geom_id2name(int(i)) for i in (contact.geom1, contact.geom2))
                if contact.dist < -0.01 and ((a.startswith("robot0") and b.startswith("robot1")) or
                                           (a.startswith("robot1") and b.startswith("robot0"))):
                    arm_collision_steps += 1
                    break
        trace.append({"time": float(current.sim.data.time), "phase": phase_category, "stage": name,
                      "robot": {"proprio20": _proprio(current).tolist(),
                                "actual_eef_rotations2x3x3": actual_eef_rotations(current).tolist(),
                                "actual_finger_qpos4": actual_gripper_positions(current).reshape(-1).tolist(),
                                "action14": None if action is None else np.asarray(action).tolist()},
                      "evaluator": {**predicate, "table_contact": table,
                                    "object_position": np.asarray(current.sim.data.body_xpos[current.raw.hammer_body_id]).tolist(),
                                    "object_quaternion": np.asarray(current.sim.data.body_xquat[current.raw.hammer_body_id]).tolist()}})
        if observer is not None:
            observer.capture(current, trace[-1])

    def phase(name, targets, grips, steps):
        expert.execute_phase(env, episode, name, targets, contact_mode=None,
                             grippers=grips, steps=steps, gain=0.8, capture=capture)

    def forbidden(name):
        def reject(*args, **kwargs):
            blocked.append({"stage": stage, "method": name})
            raise AssertionError(f"state edit/contact anchor forbidden: {name}")
        return reject

    try:
        import mujoco
        env.reset()
        warning_before = np.array([w.number for w in env.sim.data.warning])
        if observer is not None:
            observer.after_reset(env)
        row["initialization_state"] = {key: np.asarray(getattr(env.sim.data, key)).copy().tolist()
                                       for key in ("qpos", "qvel", "ctrl", "act")}
        row["initialization_state"]["time"] = float(env.sim.data.time)
        stage = "bootstrap_handover"
        env.bootstrap_handover()
        row["initialization_end_state"] = {key: np.asarray(getattr(env.sim.data, key)).copy().tolist()
                                           for key in ("qpos", "qvel", "ctrl", "act")}
        row["initialization_end_state"]["time"] = float(env.sim.data.time)
        registered = np.stack([env._eef_pos(i) for i in range(2)])
        registered_rotations = actual_eef_rotations(env)
        row["registered_robot_poses"] = {"positions": registered.tolist(), "rotations": registered_rotations.tolist()}
        reference_gaps = np.mean(np.abs(actual_gripper_positions(env)), axis=1)
        row["registered_robot_finger_gaps"] = reference_gaps.tolist()
        if observer is not None:
            observer.register(reference_gaps)
        row["bootstrap"] = env.bootstrap_info
        row["initial_predicate"] = oracle.evaluate(env)
        row["initialization_supported"] = bool(row["initial_predicate"]["giver_holds"] and
                                               row["initial_predicate"]["receiver_supports"])
        if not (row["initial_predicate"]["giver_holds"] and row["initial_predicate"]["receiver_supports"]):
            row["failure_reason"] = "initialization_not_supported"
            return row
        with ExitStack() as guards:
            for name in ("_set_hammer_pose", "_move_arm_to_pose", "_move_arm_to_position",
                         "_sync_after_kinematic_edit", "_sync_after_reset", "sync_gripper_controller",
                         "place_object_on_table", "place_object_in_receiver", "release_giver",
                         "receiver_regrasp", "restore", "bootstrap_handover", "reset"):
                guards.enter_context(patch.object(env, name, side_effect=forbidden(name)))
            guards.enter_context(patch.object(ContinuousActionExpert, "_set_contact_state", side_effect=forbidden("_set_contact_state")))
            phase_category = "prefix"
            phase("common_hold", registered, (1.0, 1.0), 12)
            targets = registered.copy()
            if scenario in (*GRASP_SCENARIOS, "object_slip"):
                from robosuite.utils import transform_utils as transforms
                phase("fault_prefix_1", targets, (1.0, -1.0), cfg.fault_steps)
                targets[1, 2] += cfg.withdrawal_m
                phase("fault_prefix_2", targets, (1.0, -1.0), cfg.fault_steps)
                hold_rotations = actual_eef_rotations(env)
                wrench = np.zeros(6)
                wrench[:3] = cfg.object_force_n if scenario == "object_slip" else (0, 0, 0)
                wrench[3:] = cfg.object_torque_nm if scenario == "object_rotation" else (0, 0, 0)
                if scenario == "object_rotation_mild":
                    wrench[3:] = cfg.object_mild_torque_nm
                row["physical_wrench"] = {"world_force_n": wrench[:3].tolist(), "world_torque_nm": wrench[3:].tolist(),
                                          "control_steps": cfg.wrench_steps, "settle_steps": cfg.wrench_settle_steps,
                                          "profile": cfg.wrench_profile}
                body = env.raw.hammer_body_id
                try:
                    for index in range(cfg.wrench_steps + cfg.wrench_settle_steps):
                        active = index < cfg.wrench_steps
                        if cfg.wrench_profile == "constant":
                            scale = 1.0
                        elif cfg.wrench_profile == "half_sine":
                            scale = float(np.sin(np.pi * (index + 0.5) / cfg.wrench_steps))
                        else:
                            raise ValueError(f"unknown wrench profile: {cfg.wrench_profile}")
                        env.sim.data.xfrc_applied[body] = wrench * scale if active else np.zeros(6)
                        action = expert.action_towards(env, targets, grippers=(1.0, -1.0), gain=0.8)
                        # Ordinary orientation feedback keeps robot motion
                        # from absorbing the whole perturbation as a rigid
                        # hand/object displacement. It creates no contact.
                        for arm, current in enumerate(actual_eef_rotations(env)):
                            error = transforms.quat2axisangle(transforms.mat2quat(hold_rotations[arm] @ current.T))
                            action[arm*7+3:arm*7+6] = np.clip(error / 0.5, -1.0, 1.0)
                        env.step(action)
                        capture(episode, env, "fault_wrench" if active else "fault_settle", action)
                finally:
                    env.sim.data.xfrc_applied[body] = 0.0
            elif scenario in ("receiver_loss", "giver_loss"):
                arm = 1 if scenario == "receiver_loss" else 0
                grips = [1.0, 1.0]
                grips[arm] = -1.0
                phase("fault_prefix_1", targets, grips, cfg.fault_steps)
                targets[arm, 2] += cfg.withdrawal_m
                phase("fault_prefix_2", targets, grips, cfg.fault_steps)
            elif scenario == "coupled_offset":
                if cfg.coupled_mode == "shared_shift_receiver_loss":
                    # A failed common carry: both robots physically move the
                    # grasp, then one loses support. This changes the shared
                    # grasp registration as well as receiver support. No
                    # object pose edits, external anchors, or truth feedback.
                    shift = np.array([cfg.coupled_shared_shift_m, 0.0, 0.0])
                    for step in range(32):
                        targets = registered + shift * ((step + 1) / 32)
                        phase("fault_prefix_shift", targets, (1.0, 1.0), 1)
                    phase("fault_prefix_1", targets, (1.0, 1.0), cfg.fault_steps)
                    phase("fault_prefix_unload", targets, (1.0, -1.0), cfg.fault_steps)
                    targets[1, 2] += cfg.withdrawal_m
                    phase("fault_prefix_2", targets, (1.0, -1.0), cfg.fault_steps)
                elif cfg.coupled_mode == "opposed_offset":
                    targets[1, 0] += cfg.coupled_offset_m
                    phase("fault_prefix_1", targets, (1.0, 1.0), cfg.fault_steps)
                    targets[0, 0] -= cfg.coupled_offset_m
                    phase("fault_prefix_2", targets, (1.0, 1.0), cfg.fault_steps)
                else:
                    raise ValueError(f"unknown coupled mode: {cfg.coupled_mode}")
            else:
                phase("fault_prefix_1", targets, (1.0, 1.0), cfg.fault_steps)
                phase("fault_prefix_2", targets, (1.0, 1.0), cfg.fault_steps)
            row["prefix_state"] = {key: np.asarray(getattr(env.sim.data, key)).copy().tolist()
                                   for key in ("qpos", "qvel", "ctrl", "act")}
            row["prefix_state"]["time"] = float(env.sim.data.time)
            row["post_fault_predicate"] = oracle.evaluate(env)
            if observer is not None:
                observer.capture(env, trace[-1], force=True)
            row["grasp_relative_change"] = _grasp_relative_change(
                [p for p in trace if p["stage"] == "common_hold"][-1], trace[-1])
            row["external_wrench_at_recovery_start"] = float(np.linalg.norm(env.sim.data.xfrc_applied))
            start_time = float(env.sim.data.time)
            phase_category, stage = "recovery", "recovery"
            if skill == "continue":
                recovery = {"skill": "continue", "control_steps": 0}
            elif skill == "R1":
                recovery = ContinuousR1Recovery(use_contact_anchor=False, approach_steps=16).run(
                    env, expert, episode, capture, steps=38, receiver_target=registered[1])
            elif skill == "R1_retarget":
                recovery = PoseRetargetedR1Recovery().run(
                    env, expert, episode, capture, registered, registered_rotations,
                    height_offset=cfg.r1_retarget_height_offset)
            elif skill == "R2":
                recovery = ContinuousR2Recovery().run(env, expert, episode, capture, registered[0])
            elif skill == "rule":
                from .recovery_rules import LocalThenTableRule
                recovery = LocalThenTableRule(cfg.rule_gap_tolerance).run(
                    env, expert, episode, capture, registered, registered_rotations, reference_gaps,
                    np.asarray(env.raw.table_offset), ContinuousR3Recovery(
                        cfg.r3_search_offsets, cfg.r3_placement_steps, cfg.r3_pickup_height_adjustment,
                        cfg.r3_receiver_height_offset))
            elif skill == "visual":
                recovery = recovery_policy.run(
                    env, expert, episode, capture, registered, registered_rotations, reference_gaps,
                    np.asarray(env.raw.table_offset), ContinuousR3Recovery(
                        cfg.r3_search_offsets, cfg.r3_placement_steps, cfg.r3_pickup_height_adjustment,
                        cfg.r3_receiver_height_offset), observe=lambda: observer.observe(env))
            else:
                recovery = ContinuousR3Recovery(
                    cfg.r3_search_offsets, cfg.r3_placement_steps, cfg.r3_pickup_height_adjustment,
                    cfg.r3_receiver_height_offset,
                ).run(env, expert, episode, capture, registered, reference_gaps, np.asarray(env.raw.table_offset))
            row["recovery"] = recovery
            used_global = skill == "R3" or recovery.get("used_global_recovery", False)
            row["used_global_recovery"] = used_global
            row["recovery_seconds"] = float(env.sim.data.time) - start_time
            row["recovery_path_m"] = recovery_path
            row["post_recovery_predicate"] = oracle.evaluate(env)
            if observer is not None:
                observer.capture(env, trace[-1], force=True)
            recovered = [p["evaluator"] for p in trace if p["phase"] == "recovery"]
            row["receiver_support_continuous_during_recovery"] = all(p["receiver_supports"] for p in recovered)
            row["giver_hold_continuous_during_recovery"] = all(p["giver_holds"] for p in recovered)
            table_points = [p["evaluator"] for p in trace if p["stage"] == "r3_table_settle"]
            row["table_setdown_verified"] = bool(table_points and any(
                p["table_contact"] and not p["giver_holds"] and not p["receiver_holds"] for p in table_points))
            lift_points = [p["evaluator"] for p in trace if p["stage"] in ("r3_pickup_lift", "r3_return_to_handover")]
            row["table_regrasp_and_lift_verified"] = bool(lift_points and any(
                p["giver_holds"] and p["clearance"] > p["height_threshold"] for p in lift_points))
            row["table_cycle_verified"] = row["table_setdown_verified"] and row["table_regrasp_and_lift_verified"]
            row["skill_goal_achieved"] = bool(
                row["table_cycle_verified"] if used_global
                else row["post_recovery_predicate"]["receiver_supports"] if skill in ("R1", "R1_retarget", "rule", "visual")
                else row["post_recovery_predicate"]["giver_holds"] and row["receiver_support_continuous_during_recovery"] if skill == "R2"
                else True)

            # Continuing does not secretly reapproach a registered grasp:
            # maintain the measured endpoint poses, open the giver and retreat.
            release_targets = np.stack([env._eef_pos(i) for i in range(2)])
            phase_category = "release"
            phase("giver_open", release_targets, (-1.0, 1.0), 14)
            release_targets[0, 2] += 0.06
            phase("giver_retreat", release_targets, (-1.0, 1.0), 12)
            released = oracle.evaluate(env)
            fingers = actual_gripper_positions(env)[0]
            row["giver_fully_released"] = bool(not released["giver_holds"] and np.min(np.abs(fingers)) >= 0.03)
            row["support_continuous_during_release"] = all(
                p["evaluator"]["receiver_supports"] for p in trace if p["phase"] == "release")
            phase_category = "hold"
            hold_start = float(env.sim.data.time)
            phase("post_release_hold", release_targets, (-1.0, 1.0), env.config.control_freq)
            held = [p["evaluator"] for p in trace if p["phase"] == "hold"]
            row["hold_seconds"] = float(env.sim.data.time) - hold_start
            row["receiver_stable_for_1s"] = bool(released["receiver_supports"] and
                all(p["receiver_supports"] and not p["giver_holds"] for p in held) and row["hold_seconds"] >= 1.0 - 1e-9)
            after = np.array([w.number for w in env.sim.data.warning])
            row["bad_state_warnings"] = {name: int(after[int(getattr(mujoco.mjtWarning, name))] - warning_before[int(getattr(mujoco.mjtWarning, name))])
                                         for name in ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC", "mjWARN_BADCTRL")}
            finite = bool(np.all(np.isfinite(env.sim.data.qpos)) and np.all(np.isfinite(env.sim.data.qvel)))
            row["handover_completed"] = bool(row["giver_fully_released"] and row["receiver_stable_for_1s"]
                                               and finite and not any(row["bad_state_warnings"].values()))
            row["success"] = bool(row["handover_completed"] and row["support_continuous_during_release"]
                                  and (not used_global or row["skill_goal_achieved"]))
            if used_global:
                row["skill_goal_achieved"] = row["success"]
            row["total_seconds_after_fault"] = float(env.sim.data.time) - start_time
            row["total_path_after_fault_m"] = total_path
            if not row["success"]:
                row["failure_reason"] = (
                    "numerical_instability" if not finite or any(row["bad_state_warnings"].values())
                    else "table_setdown_not_verified" if used_global and not row["table_setdown_verified"]
                    else "table_pickup_not_verified" if used_global and not row["table_regrasp_and_lift_verified"]
                    else "giver_not_released" if not row["giver_fully_released"]
                    else "receiver_support_lost_during_release" if not row["support_continuous_during_release"]
                    else "receiver_failed_1s_hold")
        return row
    except Exception as exc:
        row["failure_reason"] = ("numerical_instability" if isinstance(exc, FloatingPointError)
                                 else "forbidden_state_edit" if blocked
                                 else "initialization_error" if phase_category == "initialization" else "execution_error")
        if phase_category == "initialization":
            row["initialization_end_state"] = {key: np.asarray(getattr(env.sim.data, key)).copy().tolist()
                                               for key in ("qpos", "qvel", "ctrl", "act")}
            row["initialization_end_state"]["time"] = float(env.sim.data.time)
        row["error"] = repr(exc)
        row["failed_stage"] = stage
        return row
    finally:
        row["contact_anchor_count"] = len(episode.contact_anchors)
        row["state_edit_count_after_initialization"] = len(episode.state_edits)
        row["blocked_calls"] = blocked
        row["external_wrench_leak_steps"] = wrench_leak_steps
        row["protection"] = {"unprotected_without_gripper_or_table_over_0_1s": max_flight_run > 2,
                             "max_unsupported_seconds": max_flight_run / env.config.control_freq,
                             "deep_robot_robot_contact_steps": arm_collision_steps,
                             "receiver_support_discontinuous_during_release": row.get("support_continuous_during_release") is False,
                             "state_edit_attempt": bool(blocked)}
        row["protection_violation"] = bool(max_flight_run > 2 or arm_collision_steps or blocked or
                                           row.get("support_continuous_during_release") is False or
                                           any(row.get("bad_state_warnings", {}).values()))
        row["trace"] = trace
        if observer is not None:
            row["policy_observations"] = observer.policy_observations
            row["recorded_rgb_bundles"] = len(observer.rgb)
        env.sim.data.xfrc_applied[:] = 0.0
        env.close()


def _job(job):
    return run_trial(*job)


def _summary(rows):
    successful = [r for r in rows if r["success"]]
    return {"episodes": len(rows), "successes": len(successful),
            "safe_successes": sum(r["success"] and not r["protection_violation"] for r in rows),
            "numerical_instabilities": sum(r.get("failure_reason") == "numerical_instability" for r in rows),
            "success_rate": len(successful) / max(1, len(rows)),
            "handover_completed_count": sum(r["handover_completed"] for r in rows),
            "skill_goal_count": sum(r.get("skill_goal_achieved", False) for r in rows),
            "mean_recovery_seconds": float(np.mean([r.get("recovery_seconds", 0) for r in rows])),
            "mean_recovery_path_m": float(np.mean([r.get("recovery_path_m", 0) for r in rows])),
            "mean_recovery_seconds_successful": float(np.mean([r["recovery_seconds"] for r in successful])) if successful else None,
            "mean_recovery_path_m_successful": float(np.mean([r["recovery_path_m"] for r in successful])) if successful else None,
            "protection_violation_rate": float(np.mean([r["protection_violation"] for r in rows])),
            "failure_reasons": dict(Counter(r["failure_reason"] for r in rows if not r["success"]))}


def run_retarget_ablation(seeds, output_dir, config=None, workers=4):
    """Replay the coupled fault with a stronger, robot-pose-based local R1."""
    cfg, output = config or MatrixConfig(), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds)
    jobs = [(seed, "coupled_offset", skill, cfg) for seed in seeds for skill in RETARGET_SKILLS]
    rows = []
    with (output / "retarget-trials.jsonl").open("w") as log:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for row in pool.map(_job, jobs):
                rows.append(row)
                log.write(json.dumps(_jsonable(row)) + "\n"); log.flush()
                print(json.dumps({k: row.get(k) for k in ("seed", "candidate", "success", "failure_reason")}), flush=True)
    groups = {seed: {r["candidate"]: r for r in rows if r["seed"] == seed} for seed in seeds}
    matches, additional, paired = [], [], []
    for seed, group in groups.items():
        states = [r.get("prefix_state") for r in group.values()]
        matches.append({"seed": seed, "identical_prefix": bool(all(s is not None for s in states) and all(
            all(np.array_equal(states[0][k], s[k]) for k in states[0]) for s in states[1:]))})
        safe = {k: r["success"] and not r["protection_violation"] for k, r in group.items()}
        if safe["R3"] and not any(safe[k] for k in ("continue", "R1", "R1_retarget")):
            additional.append(seed)
        if safe["R1_retarget"] and safe["R3"]:
            paired.append((group["R1_retarget"], group["R3"]))
    costs = {"matched_safe_successful_scenes": len(paired)}
    if paired:
        for key in ("recovery_seconds", "recovery_path_m"):
            local = float(np.mean([a[key] for a, _ in paired]))
            global_ = float(np.mean([b[key] for _, b in paired]))
            costs[key] = {"R1_retarget": local, "R3": global_, "reduction_fraction": 1 - local / global_}
    report = {"scope": "training-only robot-pose R1 control; no online selector evaluated",
              "seeds": seeds, "config": asdict(cfg), "episodes": len(rows),
              "by_skill": {k: _summary([r for r in rows if r["candidate"] == k]) for k in RETARGET_SKILLS},
              "prefix_comparisons": matches, "all_prefixes_identical": all(x["identical_prefix"] for x in matches),
              "R3_additional_safe_seed_ids": additional, "R3_additional_safe_scenes": len(additional),
              "paired_success_costs": costs,
              "retarget_giver_hold_continuous_count": sum(r.get("giver_hold_continuous_during_recovery", False)
                                                          for r in rows if r["candidate"] == "R1_retarget"),
              "contact_anchor_count": sum(r["contact_anchor_count"] for r in rows),
              "state_edit_count_after_initialization": sum(r["state_edit_count_after_initialization"] for r in rows),
              "selector_implemented": False}
    (output / "retarget-report.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    fields = ("seed", "candidate", "success", "recovery_seconds", "recovery_path_m", "total_seconds_after_fault",
              "giver_hold_continuous_during_recovery", "protection_violation", "failure_reason")
    with (output / "retarget.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({k: r.get(k) for k in fields} for r in rows)
    return report


def run_rule_validation(seeds, output_dir, config, workers=4, reference_dir=None):
    """Validate actual rule execution against paired local/global candidates."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    reference = [] if reference_dir is None else [json.loads(x) for x in
        (Path(reference_dir) / "trials.jsonl").read_text().splitlines()]
    policies = ("rule",) if reference else ("R1_retarget", "R3", "rule")
    seeds = list(seeds)
    jobs = [(seed, s, k, config) for seed in seeds for s in GRASP_SCENARIOS for k in policies]
    rows = []
    with (output / "rule-trials.jsonl").open("w") as log:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for row in pool.map(_job, jobs):
                rows.append(row); log.write(json.dumps(_jsonable(row)) + "\n"); log.flush()
                print(json.dumps({k: row.get(k) for k in ("seed", "scenario", "candidate", "success", "failure_reason")}), flush=True)
    groups = {}
    for row in reference + rows:
        if row["seed"] in seeds and row["scenario"] in GRASP_SCENARIOS:
            groups.setdefault((row["seed"], row["scenario"]), {})[row["candidate"]] = row
    checks, paired = [], []
    for (seed, scenario), group in groups.items():
        states = [group[k].get("prefix_state") for k in ("R1_retarget", "R3", "rule")]
        checks.append({"seed": seed, "scenario": scenario, "identical_prefix": bool(
            all(x is not None for x in states) and all(all(np.array_equal(states[0][k], x[k]) for k in states[0]) for x in states[1:]))})
        if all(group[k]["success"] and not group[k]["protection_violation"] for k in ("rule", "R3")):
            paired.append((group["rule"], group["R3"]))
    rule_rows = [r for r in rows if r["candidate"] == "rule"]
    comparisons = {"matched_safe_successful_scenes": len(paired)}
    if paired:
        for key in ("recovery_seconds", "recovery_path_m"):
            rule_mean = float(np.mean([a[key] for a, _ in paired])); global_mean = float(np.mean([b[key] for _, b in paired]))
            comparisons[key] = {"rule": rule_mean, "R3": global_mean, "reduction_fraction": 1 - rule_mean / global_mean}
    report = {"seeds": seeds, "config": asdict(config), "reference_dir": str(reference_dir),
              "by_policy": {k: _summary([g[k] for g in groups.values()]) for k in ("R1_retarget", "R3", "rule")},
              "by_scenario": {s: {k: _summary([g[k] for (seed, name), g in groups.items() if name == s])
                                  for k in ("R1_retarget", "R3", "rule")} for s in GRASP_SCENARIOS},
              "prefix_comparisons": checks, "all_prefixes_identical": all(c["identical_prefix"] for c in checks),
              "global_escalations": sum(r.get("used_global_recovery", False) for r in rule_rows),
              "paired_costs_vs_R3": comparisons,
              "contact_anchor_count": sum(r["contact_anchor_count"] for r in rows),
              "state_edit_count_after_initialization": sum(r["state_edit_count_after_initialization"] for r in rows),
              "external_wrench_leak_steps": sum(r.get("external_wrench_leak_steps", 0) for r in rows)}
    (output / "rule-report.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    fields = ("seed", "scenario", "candidate", "success", "used_global_recovery", "recovery_seconds", "recovery_path_m", "protection_violation", "failure_reason")
    with (output / "rule.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({k: row.get(k) for k in fields} for row in rows)
    return report


def run_matrix(seeds, output_dir, config=None, workers=4, scenarios=SCENARIOS, skills=SKILLS):
    cfg, output = config or MatrixConfig(), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds)
    local_baseline = "R1_retarget" if "R1_retarget" in skills else "R1"
    jobs = [(seed, scenario, skill, cfg) for seed in seeds for scenario in scenarios for skill in skills]
    rows = []
    with (output / "trials.jsonl").open("w") as log:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for row in pool.map(_job, jobs):
                rows.append(row)
                log.write(json.dumps(_jsonable(row)) + "\n"); log.flush()
                print(json.dumps({k: row.get(k) for k in ("seed", "scenario", "candidate", "success", "failure_reason", "recovery_seconds")}), flush=True)
    equality = []
    choices = []
    for seed in seeds:
        for scenario in scenarios:
            group = [r for r in rows if r["seed"] == seed and r["scenario"] == scenario]
            states = [r.get("prefix_state") for r in group]
            matched = bool(all(s is not None for s in states) and all(
                all(np.array_equal(states[0][k], state[k]) for k in states[0]) for state in states[1:]))
            equality.append({"seed": seed, "scenario": scenario, "identical_prefix_state": matched})
            passed = [r for r in group if r["success"]]
            safe = [r for r in passed if not r["protection_violation"]]
            best = min(safe, key=lambda r: r["recovery_seconds"]) if safe else None
            choices.append({"seed": seed, "scenario": scenario,
                            "successful_candidates": [r["candidate"] for r in passed],
                            "safe_successful_candidates": [r["candidate"] for r in safe],
                            "minimum_cost_successful_candidate": best["candidate"] if best else None,
                            "beyond_continue_or_r1": bool(safe and not any(r["success"] and not r["protection_violation"] for r in group if r["candidate"] in ("continue", local_baseline)))})
    report = {"scope": "training/development physical candidate comparison, including executed robot-observation rules when present; no new test seeds",
              "seeds": seeds, "config": asdict(cfg), "episodes": len(rows),
              "local_baseline": local_baseline,
              "by_scenario": {s: {k: _summary([r for r in rows if r["scenario"] == s and r["candidate"] == k]) for k in skills} for s in scenarios},
              "by_skill": {k: _summary([r for r in rows if r["candidate"] == k]) for k in skills},
              "prefix_comparisons": equality, "all_prefixes_identical": all(r["identical_prefix_state"] for r in equality),
              "empirical_candidate_choices": choices,
              "additional_scope_cases": sum(c["beyond_continue_or_r1"] for c in choices),
              "oracle_candidate_upper_bound_success_rate": float(np.mean([bool(c["successful_candidates"]) for c in choices])),
              "safe_oracle_candidate_upper_bound_success_rate": float(np.mean([bool(c["safe_successful_candidates"]) for c in choices])),
              "contact_anchor_count": sum(r["contact_anchor_count"] for r in rows),
              "state_edit_count_after_initialization": sum(r["state_edit_count_after_initialization"] for r in rows)}
    (output / "matrix-report.json").write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2) + "\n")
    fields = ("seed", "scenario", "candidate", "success", "handover_completed", "skill_goal_achieved", "recovery_seconds", "recovery_path_m", "total_seconds_after_fault", "protection_violation", "failure_reason")
    with (output / "matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({k: r.get(k) for k in fields} for r in rows)
    return report


def validate_skills(seeds, output_dir, config=None, workers=4):
    cfg, output = config or MatrixConfig(), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(seed, "giver_loss", "R2", cfg) for seed in seeds]
    jobs += [(seed, "nominal", "R3", cfg) for seed in seeds]
    rows = []
    with (output / "standalone-trials.jsonl").open("w") as log:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for row in pool.map(_job, jobs):
                rows.append(row)
                log.write(json.dumps(_jsonable(row)) + "\n"); log.flush()
                print(json.dumps({k: row.get(k) for k in ("seed", "candidate", "success", "failure_reason", "recovery_seconds")}), flush=True)
    report = {"scope": "standalone skill validation on training seeds",
              "R2": _summary([r for r in rows if r["candidate"] == "R2"]),
              "R3": _summary([r for r in rows if r["candidate"] == "R3"])}
    (output / "standalone-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def _failure_origin(row):
    if row["success"]:
        return None
    before, after = row.get("post_fault_predicate", {}), row.get("post_recovery_predicate", {})
    if row["candidate"] == "R1" and before.get("receiver_supports") and not before.get("giver_holds"):
        return "R1_removed_the_only_existing_receiver_support"
    if row["candidate"] == "R2" and before.get("giver_holds") and not before.get("receiver_holds"):
        return "R2_removed_the_only_existing_giver_support"
    if row["candidate"] == "R3" and row.get("table_cycle_verified") and not after.get("receiver_holds"):
        return "R3_receiver_regrasp_failed_after_table_pickup_and_transport"
    if after and not after.get("receiver_supports"):
        return "receiver_not_supported_before_release_probe"
    return row.get("failure_reason")


def analyze_matrix(output_dir):
    output = Path(output_dir)
    report = json.loads((output / "matrix-report.json").read_text())
    rows = [json.loads(line) for line in (output / "trials.jsonl").read_text().splitlines()]
    scenarios, skills = tuple(report["by_scenario"]), tuple(report["by_skill"])
    local_baseline = report.get("local_baseline", "R1")
    for row in rows:
        row["failure_origin"] = _failure_origin(row)
    groups = {}
    for row in rows:
        groups.setdefault((row["seed"], row["scenario"]), {})[row["candidate"]] = row
    effects = {}
    for scenario in scenarios:
        scenario_rows = [g["continue"] for (_, s), g in groups.items() if s == scenario]
        prefixes = [r["post_fault_predicate"] for r in scenario_rows]
        deltas = []
        for row in scenario_rows:
            before = [p for p in row["trace"] if p["stage"] == "common_hold"][-1]
            after = [p for p in row["trace"] if p["phase"] == "prefix"][-1]
            deltas.append(np.asarray(after["robot"]["proprio20"][:6]).reshape(2, 3)
                          - np.asarray(before["robot"]["proprio20"][:6]).reshape(2, 3))
        effects[scenario] = {"scenes": len(prefixes),
                             "giver_holds_after_prefix": sum(p["giver_holds"] for p in prefixes),
                             "receiver_supports_after_prefix": sum(p["receiver_supports"] for p in prefixes),
                             "mean_actual_eef_delta_m": np.mean(deltas, axis=0).tolist()}
        if all("grasp_relative_change" in r for r in scenario_rows):
            effects[scenario]["giver_relative_translation_m"] = [r["grasp_relative_change"][0]["translation_norm_m"] for r in scenario_rows]
            effects[scenario]["giver_relative_rotation_deg"] = [r["grasp_relative_change"][0]["rotation_deg"] for r in scenario_rows]
    costs = {}
    for skill in (k for k in skills if k != "R3"):
        paired = [(g[skill], g["R3"]) for g in groups.values()
                  if all(g[k]["success"] and not g[k]["protection_violation"] for k in (skill, "R3"))]
        if paired:
            local_t = float(np.mean([a["recovery_seconds"] for a, _ in paired]))
            global_t = float(np.mean([b["recovery_seconds"] for _, b in paired]))
            local_p = float(np.mean([a["recovery_path_m"] for a, _ in paired]))
            global_p = float(np.mean([b["recovery_path_m"] for _, b in paired]))
            costs[skill] = {"matched_successful_scenes": len(paired),
                           "local_mean_seconds": local_t, "R3_mean_seconds": global_t,
                           "time_reduction_fraction": 1 - local_t / global_t,
                           "local_mean_path_m": local_p, "R3_mean_path_m": global_p,
                           "path_reduction_fraction": 1 - local_p / global_p}
        else:
            costs[skill] = {"matched_successful_scenes": 0}
    local_bound = sum(any(g[k]["success"] and not g[k]["protection_violation"] for k in ("continue", local_baseline))
                      for g in groups.values())
    report["fault_effects"] = effects
    report["continue_or_r1_safe_candidate_upper_bound"] = {"successes": local_bound, "scenes": len(groups)}
    report["continue_or_local_baseline_safe_upper_bound"] = {"baseline": local_baseline, "successes": local_bound, "scenes": len(groups)}
    report["R2_additional_safe_scenes"] = sum(
        g["R2"]["success"] and not g["R2"]["protection_violation"]
        and not any(g[k]["success"] and not g[k]["protection_violation"] for k in ("continue", local_baseline))
        for g in groups.values())
    report["R3_additional_safe_scenes"] = sum(
        g["R3"]["success"] and not g["R3"]["protection_violation"]
        and not any(g[k]["success"] and not g[k]["protection_violation"] for k in ("continue", local_baseline))
        for g in groups.values())
    oracle_rows = []
    for group in groups.values():
        feasible = [r for r in group.values() if r["success"] and not r["protection_violation"]]
        if feasible:
            oracle_rows.append(min(feasible, key=lambda r: r["recovery_seconds"]))
    report["safe_oracle_costs"] = {
        "successful_scenes": len(oracle_rows), "scenes": len(groups),
        "choice_counts": dict(Counter(r["candidate"] for r in oracle_rows)),
        "mean_recovery_seconds_successful": float(np.mean([r["recovery_seconds"] for r in oracle_rows])) if oracle_rows else None,
        "mean_recovery_path_m_successful": float(np.mean([r["recovery_path_m"] for r in oracle_rows])) if oracle_rows else None,
        "interpretation": "retrospective minimum-cost feasible candidate bound, distinct from the separately executed rule",
    }
    report["matched_success_costs_vs_R3"] = costs
    report["failure_origins_by_skill"] = {
        k: dict(Counter(r["failure_origin"] for r in rows if r["candidate"] == k and not r["success"])) for k in skills}
    report["R3_milestones"] = {key: sum(bool(r.get(key)) for r in rows if r["candidate"] == "R3")
                                for key in ("table_setdown_verified", "table_regrasp_and_lift_verified", "success")}
    report["matrix_interpretation"] = (
        "Forced physical candidates and actual sequential rule execution when present; the L1 image gate is not evaluated here. "
        f"The continue-or-{local_baseline} bound gives those two candidates perfect retrospective choice; "
        "new scope adds capacity only if R2/R3 succeeds where both fail. Candidate labels never drive a skill.")
    report["external_wrench_leak_steps"] = sum(r.get("external_wrench_leak_steps", 0) for r in rows)
    report["wrenches_cleared_before_recovery"] = all(r.get("external_wrench_at_recovery_start", 0) == 0 for r in rows)
    if report["config"].get("coupled_mode") == "shared_shift_receiver_loss":
        report["coupled_fault_interpretation"] = (
            "Physical shared translation followed by receiver opening/withdrawal; a compound common-carry failure. "
            "Evidence is relative to the frozen registered-waypoint R1. It does not establish that a table "
            "restart is necessary compared with an adaptive, retargeted local skill.")
    report["selector_investment"] = {
        "decision": ("INVALID_PREFIX_COMPARISON" if not report["all_prefixes_identical"] else
                     "PHYSICAL_SCOPE_HEADROOM_FOUND" if report["additional_scope_cases"] else "DEFER_VISUAL_SELECTOR"),
        "training_or_development_only": True,
        "reason": ("additional safe scenes beyond the stronger continue-or-R1 retrospective bound" if report["additional_scope_cases"] else
                   "no additional safe R2/R3 successes outside continue-or-local-baseline in this matrix"),
    }
    (output / "matrix-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    fields = ("seed", "scenario", "candidate", "success", "used_global_recovery", "handover_completed", "skill_goal_achieved",
              "table_setdown_verified", "table_regrasp_and_lift_verified", "recovery_seconds", "recovery_path_m",
              "total_seconds_after_fault", "protection_violation", "failure_reason", "failure_origin", "numerical_failure_stage")
    with (output / "matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({k: r.get(k) for k in fields} for r in rows)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    values = np.array([[report["by_scenario"][s][k]["success_rate"] for k in skills] for s in scenarios])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    im = ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu")
    skill_labels = {"continue": "Continue", "R1": "R1: receiver", "R1_retarget": "R1:\nretargeted", "R2": "R2:\ngiver", "R3": "R3:\ntable redo", "rule": "Rule"}
    ax.set_xticks(range(len(skills)), [skill_labels[k] for k in skills])
    coupled_label = ("Shifted carry\n+ receiver loss" if report["config"].get("coupled_mode") == "shared_shift_receiver_loss"
                     else "Coupled offset")
    scenario_labels = {"receiver_loss": "Receiver loss", "giver_loss": "Giver loss", "coupled_offset": coupled_label,
                       "object_sham": "Sham wrench", "object_slip": "Axial force",
                       "object_rotation_mild": "Mild pitch torque", "object_rotation": "Pitch torque"}
    ax.set_yticks(range(len(scenarios)), [scenario_labels[s] for s in scenarios])
    for i, s in enumerate(scenarios):
        for j, k in enumerate(skills):
            point = report["by_scenario"][s][k]
            ax.text(j, i, f"{point['successes']}/{point['episodes']}", ha="center", va="center",
                    color="white" if values[i, j] > 0.6 else "black", fontsize=12)
    ax.set_title(f"Seeds {min(report['seeds'])}-{max(report['seeds'])}: complete physical handover")
    fig.colorbar(im, ax=ax, label="Success rate")
    fig.tight_layout()
    fig.savefig(output / "success-matrix.png", dpi=180, bbox_inches="tight"); fig.savefig(output / "success-matrix.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, key, label in zip(axes, ("mean_recovery_seconds_successful", "mean_recovery_path_m_successful"),
                              ("Recovery time (s)", "Actual two-arm EEF path (m)")):
        points = [report["by_skill"][k] for k in skills]
        ax.bar(range(len(skills)), [p[key] or 0 for p in points], color=["#888888", "#4c78a8", "#54a24b", "#f58518", "#b279a2"])
        ax.set_xticks(range(len(skills)), ["R1\n(retarget)" if k == "R1_retarget" else k for k in skills])
        ax.set_ylabel(label)
        ax.set_title("Successful trials only")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "recovery-costs.png", dpi=180, bbox_inches="tight"); fig.savefig(output / "recovery-costs.pdf", bbox_inches="tight")
    plt.close(fig)
    return report


def freeze_baseline(output, baseline_model, config, active_recovery="R1"):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(baseline_model, output / "l1-calibrated.npz")
    verifier = LogisticVisualVerifier.load(output / "l1-calibrated.npz")
    frozen = {"l1_model_source": str(baseline_model), "l1_threshold": verifier.threshold,
              "active_recovery_baseline": active_recovery,
              "l1_proprio_dim": 20, "additional_logged_actual_finger_positions": 4,
              "r1": {"approach_steps": 16, "total_steps": 38, "contact_anchor": False},
              "r1_retarget": {"approach_steps": 16, "total_steps": 38,
                              "height_offset_m": config.r1_retarget_height_offset,
                              "target_source": "actual giver SE3 and registered robot relative waypoint"},
              "r3": {"placement_steps": config.r3_placement_steps,
                     "pickup_height_adjustment": config.r3_pickup_height_adjustment,
                     "receiver_height_offset": config.r3_receiver_height_offset,
                     "search_offsets": list(config.r3_search_offsets)},
              "rule": {"relative_gap_tolerance": config.rule_gap_tolerance,
                       "policy": "frozen local R1 then table R3 only if measured finger gap is inconsistent"},
              "acceptance": "giver fully open and detached, receiver stable for 1s; preserve release-support continuity as strict baseline criterion",
              "r3_goal": "verified table release plus physical pickup/lift plus complete handover; setdown alone is failure",
              "matrix_policy": "forced candidates, common physical release probe; L1 fixed but no perception-based candidate selection",
              "cost": "actual elapsed recovery sim time and sum of actual two-arm EEF path lengths",
              "protection": "no manual state edits/anchors; report neither-gripper-nor-table support >0.1s, release support gaps, deep inter-arm contacts (<-1cm)",
              "prefix_replay": "fresh independent initialization and identical actions; never restore checkpoint after initialization",
              "config": asdict(config)}
    (output / "frozen-baseline.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n")
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/l3-candidate-matrix-20261107")
    parser.add_argument("--baseline-model", default="outputs/l2-refinement-20261107/l1-calibrated.npz")
    parser.add_argument("--config", help="training-prototype MatrixConfig JSON")
    parser.add_argument("--validate-skills", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--retarget-ablation", action="store_true")
    parser.add_argument("--grasp-matrix", action="store_true")
    parser.add_argument("--include-rule", action="store_true", help="include rule execution in the grasp candidate matrix")
    parser.add_argument("--rule-validation", action="store_true")
    parser.add_argument("--reference-dir", help="paired training matrix; omit to execute local/global dev controls")
    parser.add_argument("--cohort", choices=("train", "dev"), default="train", help="grasp-matrix or rule-validation seed split")
    args = parser.parse_args()
    if args.analyze_only:
        report = analyze_matrix(args.output_dir)
        print(json.dumps(report["selector_investment"]))
        return
    cfg = MatrixConfig(**json.loads(Path(args.config).read_text())) if args.config else MatrixConfig()
    uses_frozen_retarget = args.grasp_matrix or args.rule_validation
    if uses_frozen_retarget and cfg.r1_retarget_height_offset != 0.0:
        raise ValueError("the frozen retargeted R1 requires zero height offset")
    freeze_baseline(args.output_dir, args.baseline_model, cfg, "R1_retarget" if uses_frozen_retarget else "R1")
    if args.rule_validation:
        from .physical_l2 import DEV_SEEDS
        seeds = range(args.seeds) if args.cohort == "train" else DEV_SEEDS
        report = run_rule_validation(seeds, args.output_dir, cfg, max(1, args.workers), args.reference_dir)
        print(json.dumps({"cohort": args.cohort, "rule": report["by_policy"]["rule"],
                          "prefixes_identical": report["all_prefixes_identical"]}))
        return
    if args.validate_skills:
        print(json.dumps(validate_skills(list(range(args.seeds)), args.output_dir, cfg, max(1, args.workers))))
        return
    if args.retarget_ablation:
        report = run_retarget_ablation(range(args.seeds), args.output_dir, cfg, max(1, args.workers))
        print(json.dumps({"episodes": report["episodes"], "R3_additional_safe_scenes": report["R3_additional_safe_scenes"]}))
        return
    if args.grasp_matrix:
        from .physical_l2 import DEV_SEEDS
        seeds = range(args.seeds) if args.cohort == "train" else DEV_SEEDS
        skills = (*GRASP_SKILLS, "rule") if args.include_rule else GRASP_SKILLS
        report = run_matrix(seeds, args.output_dir, cfg, max(1, args.workers), GRASP_SCENARIOS, skills)
    else:
        report = run_matrix(range(args.seeds), args.output_dir, cfg, max(1, args.workers))
    report = analyze_matrix(args.output_dir)
    print(json.dumps({"episodes": report["episodes"], "prefixes_identical": report["all_prefixes_identical"],
                      "additional_scope_cases": report["additional_scope_cases"]}))


if __name__ == "__main__":
    main()
