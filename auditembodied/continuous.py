"""Robot-only continuous trajectories and evaluator labels for L0/L1.

The trajectory file is deliberately a small, frozen interface.  Its episode
groups contain only ``rgb``, ``proprio``, ``actions`` and ``stages``.  Anything
that depends on the simulated object (contact predicates, object pose, failure
times and R1 outcomes) is written to a sidecar evaluator file.  Keeping that
sidecar separate makes it difficult for a training loader to accidentally use
privileged object state.

Collection and online evaluation share the validated physical OSC protocol.
One-time initialization is followed by actuator-only receiver withdrawal,
recovery, giver release and a one-second hold. Historical impulse/anchor
helpers remain available for old diagnostic reports.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .l0 import CAMERAS, L0Config, OracleVerifier, TwoArmL0Env, _jsonable


# These names are part of the data contract. Bump the version if a field is
# added to the robot stream; do not silently append privileged object values.
TRAJECTORY_SCHEMA = "auditembodied.l0.robot_only.v2"
EVALUATOR_SCHEMA = "auditembodied.l0.evaluator.v1"
ROBOT_PROPRIO_DIM = 20  # two EEF positions (6) + two 7-DoF arm states (14)
ACTION_DIM = 14          # two OSC position/orientation/gripper commands
TRAJECTORY_FIELDS = ("rgb", "proprio", "actions", "stages")


def evaluator_path(path: os.PathLike[str] | str) -> Path:
    """Return the sidecar path associated with a robot-only trajectory."""

    output = Path(path)
    return output.with_suffix(".evaluator.h5")


@dataclass(frozen=True)
class TrajectoryConfig:
    """Settings for physical collection and historical impulse diagnostics.

    Current collection uses receiver opening/withdrawal and applies zero body
    force. The force_* fields are retained only for legacy impulse helpers;
    they do not describe the current collection perturbation.
    """

    camera_height: int = 64
    camera_width: int = 64
    phase_steps: int = 5
    capture_stride: int = 1
    force_duration_steps: int = 6
    force_base: float = 350.0
    force_direction: Tuple[float, float, float] = (0.12, 0.0, -1.0)
    force_direction_jitter: float = 0.0
    r1_steps: int = 38
    r1_approach_steps: int = 16
    # Current collection and evaluation always disable both mechanisms.
    r1_use_contact_anchor: bool = False
    allow_safety_fallback: bool = False
    max_seeds: int = 50
    output: str = "outputs/l0-trajectory-50-robot-only.h5"
    render_frame_dir: Optional[str] = None
    # Lightweight image-side nuisance variation. These are applied after
    # capture and never change the evaluator labels.
    unrelated_changes: bool = True
    occlusion_probability: float = 0.20
    camera_jitter_pixels: int = 1
    appearance_jitter: float = 0.04
    random_seed: int = 20261015
    # Refuse to publish a full data set whose fixed protocol has an unusable R1.
    r1_min_success: float = 0.90
    require_r1_stability: bool = False


@dataclass
class TrajectoryEpisode:
    """In-memory episode with an explicit evaluator-only part.

    ``labels`` and ``predicates`` are retained while collecting so the
    evaluator sidecar can be written atomically. They are never written to the
    robot-only HDF5 file.
    """

    seed: int
    perturbation: str
    actions: List[np.ndarray] = field(default_factory=list)
    rgb: List[np.ndarray] = field(default_factory=list)
    proprio: List[np.ndarray] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)  # evaluator-only
    predicates: List[Dict[str, Any]] = field(default_factory=list)  # evaluator-only
    stages: List[str] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)  # evaluator-only
    failure_events: List[Dict[str, Any]] = field(default_factory=list)  # evaluator-only
    state_edits: List[Dict[str, Any]] = field(default_factory=list)
    contact_anchors: List[Dict[str, Any]] = field(default_factory=list)
    safety_fallbacks: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluator_metadata: Dict[str, Any] = field(default_factory=dict)

    def arrays(self) -> Dict[str, np.ndarray]:
        """Return only the four robot stream arrays in the frozen contract."""

        actions = np.asarray(self.actions, dtype=np.float32)
        rgb = np.asarray(self.rgb, dtype=np.uint8)
        proprio = np.asarray(self.proprio, dtype=np.float32)
        if not self.actions and not self.rgb and not self.proprio and not self.stages:
            return {
                "actions": np.zeros((0, ACTION_DIM), dtype=np.float32),
                "rgb": np.zeros((0, len(CAMERAS), 0, 0, 3), dtype=np.uint8),
                "proprio": np.zeros((0, ROBOT_PROPRIO_DIM), dtype=np.float32),
            }
        if actions.ndim != 2 or (len(actions) and actions.shape[1] != ACTION_DIM):
            raise ValueError(f"actions must have shape [T,{ACTION_DIM}], got {actions.shape}")
        if proprio.ndim != 2 or (len(proprio) and proprio.shape[1] != ROBOT_PROPRIO_DIM):
            raise ValueError(
                f"robot-only proprio must have shape [T,{ROBOT_PROPRIO_DIM}], got {proprio.shape}"
            )
        if len(rgb) != len(actions) or len(proprio) != len(actions) or len(self.stages) != len(actions):
            raise ValueError("rgb, proprio, actions and stages must have the same T")
        return {"actions": actions, "rgb": rgb, "proprio": proprio}

    def add_state_edit(self, kind: str, **details: Any) -> None:
        """Record a state edit without recording object truth values."""

        self.state_edits.append(
            {
                "kind": str(kind),
                "sim_time": float(details.pop("sim_time", 0.0)),
                **{str(k): _jsonable(v) for k, v in details.items()},
            }
        )


class ContinuousActionExpert:
    """Small OSC waypoint expert with explicit continuous action traces."""

    name = "continuous_waypoint_expert"

    def __init__(self, phase_steps: int = 5, rng: Optional[np.random.Generator] = None):
        self.phase_steps = max(1, int(phase_steps))
        self.rng = rng or np.random.default_rng(0)

    @staticmethod
    def action_towards(
        env: TwoArmL0Env,
        targets: Sequence[Sequence[float]],
        grippers: Sequence[float] = (1.0, 1.0),
        gain: float = 0.65,
    ) -> np.ndarray:
        """Build one normalized OSC delta action for both opposed Panda arms."""

        action = np.zeros(env.action_dim, dtype=np.float64)
        if env.action_dim != ACTION_DIM:
            raise ValueError(f"unexpected environment action dimension: {env.action_dim}")
        for robot_index, target in enumerate(targets):
            target_position = np.asarray(target, dtype=np.float64)
            delta = np.clip(
                gain * (target_position - env._eef_pos(robot_index)) / 0.05,
                -1.0,
                1.0,
            )
            start = robot_index * 7
            action[start : start + 3] = delta
            action[start + 3 : start + 6] = 0.0
            action[start + 6] = float(grippers[robot_index])
        return action

    @staticmethod
    def _set_contact_state(
        env: TwoArmL0Env,
        mode: str,
        target: np.ndarray,
        episode: Optional[TrajectoryEpisode] = None,
        stage: str = "",
    ) -> None:
        """Stabilize a contact state after a controller step and log the edit."""

        if mode == "both":
            # The object anchor itself is not serialized in the robot stream;
            # only the fact and reason for the edit are recorded.
            env._set_hammer_pose(target)
            for robot in env.raw.robots:
                env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes["right"]] = 0.0
            env.sync_gripper_controller(0, closed=True)
            env.sync_gripper_controller(1, closed=True)
            anchor_role = "shared_handover"
        elif mode == "receiver":
            receiver_target = np.asarray(target, dtype=np.float64) + env._initial_handle_offset
            env._set_hammer_pose(receiver_target)
            giver = env.raw.robots[0]
            receiver = env.raw.robots[1]
            env.sim.data.qpos[giver._ref_gripper_joint_pos_indexes["right"]] = giver.gripper["right"].init_qpos
            env.sim.data.qpos[receiver._ref_gripper_joint_pos_indexes["right"]] = 0.0
            env.sync_gripper_controller(0, closed=False)
            env.sync_gripper_controller(1, closed=True)
            anchor_role = "receiver_regrasp"
        else:
            raise ValueError(f"unknown contact mode: {mode}")
        env.sim.data.qvel[:] = 0.0
        env.sim.forward()
        if episode is not None:
            sim_time = float(env.sim.data.time)
            episode.contact_anchors.append(
                {
                    "stage": str(stage),
                    "mode": str(mode),
                    "role": anchor_role,
                    "sim_time": sim_time,
                    "target_frame": "receiver_eef" if mode == "receiver" else "shared_eef",
                }
            )
            episode.add_state_edit(
                "contact_anchor",
                sim_time=sim_time,
                mode=mode,
                role=anchor_role,
                stage=stage,
                entities=["gripper_state", "support_state"],
            )

    def execute_phase(
        self,
        env: TwoArmL0Env,
        episode: TrajectoryEpisode,
        stage: str,
        targets: Sequence[Sequence[float]],
        contact_mode: Optional[str],
        grippers: Sequence[float] = (1.0, 1.0),
        gain: float = 0.65,
        steps: Optional[int] = None,
        capture: Optional[Any] = None,
    ) -> None:
        for _ in range(self.phase_steps if steps is None else max(1, int(steps))):
            action = self.action_towards(env, targets, grippers=grippers, gain=gain)
            env.step(action)
            if contact_mode is not None:
                contact_target = targets[1] if contact_mode == "receiver" else targets[0]
                self._set_contact_state(
                    env, contact_mode, np.asarray(contact_target, dtype=np.float64), episode, stage
                )
            if capture is not None:
                capture(episode, env, stage, action)


class ContinuousR1Recovery:
    """Receiver-side OSC recovery with no hidden position-only finalizer."""

    name = "continuous_receiver_regrasp"

    def __init__(self, use_contact_anchor: bool = True, approach_steps: int = 8):
        self.use_contact_anchor = bool(use_contact_anchor)
        self.approach_steps = max(0, int(approach_steps))

    def run(
        self,
        env: TwoArmL0Env,
        expert: ContinuousActionExpert,
        episode: TrajectoryEpisode,
        capture: Any,
        steps: Optional[int] = None,
        receiver_target: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        giver_reset = env._eef_pos(0).copy()
        if self.use_contact_anchor:
            # Preserve the historical anchored diagnostic path.
            env.sim.forward()
            current_handle = np.asarray(env.raw._handle_xpos, dtype=np.float64).copy()
            receiver_target = current_handle - env._initial_handle_offset
        elif receiver_target is None:
            targets = env.bootstrap_info.get("initial_grasp_targets", [])
            if len(targets) != 2:
                raise ValueError("unanchored R1 requires a registered receiver grasp target")
            receiver_target = targets[1]
        receiver_target = np.asarray(receiver_target, dtype=np.float64).copy()
        control_steps = max(expert.phase_steps, 20) if steps is None else max(1, int(steps))
        if self.use_contact_anchor:
            expert.execute_phase(
                env,
                episode,
                "r1_recovery_action",
                (giver_reset, receiver_target),
                contact_mode="receiver",
                grippers=(-1.0, 1.0),
                gain=1.0,
                steps=control_steps,
                capture=capture,
            )
            # Retained only for the historical anchored baseline. It is
            # explicitly disabled in the no-anchor audit path below.
            final_receiver_target = env._eef_pos(1).copy()
            expert._set_contact_state(
                env,
                "receiver",
                final_receiver_target,
                episode,
                "r1_contact_anchor",
            )
            if capture is not None:
                capture(episode, env, "r1_recovery_anchor", None)
        else:
            # Reapproach with the receiver open, then close and settle while
            # the giver continues supporting the object. The registered
            # pre-fault robot waypoint avoids reading privileged object pose.
            approach_steps = min(self.approach_steps, max(0, control_steps - 1))
            if approach_steps:
                expert.execute_phase(
                    env,
                    episode,
                    "r1_recovery_approach",
                    (giver_reset, receiver_target),
                    contact_mode=None,
                    grippers=(1.0, -1.0),
                    gain=1.0,
                    steps=approach_steps,
                    capture=capture,
                )
            action_steps = max(1, control_steps - approach_steps)
            expert.execute_phase(
                env,
                episode,
                "r1_recovery_action",
                (giver_reset, receiver_target),
                contact_mode=None,
                grippers=(1.0, 1.0),
                gain=1.0,
                steps=action_steps,
                capture=capture,
            )
        predicate = OracleVerifier().evaluate(env)
        result = {
            "skill": self.name,
            "success": bool(predicate["receiver_supports"]),
            "success_scope": "support_reacquired_before_giver_release",
            "handover_success": None,
            "predicate": predicate,
            "contact_finalizer": False,
            "position_only_finalizer": False,
            "contact_anchor_steps": control_steps if self.use_contact_anchor else 0,
            "control_steps": control_steps,
            "approach_steps": int(min(self.approach_steps, max(0, control_steps - 1))) if not self.use_contact_anchor else 0,
            "use_contact_anchor": bool(self.use_contact_anchor),
            "target_source": "legacy_handle_state" if self.use_contact_anchor else "registered_robot_waypoint",
        }
        episode.evaluator_metadata["r1"] = _jsonable(result)
        return result


def _proprio(env: TwoArmL0Env) -> np.ndarray:
    """Return the 20-D robot-only state used by the online verifier."""

    arm_qpos = [robot.get_robot_joint_positions() for robot in env.raw.robots]
    result = np.concatenate(
        [env._eef_pos(0), env._eef_pos(1), np.asarray(arm_qpos[0]), np.asarray(arm_qpos[1])]
    ).astype(np.float32)
    if result.shape != (ROBOT_PROPRIO_DIM,):
        raise ValueError(f"expected {ROBOT_PROPRIO_DIM}-D robot proprioception, got {result.shape}")
    return result


def _apply_rgb_nuisance(
    frame_bundle: np.ndarray,
    rng: np.random.Generator,
    config: TrajectoryConfig,
) -> np.ndarray:
    """Add nuisance variation without changing the evaluator label."""

    frames = np.asarray(frame_bundle, dtype=np.uint8).copy()
    if not config.unrelated_changes:
        return frames
    for view in range(frames.shape[0]):
        if config.camera_jitter_pixels:
            dy = int(rng.integers(-config.camera_jitter_pixels, config.camera_jitter_pixels + 1))
            dx = int(rng.integers(-config.camera_jitter_pixels, config.camera_jitter_pixels + 1))
            frames[view] = np.roll(np.roll(frames[view], dy, axis=0), dx, axis=1)
        if config.appearance_jitter > 0:
            gain = float(1.0 + rng.normal(0.0, config.appearance_jitter))
            frames[view] = np.clip(frames[view].astype(np.float32) * gain, 0.0, 255.0).astype(np.uint8)
        if config.occlusion_probability > 0 and float(rng.random()) < config.occlusion_probability:
            height, width = frames.shape[1:3]
            box_h = max(2, int(height * 0.12))
            box_w = max(2, int(width * 0.16))
            top = int(rng.integers(0, max(1, height - box_h + 1)))
            left = int(rng.integers(0, max(1, width - box_w + 1)))
            frames[view, top : top + box_h, left : left + box_w] = 0
    return frames


def _force_vector(config: TrajectoryConfig, rng: np.random.Generator) -> np.ndarray:
    direction = np.asarray(config.force_direction, dtype=np.float64)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)) or np.linalg.norm(direction) == 0:
        raise ValueError("force_direction must be a finite non-zero 3-vector")
    direction = direction / np.linalg.norm(direction)
    if config.force_direction_jitter:
        direction = direction + rng.normal(0.0, config.force_direction_jitter, size=3)
        direction = direction / max(1e-8, np.linalg.norm(direction))
    return direction * float(config.force_base)


def _hold_action(env: TwoArmL0Env) -> np.ndarray:
    """Keep both EEFs and grippers under the OSC controller during impulses."""

    return ContinuousActionExpert.action_towards(
        env,
        (env._eef_pos(0), env._eef_pos(1)),
        grippers=(1.0, 1.0),
        gain=0.0,
    )


def _state_is_safe(env: TwoArmL0Env) -> bool:
    qpos = np.asarray(env.sim.data.qpos, dtype=np.float64)
    if not np.all(np.isfinite(qpos)):
        return False
    object_position = qpos[env._hammer_qpos_index : env._hammer_qpos_index + 3]
    # The handover table is around z=.8 and the reachable workspace is compact.
    return bool(np.all(np.abs(object_position[:2]) < 1.0) and -0.05 < object_position[2] < 2.0)


def _record_failure_event(episode: TrajectoryEpisode) -> None:
    """Record every observed supported-to-unsupported transition."""

    if len(episode.labels) < 2:
        return
    labels = np.asarray(episode.labels, dtype=np.int8)
    known = {
        int(event.get("frame_index", -1))
        for event in episode.failure_events
        if event.get("kind") == "support_failure"
    }
    for index in np.flatnonzero((labels[:-1] == 1) & (labels[1:] == 0)) + 1:
        index = int(index)
        if index in known:
            continue
        episode.failure_events.append(
            {
                "kind": "support_failure",
                "frame_index": index,
                "stage": str(episode.stages[index]),
                "sim_time": float(episode.timestamps[index]),
                "source": "evaluator_predicate_transition",
            }
        )
        known.add(index)


def collect_episode(
    seed: int,
    perturbation: str,
    config: Optional[TrajectoryConfig] = None,
    output_frame_dir: Optional[str] = None,
) -> TrajectoryEpisode:
    """Collect the same no-anchor physical protocol used by online evaluation."""
    from dataclasses import replace
    from .r1_validation import run_physical_trial

    cfg = config or TrajectoryConfig()
    if output_frame_dir is not None:
        cfg = replace(cfg, render_frame_dir=output_frame_dir)
    _, episode = run_physical_trial(seed, perturbation, render=True, config=cfg)
    return episode


def _normalise_episode_for_write(episode: TrajectoryEpisode) -> Tuple[TrajectoryEpisode, bool, int]:
    """Normalize a legacy all-23-D in-memory object without mixing schemas."""

    original = np.asarray(episode.proprio)
    if not episode.proprio:
        return episode, False, ROBOT_PROPRIO_DIM
    if original.ndim != 2 or original.shape[1] < ROBOT_PROPRIO_DIM:
        raise ValueError(f"proprio must be [T,>=20], got {original.shape}")
    input_dim = int(original.shape[1])
    if input_dim == ROBOT_PROPRIO_DIM:
        return episode, False, input_dim
    # A synthetic/legacy caller may still construct a 23-D object. Convert
    # the entire file to v2 robot-only before writing; never store the three
    # object coordinates or mix episodes of different dimensions.
    converted = TrajectoryEpisode(
        seed=episode.seed,
        perturbation=episode.perturbation,
        actions=list(episode.actions),
        rgb=list(episode.rgb),
        proprio=[np.asarray(row, dtype=np.float32)[:ROBOT_PROPRIO_DIM].copy() for row in episode.proprio],
        labels=list(episode.labels),
        predicates=list(episode.predicates),
        stages=list(episode.stages),
        timestamps=list(episode.timestamps),
        failure_events=list(episode.failure_events),
        state_edits=list(episode.state_edits),
        contact_anchors=list(episode.contact_anchors),
        safety_fallbacks=list(episode.safety_fallbacks),
        metadata=dict(episode.metadata),
        evaluator_metadata=dict(episode.evaluator_metadata),
    )
    converted.metadata["legacy_proprio_converted"] = True
    converted.metadata["legacy_input_proprio_dim"] = input_dim
    return converted, True, input_dim


def write_hdf5(episodes: Sequence[TrajectoryEpisode], path: os.PathLike[str] | str) -> Dict[str, Any]:
    """Write a robot-only HDF5 file and its evaluator-only sidecar."""

    import h5py

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized: List[TrajectoryEpisode] = []
    input_dims: set[int] = set()
    for episode in episodes:
        item, _, input_dim = _normalise_episode_for_write(episode)
        normalized.append(item)
        input_dims.add(input_dim)
    if len(input_dims) > 1:
        raise ValueError(
            "mixed proprioception dimensions are forbidden in one main experiment: "
            f"{sorted(input_dims)}"
        )
    sidecar = evaluator_path(output)
    with h5py.File(output, "w") as handle:
        handle.attrs["schema"] = TRAJECTORY_SCHEMA
        handle.attrs["camera_names"] = json.dumps(list(CAMERAS))
        handle.attrs["episode_count"] = len(normalized)
        handle.attrs["trajectory_fields"] = json.dumps(list(TRAJECTORY_FIELDS))
        handle.attrs["proprio_dim"] = ROBOT_PROPRIO_DIM
        handle.attrs["action_dim"] = ACTION_DIM
        groups = handle.create_group("episodes")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        for index, episode in enumerate(normalized):
            group = groups.create_group(f"episode_{index:05d}")
            arrays = episode.arrays()
            group.create_dataset("rgb", data=arrays["rgb"], compression="gzip", compression_opts=4, shuffle=True)
            group.create_dataset("actions", data=arrays["actions"], compression="gzip", compression_opts=4, shuffle=True)
            group.create_dataset("proprio", data=arrays["proprio"], compression="gzip", compression_opts=4, shuffle=True)
            group.create_dataset("stages", data=np.asarray(episode.stages, dtype=object), dtype=string_dtype)
            group.attrs["seed"] = int(episode.seed)
            group.attrs["perturbation"] = episode.perturbation
            # State edits and anchors are protocol metadata, not evaluator truth.
            metadata = dict(episode.metadata)
            forbidden_metadata = {
                key
                for key in metadata
                if str(key).lower() in {
                    "labels",
                    "predicates",
                    "object_pose",
                    "object_truth",
                    "hammer_pose",
                    "hammer_xyz",
                }
            }
            if forbidden_metadata:
                raise ValueError(
                    "object truth cannot be written to the robot stream metadata: "
                    + ", ".join(sorted(forbidden_metadata))
                )
            metadata["state_edits"] = _jsonable(episode.state_edits)
            metadata["contact_anchors"] = _jsonable(episode.contact_anchors)
            metadata["safety_fallbacks"] = _jsonable(episode.safety_fallbacks)
            group.attrs["metadata"] = json.dumps(metadata, ensure_ascii=False)

    with h5py.File(sidecar, "w") as handle:
        handle.attrs["schema"] = EVALUATOR_SCHEMA
        handle.attrs["trajectory"] = str(output)
        handle.attrs["episode_count"] = len(normalized)
        groups = handle.create_group("episodes")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        for index, episode in enumerate(normalized):
            group = groups.create_group(f"episode_{index:05d}")
            group.create_dataset("labels", data=np.asarray(episode.labels, dtype=np.int8), compression="gzip", shuffle=True)
            group.create_dataset("timestamps", data=np.asarray(episode.timestamps, dtype=np.float64), compression="gzip", shuffle=True)
            group.create_dataset(
                "predicates",
                data=np.asarray([json.dumps(_jsonable(p), ensure_ascii=False) for p in episode.predicates], dtype=object),
                dtype=string_dtype,
            )
            group.create_dataset(
                "failure_events",
                data=np.asarray([json.dumps(_jsonable(p), ensure_ascii=False) for p in episode.failure_events], dtype=object),
                dtype=string_dtype,
            )
            group.create_dataset(
                "failure_event_times",
                data=np.asarray(
                    [float(event.get("sim_time", np.nan)) for event in episode.failure_events],
                    dtype=np.float64,
                ),
                compression="gzip",
                shuffle=True,
            )
            group.attrs["seed"] = int(episode.seed)
            group.attrs["perturbation"] = episode.perturbation
            group.attrs["metadata"] = json.dumps(_jsonable(episode.evaluator_metadata), ensure_ascii=False)

    frames = int(sum(len(ep.rgb) for ep in normalized))
    labels = [label for ep in normalized for label in ep.labels]
    return {
        "schema": TRAJECTORY_SCHEMA,
        "evaluator_schema": EVALUATOR_SCHEMA,
        "path": str(output),
        "evaluator_path": str(sidecar),
        "episodes": len(normalized),
        "frames": frames,
        "proprio_dim": ROBOT_PROPRIO_DIM,
        "action_dim": ACTION_DIM,
        "positive_frame_rate": float(np.mean(labels)) if labels else 0.0,
        "legacy_input_proprio_dim": sorted(input_dims)[0] if input_dims and sorted(input_dims)[0] != ROBOT_PROPRIO_DIM else None,
    }


def validate_protocol(path: os.PathLike[str] | str) -> Dict[str, Any]:
    """Audit the frozen main/sidecar pair before it is used by L1."""

    import h5py

    source = Path(path)
    sidecar = evaluator_path(source)
    report: Dict[str, Any] = {
        "schema": "auditembodied.l0.protocol_integrity.v1",
        "trajectory": str(source),
        "evaluator": str(sidecar),
        "errors": [],
        "episodes": 0,
        "seeds": [],
        "frames": 0,
        "proprio_dim": ROBOT_PROPRIO_DIM,
        "action_dim": ACTION_DIM,
    }
    with h5py.File(source, "r") as main:
        if str(main.attrs.get("schema", "")) != TRAJECTORY_SCHEMA:
            report["errors"].append("main schema is not robot_only.v2")
        groups = main.get("episodes")
        if groups is None:
            report["errors"].append("main file has no episodes")
            return report
        report["episodes"] = len(groups)
        report["seeds"] = sorted({int(group.attrs.get("seed", -1)) for group in groups.values()})
        with h5py.File(sidecar, "r") as evaluator:
            if str(evaluator.attrs.get("schema", "")) != EVALUATOR_SCHEMA:
                report["errors"].append("evaluator schema mismatch")
            for name, group in groups.items():
                forbidden = {"labels", "predicates", "timestamps", "failure_events"}.intersection(group.keys())
                if forbidden:
                    report["errors"].append(f"privileged fields in main/{name}: {sorted(forbidden)}")
                required = set(TRAJECTORY_FIELDS)
                if set(group.keys()) != required:
                    report["errors"].append(f"main/{name} fields are {sorted(group.keys())}")
                rgb = group["rgb"]
                proprio = group["proprio"]
                actions = group["actions"]
                stages = group["stages"]
                report["frames"] += int(len(stages))
                if proprio.ndim != 2 or proprio.shape[1] != ROBOT_PROPRIO_DIM:
                    report["errors"].append(f"main/{name} proprio shape {proprio.shape}")
                if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
                    report["errors"].append(f"main/{name} action shape {actions.shape}")
                if len(rgb) != len(proprio) or len(actions) != len(stages):
                    report["errors"].append(f"main/{name} stream lengths disagree")
                if name not in evaluator.get("episodes", {}):
                    report["errors"].append(f"evaluator missing {name}")
                else:
                    ev = evaluator["episodes"][name]
                    if len(ev["labels"]) != len(stages) or len(ev["timestamps"]) != len(stages):
                        report["errors"].append(f"evaluator/{name} label/time length mismatch")
                    if "failure_event_times" in ev and len(ev["failure_event_times"]) != len(ev.get("failure_events", [])):
                        report["errors"].append(f"evaluator/{name} failure event time mismatch")
    report["ok"] = not report["errors"]
    return report


def _r1_trial(
    seed: int,
    force_base: float,
    duration: int,
    direction: Sequence[float],
    *,
    r1_steps: int = 32,
    r1_approach_steps: int = 16,
    use_contact_anchor: bool = False,
    allow_safety_fallback: bool = False,
) -> Dict[str, Any]:
    """Run one no-render physical R1 trial for the protocol scan.

    The default scan path has no contact anchor and no synthetic table
    fallback.  This makes the reported R1 rate a physical execution result;
    callers can enable the historical anchor/fallback path explicitly for a
    comparison.
    """

    env = TwoArmL0Env(seed=int(seed), config=L0Config(render=False, horizon=300, expert_mode="continuous"))
    verifier = OracleVerifier()
    try:
        env.reset()
        env.bootstrap_handover()
        initial = verifier.evaluate(env)
        direction_array = np.asarray(direction, dtype=np.float64)
        direction_array /= max(1e-8, np.linalg.norm(direction_array))
        env.apply_body_force(
            direction_array * float(force_base),
            int(duration),
            action=_hold_action(env),
        )
        disturbed = verifier.evaluate(env)
        unstable = not _state_is_safe(env)
        physical_failure = not bool(disturbed["receiver_supports"])
        safety_fallback = False
        if not unstable and not physical_failure and allow_safety_fallback:
            env.place_object_on_table()
            disturbed = verifier.evaluate(env)
            safety_fallback = True
        failed = not bool(disturbed["receiver_supports"])
        recovery = None
        if failed and not unstable:
            recovery = ContinuousR1Recovery(
                use_contact_anchor=use_contact_anchor,
                approach_steps=int(r1_approach_steps),
            ).run(
                env,
                ContinuousActionExpert(phase_steps=5),
                TrajectoryEpisode(int(seed), "late"),
                lambda *args: None,
                steps=int(r1_steps),
            )
        return {
            "seed": int(seed),
            "initial_support": bool(initial["receiver_supports"]),
            "disturbance_support": bool(disturbed["receiver_supports"]),
            "physical_failure": bool(physical_failure),
            "safety_fallback": bool(safety_fallback),
            "failure_created": bool(failed),
            "unstable": bool(unstable),
            "r1_success": bool(recovery and recovery.get("success", False)),
            "r1_use_contact_anchor": bool(use_contact_anchor),
            "allow_safety_fallback": bool(allow_safety_fallback),
            "r1_steps": int(r1_steps),
            "r1_approach_steps": int(r1_approach_steps),
            "object_height": float(disturbed["object_height"]),
            "relative_speed": float(disturbed["relative_speed"]),
        }
    except Exception as exc:
        return {
            "seed": int(seed),
            "initial_support": False,
            "disturbance_support": False,
            "failure_created": False,
            "unstable": True,
            "r1_success": False,
            "error": repr(exc),
        }
    finally:
        env.close()


def scan_r1_protocol(
    seeds: Iterable[int] = range(10),
    force_bases: Sequence[float] = (150.0, 200.0, 250.0, 300.0, 350.0, 500.0, 700.0, 900.0),
    durations: Sequence[int] = (2, 4, 6, 8, 12),
    direction: Sequence[float] = (0.12, 0.0, -1.0),
    output: os.PathLike[str] | str = "outputs/r1-protocol-scan-no-anchor.json",
    min_failure_rate: float = 0.50,
    min_physical_failure_rate: float = 0.40,
    max_safety_fallback_rate: float = 0.20,
    min_r1_success: float = 0.60,
    max_unstable_rate: float = 0.05,
    r1_steps: int = 32,
    r1_approach_steps: int = 16,
    use_contact_anchor: bool = False,
    allow_safety_fallback: bool = False,
) -> Dict[str, Any]:
    """Scan physical R1 protocols before collecting a new data set."""

    seed_list = [int(seed) for seed in seeds]
    rows: List[Dict[str, Any]] = []
    selected: Optional[Dict[str, Any]] = None
    for force_base in force_bases:
        for duration in durations:
            trials = [
                _r1_trial(
                    seed,
                    float(force_base),
                    int(duration),
                    direction,
                    r1_steps=int(r1_steps),
                    r1_approach_steps=int(r1_approach_steps),
                    use_contact_anchor=bool(use_contact_anchor),
                    allow_safety_fallback=bool(allow_safety_fallback),
                )
                for seed in seed_list
            ]
            failure_rate = float(np.mean([row["failure_created"] for row in trials])) if trials else 0.0
            unstable_rate = float(np.mean([row["unstable"] for row in trials])) if trials else 0.0
            physical_failure_rate = float(np.mean([row.get("physical_failure", False) for row in trials])) if trials else 0.0
            fallback_rate = float(np.mean([row.get("safety_fallback", False) for row in trials])) if trials else 0.0
            failed_trials = [row for row in trials if row.get("physical_failure", False) and not row["unstable"]]
            r1_success = float(np.mean([row["r1_success"] for row in failed_trials])) if failed_trials else 0.0
            candidate = {
                "force_base": float(force_base),
                "duration_steps": int(duration),
                "direction": [float(value) for value in direction],
                "seeds": seed_list,
                "failure_rate": failure_rate,
                "physical_failure_rate": physical_failure_rate,
                "safety_fallback_rate": fallback_rate,
                "unstable_rate": unstable_rate,
                "r1_success_rate_conditional": r1_success,
                "r1_steps": int(r1_steps),
                "r1_approach_steps": int(r1_approach_steps),
                "r1_use_contact_anchor": bool(use_contact_anchor),
                "allow_safety_fallback": bool(allow_safety_fallback),
                "trials": trials,
            }
            rows.append(candidate)
            if (
                selected is None
                and failure_rate >= float(min_failure_rate)
                and physical_failure_rate >= float(min_physical_failure_rate)
                and fallback_rate <= float(max_safety_fallback_rate)
                and r1_success >= float(min_r1_success)
                and unstable_rate <= float(max_unstable_rate)
            ):
                selected = candidate
    if selected is None and rows:
        # Report the closest candidate, but do not pretend it passed the gate.
        selected = min(
            rows,
            key=lambda row: (
                max(0.0, float(min_failure_rate) - row["failure_rate"]),
                max(0.0, float(min_physical_failure_rate) - row["physical_failure_rate"]),
                max(0.0, row["safety_fallback_rate"] - float(max_safety_fallback_rate)),
                max(0.0, float(min_r1_success) - row["r1_success_rate_conditional"]),
                row["unstable_rate"],
            ),
        )
    result = {
        "schema": "auditembodied.l0.r1_scan.v2",
        "recovery_mode": "contact_anchor" if use_contact_anchor else "no_contact_anchor",
        "fallback_mode": "enabled" if allow_safety_fallback else "disabled",
        "protocol_frozen": bool(
            selected
            and selected["failure_rate"] >= float(min_failure_rate)
            and selected["physical_failure_rate"] >= float(min_physical_failure_rate)
            and selected["safety_fallback_rate"] <= float(max_safety_fallback_rate)
            and selected["r1_success_rate_conditional"] >= float(min_r1_success)
            and selected["unstable_rate"] <= float(max_unstable_rate)
        ),
        "criteria": {
            "min_failure_rate": float(min_failure_rate),
            "min_physical_failure_rate": float(min_physical_failure_rate),
            "max_safety_fallback_rate": float(max_safety_fallback_rate),
            "min_r1_success": float(min_r1_success),
            "max_unstable_rate": float(max_unstable_rate),
        },
        "selected": selected,
        "candidates": rows,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["output"] = str(output_path)
    return result


def collect_dataset(
    seeds: Iterable[int],
    perturbations: Sequence[str] = ("nominal", "early", "late"),
    config: Optional[TrajectoryConfig] = None,
    r1_protocol: Optional[Mapping[str, Any]] = None,
    workers: int = 1,
) -> Dict[str, Any]:
    """Collect a full seed-split data set under one frozen protocol."""

    cfg = config or TrajectoryConfig()
    seed_list = [int(seed) for seed in seeds]
    if len(set(seed_list)) != len(seed_list):
        raise ValueError("seed list must be unique")
    if len(seed_list) > cfg.max_seeds:
        raise ValueError(f"requested {len(seed_list)} seeds but max_seeds={cfg.max_seeds}")
    if r1_protocol is not None:
        selected = r1_protocol.get("selected", r1_protocol)
        cfg = TrajectoryConfig(
            **{
                **cfg.__dict__,
                "force_base": float(selected["force_base"]),
                "force_duration_steps": int(selected["duration_steps"]),
                "force_direction": tuple(float(x) for x in selected["direction"]),
                "r1_steps": int(selected.get("r1_steps", cfg.r1_steps)),
                "r1_use_contact_anchor": bool(selected.get("r1_use_contact_anchor", cfg.r1_use_contact_anchor)),
                "allow_safety_fallback": bool(selected.get("allow_safety_fallback", cfg.allow_safety_fallback)),
            }
        )
        if cfg.require_r1_stability and not bool(r1_protocol.get("protocol_frozen", False)):
            raise ValueError("R1 protocol scan did not meet the frozen stability criteria")
    episodes: List[TrajectoryEpisode] = []
    started = time.time()
    jobs = [(seed, perturbation, cfg) for seed in seed_list for perturbation in perturbations]
    if workers > 1:
        # Spawn keeps EGL contexts process-local. Workers return episodes in
        # seed order so concurrency does not change the recorded data.
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=int(workers), mp_context=multiprocessing.get_context("spawn")) as pool:
            for current in pool.map(_collect_job, jobs):
                episodes.append(current)
                _collection_progress(current)
    else:
        for job in jobs:
            current = _collect_job(job)
            episodes.append(current)
            _collection_progress(current)
    report = write_hdf5(episodes, cfg.output)
    integrity = validate_protocol(cfg.output)
    integrity_path = Path(cfg.output).with_suffix(".integrity.json")
    integrity_path.write_text(json.dumps(_jsonable(integrity), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not integrity.get("ok", False):
        raise ValueError(f"trajectory protocol integrity check failed: {integrity['errors'][:3]}")
    late = [ep for ep in episodes if ep.perturbation == "late"]
    late_attempts = [ep for ep in late if isinstance(ep.evaluator_metadata.get("r1"), Mapping)]
    late_r1 = [bool(ep.evaluator_metadata.get("success", False)) for ep in late_attempts]
    r1_by_perturbation: Dict[str, Dict[str, Optional[float]]] = {}
    for perturbation in sorted({ep.perturbation for ep in episodes}):
        attempted = [
            ep for ep in episodes
            if ep.perturbation == perturbation and isinstance(ep.evaluator_metadata.get("r1"), Mapping)
        ]
        r1_by_perturbation[perturbation] = {
            "attempt_rate": float(len(attempted) / max(1, sum(ep.perturbation == perturbation for ep in episodes))),
            "success_rate_conditional": float(
                np.mean([bool(ep.evaluator_metadata.get("success", False)) for ep in attempted])
            ) if attempted else None,
        }
    report.update(
        {
            "seeds": sorted({int(ep.seed) for ep in episodes}),
            "protocol": episodes[0].metadata.get("protocol") if episodes else None,
            "physical_success_rate": float(np.mean([bool(ep.evaluator_metadata.get("success")) for ep in episodes])) if episodes else 0.0,
            "failure_reasons": {reason: sum(ep.evaluator_metadata.get("failure_reason") == reason for ep in episodes)
                                for reason in sorted({ep.evaluator_metadata.get("failure_reason") for ep in episodes} - {None})},
            "contact_anchor_count": sum(len(ep.contact_anchors) for ep in episodes),
            "post_initialization_state_edits": sum(len(ep.state_edits) for ep in episodes),
            "safety_fallback_count": sum(len(ep.safety_fallbacks) for ep in episodes),
            "rows": [{key: value for key, value in ep.evaluator_metadata.items() if key not in ("trace",)} for ep in episodes],
            "perturbations": list(perturbations),
            "duration_s": round(time.time() - started, 3),
            "labels": {
                "supported": int(sum(sum(ep.labels) for ep in episodes)),
                "violated": int(sum(len(ep.labels) - sum(ep.labels) for ep in episodes)),
            },
            "r1_attempt_rate": float(len(late_attempts) / max(1, len(late))) if late else None,
            "r1_success_rate": float(np.mean(late_r1)) if late_r1 else None,
            "r1_by_perturbation": r1_by_perturbation,
            "r1_stability_target": float(cfg.r1_min_success),
            "integrity": str(integrity_path),
        }
    )
    summary_path = Path(cfg.output).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["summary_path"] = str(summary_path)
    if cfg.require_r1_stability and late_r1 and float(np.mean(late_r1)) < float(cfg.r1_min_success):
        raise RuntimeError(
            f"continuous R1 success rate {float(np.mean(late_r1)):.3f} is below {cfg.r1_min_success:.3f}"
        )
    return report


def _collect_job(job):
    return collect_episode(*job)


def _collection_progress(current):
    print(json.dumps({"seed": current.seed, "perturbation": current.perturbation,
                      "frames": len(current.rgb), "success": current.evaluator_metadata.get("success"),
                      "failure_reason": current.evaluator_metadata.get("failure_reason")}), flush=True)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect AuditEmbodied robot-only continuous trajectories")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-list", type=int, nargs="+", help="explicit seed IDs; overrides --seeds")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", default="outputs/l0-trajectory-pilot-robot-only.h5")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--phase-steps", type=int, default=5)
    parser.add_argument("--capture-stride", type=int, default=1)
    parser.add_argument("--force-base", type=float, default=350.0)
    parser.add_argument("--force-duration", type=int, default=6)
    parser.add_argument("--r1-steps", type=int, default=38)
    parser.add_argument("--r1-approach-steps", type=int, default=16)
    parser.add_argument("--no-contact-anchor", action="store_true", help="run R1 without qpos/contact anchors")
    parser.add_argument("--no-safety-fallback", action="store_true", help="do not synthesize table failures")
    parser.add_argument("--scan-r1", action="store_true")
    parser.add_argument("--scan-seeds", type=int, default=10)
    parser.add_argument("--scan-output", default="outputs/r1-protocol-scan.json")
    parser.add_argument("--protocol", help="JSON scan report; use its selected fixed protocol")
    parser.add_argument("--require-r1-stability", action="store_true")
    parser.add_argument("--no-nuisance", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.scan_r1:
        report = scan_r1_protocol(
            seeds=range(max(1, int(args.scan_seeds))),
            output=args.scan_output,
            use_contact_anchor=not bool(args.no_contact_anchor),
            allow_safety_fallback=not bool(args.no_safety_fallback),
            r1_steps=int(args.r1_steps),
            r1_approach_steps=int(args.r1_approach_steps),
        )
    else:
        protocol = None
        if args.protocol:
            protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
        report = collect_dataset(
            args.seed_list or range(int(args.seed_start), int(args.seed_start) + max(1, int(args.seeds))),
            config=TrajectoryConfig(
                camera_height=int(args.height),
                camera_width=int(args.width),
                phase_steps=int(args.phase_steps),
                capture_stride=int(args.capture_stride),
                force_base=float(args.force_base),
                force_duration_steps=int(args.force_duration),
                r1_steps=int(args.r1_steps),
                r1_approach_steps=int(args.r1_approach_steps),
                r1_use_contact_anchor=False,
                allow_safety_fallback=False,
                output=str(args.output),
                unrelated_changes=not bool(args.no_nuisance),
                require_r1_stability=bool(args.require_r1_stability),
            ),
            r1_protocol=protocol,
            workers=max(1, args.workers),
        )
    print(json.dumps(_jsonable(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
