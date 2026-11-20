"""L0 oracle closed-loop handover.

The L0 implementation deliberately keeps the perception side out of the loop.  A
MuJoCo contact predicate is exposed through :class:`OracleVerifier`, the release
gate only consumes that predicate, and R1 is a fixed receiver re-grasp action.
This makes the first experiment answer one narrow question: can a correct gate
and one recovery skill prevent an unsupported release?

The robosuite 1.5.2 ``TwoArmHandover`` reset path places the hammer in the
giver's gripper by running a short settling loop.  That loop leaves non-zero
velocities and stale OSC controller state in this release.  ``TwoArmL0Env``
zeros the velocities and synchronizes the controllers after reset.  The
kinematic bootstrap used by the smoke benchmark then moves both Panda end
effectors to a shared handover pose and closes both grippers around the hammer;
the MuJoCo contact predicate is still used for every decision.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


CAMERAS = (
    "frontview",
    "agentview",
    "robot0_eye_in_hand",
    "robot1_eye_in_hand",
)


class RenderUnavailable(RuntimeError):
    """Raised when an off-screen GL context cannot be created."""


def _prepare_gl(render: bool) -> None:
    """Set a headless GL backend before importing mujoco / robosuite.

    The server image does not expose the NVIDIA EGL device extension.  A small
    Mesa OSMesa runtime is available under ``/tmp/ae-x11`` after the environment
    setup, so rendering can still be tested without a display server.  Users
    with a system EGL installation can override this with ``MUJOCO_GL=egl``.
    """

    if render:
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
        mesa_dirs = [
            "/tmp/ae-x11/usr/lib/x86_64-linux-gnu",
            "/tmp/ae-x11/usr/lib",
        ]
        existing = os.environ.get("LD_LIBRARY_PATH", "").split(":") if os.environ.get("LD_LIBRARY_PATH") else []
        for directory in reversed(mesa_dirs):
            if os.path.isdir(directory) and directory not in existing:
                existing.insert(0, directory)
        if existing:
            os.environ["LD_LIBRARY_PATH"] = ":".join(existing)
        # PyOpenGL's ``ctypes.util.find_library`` is evaluated during import;
        # changing LD_LIBRARY_PATH from inside an already running Python
        # process is too late for some loader configurations.  Loading the
        # OSMesa shared object explicitly makes the backend available to both
        # PyOpenGL and MuJoCo.
        for candidate in (
            os.path.join(sys.prefix, "lib", "libOSMesa.so.8"),
            os.path.join(sys.prefix, "lib", "libOSMesa.so.6"),
            "/tmp/ae-x11/usr/lib/x86_64-linux-gnu/libOSMesa.so.8",
            "/tmp/ae-x11/usr/lib/x86_64-linux-gnu/libOSMesa.so.6",
        ):
            if os.path.exists(candidate):
                try:
                    ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break
    else:
        # GLFW is only used for non-rendering environments; no display is
        # opened when ``has_offscreen_renderer=False``.
        os.environ.setdefault("MUJOCO_GL", "glfw")


def _import_robosuite(render: bool):
    _prepare_gl(render)
    try:
        from robosuite import make
        from robosuite.controllers import load_composite_controller_config
        from robosuite.robots.fixed_base_robot import FixedBaseRobot
    except Exception as exc:  # pragma: no cover - depends on an external install
        raise RuntimeError(
            "robosuite/MuJoCo could not be imported. Activate the auditembodied "
            "environment and install the pinned dependencies."
        ) from exc

    # robosuite 1.5.2's TwoArmHandover reset code still references the old
    # ``robot.controller`` mapping.  The current API stores the same mapping in
    # ``part_controllers``.  Install a local compatibility property once.
    if not hasattr(FixedBaseRobot, "controller"):
        FixedBaseRobot.controller = property(lambda robot: robot.part_controllers)
    if render:
        from robosuite.utils.binding_utils import MjRenderContext

        # robosuite 1.5.2 creates a fresh context on hard reset. A previous
        # context can be garbage-collected while the new one is current;
        # freeing its GL objects then deletes IDs belonging to the new one.
        # Always bind the owning context for drawing and destruction.
        if not getattr(MjRenderContext, "_auditembodied_context_binding", False):
            original_render = MjRenderContext.render
            original_del = MjRenderContext.__del__

            def render_current(context, *args, **kwargs):
                context.gl_ctx.make_current()
                return original_render(context, *args, **kwargs)

            def destroy_current(context):
                if not hasattr(context, "con") or not hasattr(context, "gl_ctx"):
                    return
                context.gl_ctx.make_current()
                original_del(context)

            MjRenderContext.render = render_current
            MjRenderContext.__del__ = destroy_current
            MjRenderContext._auditembodied_context_binding = True
    return make, load_composite_controller_config


@dataclass(frozen=True)
class L0Config:
    """Runtime settings shared by the environment and benchmark."""

    horizon: int = 80
    control_freq: int = 20
    lite_physics: bool = False
    camera_names: Tuple[str, ...] = CAMERAS
    camera_height: int = 64
    camera_width: int = 64
    render: bool = False
    kinematic_bootstrap: bool = True
    # ``kinematic`` keeps the frozen L0 logic benchmark reproducible.  The
    # trajectory collector uses ``continuous`` and feeds 14-D OSC+gripper
    # actions through ``TwoArmL0Env.step``.
    expert_mode: str = "kinematic"
    frame_dir: Optional[str] = None
    relative_speed_threshold: float = 0.2
    max_frame_events: int = 4


@dataclass
class FullState:
    """Restorable simulator and controller state.

    MuJoCo's qpos/qvel state is supplemented with controls, time, episode
    counters, RNG state, and controller goals.  This is the state needed for a
    deterministic replay check in L0; it is intentionally a plain dataclass so
    it can also be serialized by downstream evaluators if desired.
    """

    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    act: np.ndarray
    timestep: int
    cur_time: float
    done: bool
    rng_state: Any = None
    controller_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    mj_state: Optional[np.ndarray] = None


def _copy_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


class TwoArmL0Env:
    """Thin robosuite wrapper exposing L0 state and deterministic helpers."""

    def __init__(self, seed: int, config: Optional[L0Config] = None):
        self.seed = int(seed)
        self.config = config or L0Config()
        make, load_controller_config = _import_robosuite(self.config.render)

        controller_configs = []
        for _ in range(2):
            cfg = copy.deepcopy(load_controller_config(robot="Panda"))
            # World-frame deltas make a scripted Cartesian expert independent of
            # the opposed robot base orientations.  The smoke path uses direct
            # state placement, but the exposed action space remains standard
            # robosuite OSC_POSE + GRIP (7 dimensions per arm).
            cfg["body_parts"]["right"]["input_ref_frame"] = "world"
            controller_configs.append(cfg)

        try:
            self.raw = make(
                "TwoArmHandover",
                robots=["Panda", "Panda"],
                env_configuration="opposed",
                controller_configs=controller_configs,
                prehensile=False,
                use_object_obs=False,
                use_camera_obs=self.config.render,
                has_offscreen_renderer=self.config.render,
                camera_names=list(self.config.camera_names),
                camera_heights=self.config.camera_height,
                camera_widths=self.config.camera_width,
                horizon=self.config.horizon,
                control_freq=self.config.control_freq,
                lite_physics=self.config.lite_physics,
                reward_shaping=False,
                seed=self.seed,
            )
        except Exception as exc:
            if self.config.render:
                raise RenderUnavailable(
                    "Four-camera rendering could not initialize. Try the setup "
                    "command from the L0 document or run with render=False."
                ) from exc
            raise

        self.obs: Mapping[str, Any] = {}
        self._initial_handle_offset = np.zeros(3, dtype=np.float64)
        self._initial_hammer_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._reset_arm_qpos: List[np.ndarray] = []
        self.reset_repaired: bool = False
        self.reset_repair_reason: Optional[str] = None
        self.bootstrap_info: Dict[str, Any] = {}
        self._hammer_qpos_index = self.raw.sim.model.joint_name2id(self.raw.hammer.joints[0])

    @property
    def action_dim(self) -> int:
        return int(self.raw.action_dim)

    @property
    def sim(self):
        return self.raw.sim

    def _eef_pos(self, robot_index: int) -> np.ndarray:
        robot = self.raw.robots[robot_index]
        site_id = robot.eef_site_id["right"]
        return np.asarray(self.raw.sim.data.site_xpos[site_id], dtype=np.float64).copy()

    def _sync_after_reset(self) -> None:
        # The handover reset settling loop can leave qvel and OSC's internal
        # state inconsistent.  Zeroing qvel is a reset-time operation only; it
        # is never performed between ordinary control steps.
        self.raw.sim.data.qvel[:] = 0.0
        self.raw.sim.forward()
        for robot_index, robot in enumerate(self.raw.robots):
            arm = robot.composite_controller.get_controller("right")
            arm.update(force=True)
            arm.update_initial_joints(robot.get_robot_joint_positions())
            arm.reset_goal()
            grip = robot.composite_controller.get_controller("right_gripper")
            grip.update(force=True)
            grip.reset_goal()

    def _sync_after_kinematic_edit(self) -> None:
        """Clear velocities and align controller targets after a state edit."""

        self.raw.sim.data.qvel[:] = 0.0
        self.raw.sim.forward()
        for robot_index, robot in enumerate(self.raw.robots):
            arm = robot.composite_controller.get_controller("right")
            arm.update(force=True)
            arm.update_initial_joints(robot.get_robot_joint_positions())
            arm.reset_goal()
            grip = robot.composite_controller.get_controller("right_gripper")
            grip.update(force=True)
            grip.reset_goal()
            # A direct qpos edit does not update PandaGripper's accumulated
            # action or MuJoCo ctrl. Align both at the initialization boundary;
            # otherwise the first physical step can reopen a closed gripper
            # and apply arm torques left over from the reset pose.
            grip_qpos = self.sim.data.qpos[robot._ref_gripper_joint_pos_indexes["right"]]
            closed = abs(float(grip_qpos[0])) < 0.01
            self.sync_gripper_controller(robot_index, closed=closed)

    def sync_gripper_controller(self, robot_index: int, closed: bool) -> None:
        """Align a Panda gripper's accumulated action and MuJoCo ctrl."""

        robot = self.raw.robots[int(robot_index)]
        gripper_action = np.array([-1.0, 1.0]) if closed else np.array([1.0, -1.0])
        robot.gripper["right"].current_action = gripper_action
        actuator_indexes = np.asarray(
            robot._ref_joint_gripper_actuator_indexes["right"], dtype=np.int64
        )
        ctrl_low = self.sim.model.actuator_ctrlrange[actuator_indexes, 0]
        ctrl_high = self.sim.model.actuator_ctrlrange[actuator_indexes, 1]
        self.sim.data.ctrl[actuator_indexes] = ctrl_low + 0.5 * (gripper_action + 1.0) * (
            ctrl_high - ctrl_low
        )

    def reset(self) -> Mapping[str, Any]:
        self.reset_repaired = False
        self.reset_repair_reason = None
        self.obs = self.raw.reset()
        self._sync_after_reset()
        self.obs = self.raw._get_observations(force_update=True)
        self._reset_arm_qpos = [robot.get_robot_joint_positions().copy() for robot in self.raw.robots]
        self._initial_handle_offset = np.asarray(self.raw._handle_xpos, dtype=np.float64).copy() - self._eef_pos(0)
        self._initial_hammer_quat = self.sim.data.qpos[self._hammer_qpos_index + 3 : self._hammer_qpos_index + 7].copy()
        # A small number of robosuite seeds leave the hammer outside the
        # giver's fingers after the built-in settling loop.  Re-anchor those
        # rare reset states before the L0 expert starts, so a failed reset is
        # not mislabelled as a gate or recovery failure.
        giver_holds, _, _, _ = self.raw._get_task_info()
        if not giver_holds:
            giver = self.raw.robots[0]
            self.reset_repaired = True
            self.reset_repair_reason = "giver_contact_missing_after_robosuite_settling"
            self._set_hammer_pose(self._eef_pos(0), self._initial_hammer_quat)
            self.sim.data.qpos[giver._ref_gripper_joint_pos_indexes["right"]] = 0.0
            self._sync_after_kinematic_edit()
            self.obs = self.raw._get_observations(force_update=True)
            self._initial_handle_offset = np.asarray(self.raw._handle_xpos, dtype=np.float64).copy() - self._eef_pos(0)
        return self.obs

    def step(self, action: Optional[Sequence[float]] = None) -> Mapping[str, Any]:
        if action is None:
            action = np.zeros(self.action_dim, dtype=np.float64)
        self.obs, _, _, _ = self.raw.step(np.asarray(action, dtype=np.float64))
        return self.obs

    def apply_body_force(
        self,
        force: Sequence[float],
        duration_steps: int,
        action: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Apply a registered MuJoCo external force to the hammer body.

        This is the physical perturbation primitive used by the trajectory
        collector.  It deliberately operates through MuJoCo's force buffer
        and advances the simulator, rather than teleporting the object.  The
        caller records the force and duration alongside the resulting oracle
        predicate.  A small amount of controller action can be supplied to
        keep the grippers closed while the force is active.
        """

        force_vec = np.asarray(force, dtype=np.float64).reshape(3)
        steps = max(1, int(duration_steps))
        body_id = int(self.sim.model.body_name2id(self.raw.hammer.root_body))
        self.sim.data.xfrc_applied[body_id, :] = 0.0
        self.sim.data.xfrc_applied[body_id, :3] = force_vec
        try:
            for _ in range(steps):
                if action is None:
                    # Direct MuJoCo stepping preserves the current controller
                    # torques and avoids injecting a zero gripper command.
                    self.sim.step()
                else:
                    self.step(action)
        finally:
            self.sim.data.xfrc_applied[body_id, :] = 0.0
            self.sim.forward()
            # Direct MuJoCo stepping advances simulator time without passing
            # through robosuite's high-level ``step`` bookkeeping.  Keep the
            # wrapper's replay metadata aligned with the physical clock.
            self.raw.cur_time = float(self.sim.data.time)
            self.raw.timestep = int(round(self.raw.cur_time / float(self.raw.control_timestep)))
            if not self.config.render:
                self.obs = self.raw._get_observations(force_update=True)
        return {
            "body": self.raw.hammer.root_body,
            "force": force_vec.copy(),
            "duration_steps": steps,
            "timestep": int(getattr(self.raw, "timestep", 0)),
        }

    def close(self) -> None:
        # On the current server, robosuite 1.5.2's OSMesa context destructor
        # can block after several camera observations and a direct MuJoCo
        # state edit.  Rendered benchmark workers write their frames and exit;
        # letting the OS reclaim that context is safer than hanging the whole
        # matrix.  The ordinary render smoke (without frame_dir) still closes
        # the context normally.
        if self.config.render and self.config.frame_dir:
            return
        close = getattr(self.raw, "close", None)
        if close is not None:
            close()

    def _set_hammer_pose(self, position: Sequence[float], quat: Optional[Sequence[float]] = None) -> None:
        qpos = self.sim.data.qpos[self._hammer_qpos_index : self._hammer_qpos_index + 7]
        qpos[:3] = np.asarray(position, dtype=np.float64)
        qpos[3:] = self._initial_hammer_quat if quat is None else np.asarray(quat, dtype=np.float64)
        self.sim.data.qvel[:] = 0.0
        self.sim.forward()
        if not self.config.render:
            self.obs = self.raw._get_observations(force_update=True)

    def _move_arm_to_position(
        self,
        robot_index: int,
        target: Sequence[float],
        max_iterations: int = 100,
        tolerance: float = 2e-4,
    ) -> None:
        """Use a small position-only MuJoCo IK solve for the compact expert."""

        import mujoco

        robot = self.raw.robots[robot_index]
        site_id = robot.eef_site_id["right"]
        joint_indexes = np.asarray(robot._ref_arm_joint_pos_indexes, dtype=np.int64)
        target_position = np.asarray(target, dtype=np.float64)
        for _ in range(max_iterations):
            self.sim.forward()
            error = target_position - np.asarray(self.sim.data.site_xpos[site_id], dtype=np.float64)
            if float(np.linalg.norm(error)) <= tolerance:
                break
            jacobian_position = np.zeros((3, self.sim.model.nv), dtype=np.float64)
            jacobian_rotation = np.zeros((3, self.sim.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(
                self.sim.model._model,
                self.sim.data._data,
                jacobian_position,
                jacobian_rotation,
                site_id,
            )
            jacobian = jacobian_position[:, joint_indexes]
            damping = 1e-4
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(3),
                0.6 * error,
            )
            lower = self.sim.model.jnt_range[joint_indexes, 0]
            upper = self.sim.model.jnt_range[joint_indexes, 1]
            self.sim.data.qpos[joint_indexes] = np.clip(
                self.sim.data.qpos[joint_indexes] + delta,
                lower + 1e-5,
                upper - 1e-5,
            )
            self.sim.data.qvel[:] = 0.0
        self.sim.forward()
        residual = float(np.linalg.norm(target_position - self.sim.data.site_xpos[site_id]))
        if residual > 2e-3:
            raise RuntimeError(
                f"position-only IK failed for robot {robot_index}: residual={residual:.6f} m"
            )

    def place_object_in_receiver(self) -> None:
        """Kinematically place the hammer in robot 1's gripper.

        The relative handle-to-giver-EFF offset is measured from the actual
        reset state, so the resulting state is valid across placement seeds and
        is checked by MuJoCo contact before it is accepted.
        """

        receiver_eef = self._eef_pos(1)
        self._set_hammer_pose(receiver_eef + self._initial_handle_offset)

    def _move_arm_to_pose(
        self, robot_index: int, position: Sequence[float], rotation: np.ndarray
    ) -> None:
        """Set a separated initial arm pose with position and orientation IK.

        This runs only at the initialization boundary. Once closing begins,
        the simulator advances exclusively through the ordinary OSC actions.
        """

        import mujoco
        from robosuite.utils import transform_utils as transforms

        robot = self.raw.robots[robot_index]
        site_id = robot.eef_site_id["right"]
        joint_indexes = np.asarray(robot._ref_arm_joint_pos_indexes, dtype=np.int64)
        target = np.asarray(position, dtype=np.float64)
        rotation = np.asarray(rotation, dtype=np.float64)
        for _ in range(500):
            self.sim.forward()
            current_rotation = self.sim.data.site_xmat[site_id].reshape(3, 3)
            position_error = target - self.sim.data.site_xpos[site_id]
            rotation_error = transforms.quat2axisangle(
                transforms.mat2quat(rotation @ current_rotation.T)
            )
            if np.linalg.norm(position_error) < 2e-4 and np.linalg.norm(rotation_error) < 0.003:
                return
            jacobian_position = np.zeros((3, self.sim.model.nv), dtype=np.float64)
            jacobian_rotation = np.zeros_like(jacobian_position)
            mujoco.mj_jacSite(
                self.sim.model._model, self.sim.data._data,
                jacobian_position, jacobian_rotation, site_id,
            )
            jacobian = np.vstack(
                [jacobian_position[:, joint_indexes], 0.25 * jacobian_rotation[:, joint_indexes]]
            )
            error = np.concatenate([position_error, 0.25 * rotation_error])
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 1e-4 * np.eye(6), 0.5 * error
            )
            delta *= min(1.0, 0.15 / max(float(np.linalg.norm(delta)), 1e-12))
            limits = self.sim.model.jnt_range[joint_indexes]
            self.sim.data.qpos[joint_indexes] = np.clip(
                self.sim.data.qpos[joint_indexes] + delta,
                limits[:, 0] + 1e-5, limits[:, 1] - 1e-5,
            )
        raise RuntimeError(
            f"initial pose IK failed for robot {robot_index}: "
            f"position={np.linalg.norm(position_error):.6f}, "
            f"rotation={np.linalg.norm(rotation_error):.6f}"
        )

    def _bootstrap_physical_handover(self) -> None:
        """Initialize separated grasps, then close and settle without anchors."""

        center = np.array([0.0, 0.0, 1.05])
        # A horizontal handle along world Y, with opposed top-down grasps.
        self._set_hammer_pose(center, [2 ** -0.5, -(2 ** -0.5), 0.0, 0.0])
        length = float(self.raw.hammer.handle_length)
        radius = float(self.raw.hammer.handle_radius)
        # Short handles brought the opposed wrists into deep collision even
        # before closing. A 9 cm minimum separation clears the arm links;
        # the grasp pads still overlap the shortest supported handle.
        grasp_half_spacing = max(0.045, 0.32 * length)
        targets = []
        for robot_index, rotation in enumerate(
            (np.diag([-1.0, 1.0, -1.0]), np.diag([1.0, -1.0, -1.0]))
        ):
            target = center + np.array(
                [0.0, (-1.0 if robot_index == 0 else 1.0) * grasp_half_spacing, -0.0036]
            )
            self._move_arm_to_pose(robot_index, target, rotation)
            robot = self.raw.robots[robot_index]
            # The Panda pad's inner surface lies 0.5 mm inside the finger
            # joint displacement. Start with clearance, not interpenetration.
            gap = radius + 0.001
            self.sim.data.qpos[robot._ref_gripper_joint_pos_indexes["right"]] = [gap, -gap]
            targets.append(target)
        self._sync_after_kinematic_edit()
        initial_min_distance = min(
            (float(self.sim.data.contact[i].dist) for i in range(self.sim.data.ncon)),
            default=0.0,
        )
        for robot_index in range(2):
            # These are actuator targets only. Actual finger positions now
            # evolve through MuJoCo contact while the OSC holds the arm pose.
            self.sync_gripper_controller(robot_index, closed=True)
        action = np.zeros(self.action_dim)
        action[6] = action[13] = 1.0
        settle_steps = max(1, int(round(0.5 * self.config.control_freq)))
        for _ in range(settle_steps):
            self.step(action)
        self.bootstrap_info = {
            "method": "separated_pose_ik_then_physical_close",
            "initial_grasp_targets": [target.tolist() for target in targets],
            "grasp_half_spacing_m": grasp_half_spacing,
            "initial_min_contact_distance": initial_min_distance,
            "settle_steps": settle_steps,
            "contact_anchor_steps": 0,
            "post_close_state_edits": 0,
        }

    def receiver_regrasp(self) -> None:
        """Retreat the giver, then place the object in the receiver gripper."""

        self.release_giver()
        self.place_object_in_receiver()

    def bootstrap_handover(self) -> None:
        """Create a deterministic shared-contact handover state.

        Continuous mode uses separated pose IK and physical closing/settling.
        Kinematic mode preserves the frozen L0 position-only logic baseline.
        """

        if self.config.expert_mode == "continuous":
            self._bootstrap_physical_handover()
            return

        handle_target = np.array([0.0, 0.0, 1.05], dtype=np.float64)
        self._set_hammer_pose(handle_target, self._initial_hammer_quat)
        eef_target = handle_target - self._initial_handle_offset
        self._move_arm_to_position(0, eef_target)
        self._move_arm_to_position(1, eef_target)
        for robot in self.raw.robots:
            robot_gripper = robot._ref_gripper_joint_pos_indexes["right"]
            self.sim.data.qpos[robot_gripper] = 0.0
        self._sync_after_kinematic_edit()
        if not self.config.render:
            self.obs = self.raw._get_observations(force_update=True)

    def place_object_on_table(self) -> None:
        """Create a clearly unsupported object state for a disturbance."""

        table_top = float(self.sim.data.site_xpos[self.raw.table_top_id][2])
        table_offset = np.asarray(self.raw.table_offset, dtype=np.float64)
        self._set_hammer_pose([float(table_offset[0]), float(table_offset[1]), table_top + 0.06])

    def release_giver(self) -> None:
        """Open and retreat the giver gripper at the release decision point."""

        giver = self.raw.robots[0]
        giver.set_gripper_joint_positions(giver.gripper["right"].init_qpos, gripper_arm="right")
        if self._reset_arm_qpos:
            self.sim.data.qpos[giver._ref_arm_joint_pos_indexes] = self._reset_arm_qpos[0]
        self._sync_after_kinematic_edit()
        if not self.config.render:
            self.obs = self.raw._get_observations(force_update=True)

    def checkpoint(self) -> FullState:
        controller_state: Dict[str, Dict[str, Any]] = {}
        for robot_index, robot in enumerate(self.raw.robots):
            for name, controller in robot.composite_controller.part_controllers.items():
                state: Dict[str, Any] = {}
                for attr in (
                    "goal_pos",
                    "goal_ori",
                    "initial_joint",
                    "joint_pos",
                    "joint_vel",
                    "torques",
                    "goal_qvel",
                    "relative_ori",
                    "ori_ref",
                ):
                    if hasattr(controller, attr):
                        state[attr] = _copy_value(getattr(controller, attr))
                if name.endswith("gripper") and hasattr(robot.gripper["right"], "current_action"):
                    state["current_action"] = _copy_value(robot.gripper["right"].current_action)
                controller_state[f"{robot_index}:{name}"] = state

        rng_state = None
        rng = getattr(self.raw, "rng", None)
        if rng is not None and hasattr(rng, "get_state"):
            rng_state = copy.deepcopy(rng.get_state())

        act = np.asarray(getattr(self.sim.data, "act", np.zeros(0)), dtype=np.float64).copy()
        mj_state = None
        try:
            import mujoco

            state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
            mj_state = np.zeros(mujoco.mj_stateSize(self.sim.model._model, state_spec), dtype=np.float64)
            mujoco.mj_getState(self.sim.model._model, self.sim.data._data, mj_state, state_spec)
        except (AttributeError, ImportError):  # pragma: no cover - only old MuJoCo bindings
            mj_state = None
        return FullState(
            time=float(self.sim.data.time),
            qpos=np.asarray(self.sim.data.qpos, dtype=np.float64).copy(),
            qvel=np.asarray(self.sim.data.qvel, dtype=np.float64).copy(),
            ctrl=np.asarray(self.sim.data.ctrl, dtype=np.float64).copy(),
            act=act,
            timestep=int(getattr(self.raw, "timestep", 0)),
            cur_time=float(getattr(self.raw, "cur_time", 0.0)),
            done=bool(getattr(self.raw, "done", False)),
            rng_state=rng_state,
            controller_state=controller_state,
            mj_state=mj_state,
        )

    def restore(self, state: FullState) -> None:
        if state.mj_state is not None:
            import mujoco

            mujoco.mj_setState(
                self.sim.model._model,
                self.sim.data._data,
                state.mj_state,
                mujoco.mjtState.mjSTATE_FULLPHYSICS,
            )
        else:
            self.sim.data.time = state.time
            self.sim.data.qpos[:] = state.qpos
            self.sim.data.qvel[:] = state.qvel
            self.sim.data.ctrl[:] = state.ctrl
            if state.act.size and hasattr(self.sim.data, "act") and self.sim.data.act.size:
                self.sim.data.act[:] = state.act
        self.sim.forward()
        self.raw.timestep = state.timestep
        self.raw.cur_time = state.cur_time
        self.raw.done = state.done

        rng = getattr(self.raw, "rng", None)
        if rng is not None and state.rng_state is not None and hasattr(rng, "set_state"):
            rng.set_state(copy.deepcopy(state.rng_state))
        for robot_index, robot in enumerate(self.raw.robots):
            for name, controller in robot.composite_controller.part_controllers.items():
                saved = state.controller_state.get(f"{robot_index}:{name}", {})
                for attr, value in saved.items():
                    if attr == "current_action":
                        robot.gripper["right"].current_action = _copy_value(value)
                    elif hasattr(controller, attr):
                        setattr(controller, attr, _copy_value(value))
        self.obs = self.raw._get_observations(force_update=True)

    def camera_frames(self) -> Dict[str, np.ndarray]:
        """Return the four configured RGB frames from the current state."""

        frames: Dict[str, np.ndarray] = {}
        for camera in self.config.camera_names:
            key = f"{camera}_image"
            if key not in self.obs:
                raise RenderUnavailable(f"Camera observation {key!r} is missing")
            frames[camera] = np.asarray(self.obs[key]).copy()
        return frames


class OracleVerifier:
    """Ground-truth handover predicate used only for L0."""

    def evaluate(self, env: TwoArmL0Env) -> Dict[str, Any]:
        giver_holds, receiver_holds, object_height, table_height = env.raw._get_task_info()
        clearance = float(object_height - table_height)
        threshold = float(getattr(env.raw, "height_threshold", 0.04))
        handle_name = env.sim.model.geom_id2name(env.raw.hammer_handle_geom_id)
        handle_speed = np.asarray(env.sim.data.get_geom_xvelp(handle_name), dtype=np.float64)
        receiver_site_id = env.raw.robots[1].eef_site_id["right"]
        receiver_site_name = env.sim.model.site_id2name(receiver_site_id)
        receiver_speed = np.asarray(env.sim.data.get_site_xvelp(receiver_site_name), dtype=np.float64)
        relative_speed = float(np.linalg.norm(handle_speed - receiver_speed))
        velocity_threshold = float(env.config.relative_speed_threshold)
        receiver_supports = bool(
            receiver_holds and clearance > threshold and relative_speed <= velocity_threshold
        )
        return {
            "giver_holds": bool(giver_holds),
            "receiver_holds": bool(receiver_holds),
            "receiver_supports": receiver_supports,
            "object_height": float(object_height),
            "table_height": float(table_height),
            "clearance": clearance,
            "height_threshold": threshold,
            "relative_speed": relative_speed,
            "relative_speed_threshold": velocity_threshold,
        }


class OracleGate:
    """Release gate: release only after current receiver support is true."""

    def decide(self, predicate: Mapping[str, Any]) -> str:
        return "RELEASE" if bool(predicate["receiver_supports"]) else "HOLD"


class R1Recovery:
    """Fixed recovery skill: giver stays out, receiver re-approaches and re-grasps."""

    name = "receiver_regrasp"

    def run(self, env: TwoArmL0Env, verifier: OracleVerifier) -> Dict[str, Any]:
        env.receiver_regrasp()
        predicate = verifier.evaluate(env)
        return {
            "skill": self.name,
            "success": bool(predicate["receiver_supports"]),
            "predicate": predicate,
        }


@dataclass
class EpisodeResult:
    seed: int
    perturbation: str
    condition: str
    success: bool
    unsupported_release: bool
    recovery_attempted: bool
    recovery_success: bool
    extra_time: int
    events: List[Dict[str, Any]]

    def summary(self) -> Dict[str, Any]:
        result = asdict(self)
        result.pop("events", None)
        return result


class L0Episode:
    """Run one deterministic L0 episode and emit JSONL-compatible events."""

    CONDITIONS = ("no_gate", "oracle_gate", "oracle_gate_r1")
    PERTURBATIONS = ("nominal", "early", "late")

    def __init__(self, config: Optional[L0Config] = None):
        self.config = config or L0Config()
        self.verifier = OracleVerifier()
        self.gate = OracleGate()
        self.recovery = R1Recovery()
        self._frame_event_count = 0

    def _save_frames(self, env: TwoArmL0Env, seed: int, perturbation: str, condition: str, step: int) -> Dict[str, str]:
        if (
            not self.config.render
            or not self.config.frame_dir
            or self._frame_event_count >= self.config.max_frame_events
        ):
            return {}
        try:
            from imageio.v2 import imwrite
        except Exception as exc:  # pragma: no cover - dependency is pinned
            raise RenderUnavailable("imageio is required to save L0 camera frames") from exc
        directory = Path(self.config.frame_dir) / f"seed_{seed:04d}" / perturbation / condition
        directory.mkdir(parents=True, exist_ok=True)
        env.obs = env.raw._get_observations(force_update=True)
        paths: Dict[str, str] = {}
        for camera, frame in env.camera_frames().items():
            path = directory / f"step_{step:04d}_{camera}.png"
            imwrite(path, frame)
            paths[camera] = str(path)
        self._frame_event_count += 1
        return paths

    def run(self, seed: int, perturbation: str = "nominal", condition: str = "oracle_gate_r1") -> EpisodeResult:
        if perturbation not in self.PERTURBATIONS:
            raise ValueError(f"unknown perturbation: {perturbation}")
        if condition not in self.CONDITIONS:
            raise ValueError(f"unknown condition: {condition}")

        env = TwoArmL0Env(seed=seed, config=self.config)
        self._frame_event_count = 0
        events: List[Dict[str, Any]] = []
        step = 0
        unsupported_release = False
        recovery_attempted = False
        recovery_success = False
        extra_time = 0

        def record(stage: str, action: str, predicate: Optional[Mapping[str, Any]] = None, **extra: Any) -> None:
            nonlocal step
            if predicate is None:
                predicate = self.verifier.evaluate(env)
            event: Dict[str, Any] = {
                "schema": "auditembodied.l0.event.v1",
                "seed": int(seed),
                "perturbation": perturbation,
                "condition": condition,
                "step": int(step),
                "stage": stage,
                "action": action,
                "predicate": dict(predicate),
            }
            frame_paths = self._save_frames(env, seed, perturbation, condition, step)
            if frame_paths:
                event["frames"] = frame_paths
            event.update(extra)
            events.append(event)
            step += 1

        try:
            env.reset()
            record(
                "reset",
                "HOLD",
                reset_repaired=bool(env.reset_repaired),
                reset_repair_reason=env.reset_repair_reason,
            )

            if perturbation == "early":
                env.place_object_on_table()
                record("disturbance_early", "HOLD", disturbance="object_displaced_before_receiver_evidence")

            # Scripted approach / grasp.  The compact L0 expert uses position-only
            # IK to create a shared-contact pose; its result is accepted only if
            # the oracle sees both grippers holding the handle.
            if self.config.kinematic_bootstrap:
                env.bootstrap_handover()
            else:
                env.place_object_in_receiver()
            record("receiver_grasp", "HOLD")

            if perturbation == "late":
                # Evidence was true at the previous event, then becomes false
                # before release: this is the central stale-evidence case.
                env.place_object_on_table()
                record("disturbance_late", "HOLD", disturbance="support_invalidated_after_evidence")

            predicate = self.verifier.evaluate(env)
            decision = self.gate.decide(predicate)

            if condition == "no_gate":
                if not predicate["receiver_supports"]:
                    unsupported_release = True
                record("release_decision", "RELEASE", predicate, gate="disabled")
                env.release_giver()
            else:
                if decision == "HOLD" and condition == "oracle_gate_r1":
                    recovery_attempted = True
                    recovery = self.recovery.run(env, self.verifier)
                    recovery_success = bool(recovery["success"])
                    extra_time += 1
                    record("r1_recovery", "REGRASP", recovery["predicate"], recovery=self.recovery.name)
                    predicate = recovery["predicate"]
                    decision = self.gate.decide(predicate)
                if decision == "HOLD":
                    # The gate waits until timeout when no recovery skill is
                    # enabled.  Count the waits explicitly for extra-time.
                    for _ in range(3):
                        extra_time += 1
                        record("gate_wait", "HOLD")
                    record("release_decision", "HOLD", predicate, gate="oracle")
                else:
                    record("release_decision", "RELEASE", predicate, gate="oracle")
                    env.release_giver()

            final_predicate = self.verifier.evaluate(env)
            success = bool(final_predicate["receiver_supports"] and not unsupported_release)
            record("episode_end", "DONE" if success else "FAIL", final_predicate, success=success)
            return EpisodeResult(
                seed=int(seed),
                perturbation=perturbation,
                condition=condition,
                success=success,
                unsupported_release=unsupported_release,
                recovery_attempted=recovery_attempted,
                recovery_success=recovery_success,
                extra_time=extra_time,
                events=events,
            )
        finally:
            env.close()


def run_benchmark(
    seeds: Iterable[int],
    perturbations: Sequence[str] = L0Episode.PERTURBATIONS,
    conditions: Sequence[str] = L0Episode.CONDITIONS,
    config: Optional[L0Config] = None,
) -> List[EpisodeResult]:
    runner = L0Episode(config=config)
    results: List[EpisodeResult] = []
    for seed in seeds:
        for perturbation in perturbations:
            for condition in conditions:
                results.append(runner.run(int(seed), perturbation, condition))
    return results


def aggregate_metrics(results: Sequence[EpisodeResult]) -> Dict[str, Any]:
    by_condition: Dict[str, Dict[str, Any]] = {}
    for condition in sorted({result.condition for result in results}):
        subset = [result for result in results if result.condition == condition]
        by_condition[condition] = {
            "episodes": len(subset),
            "success_rate": float(np.mean([result.success for result in subset])) if subset else 0.0,
            "false_pass_rate": float(np.mean([result.unsupported_release for result in subset])) if subset else 0.0,
            "recovery_success_rate": float(
                np.mean([result.recovery_success for result in subset if result.recovery_attempted])
            )
            if any(result.recovery_attempted for result in subset)
            else 0.0,
            "mean_extra_time": float(np.mean([result.extra_time for result in subset])) if subset else 0.0,
        }
    by_perturbation: Dict[str, Dict[str, Any]] = {}
    for perturbation in sorted({result.perturbation for result in results}):
        subset = [result for result in results if result.perturbation == perturbation]
        by_perturbation[perturbation] = {
            condition: float(np.mean([r.success for r in subset if r.condition == condition]))
            if any(r.condition == condition for r in subset)
            else 0.0
            for condition in L0Episode.CONDITIONS
        }
    return {"by_condition": by_condition, "success_rate_by_perturbation": by_perturbation}


def write_jsonl(results: Sequence[EpisodeResult], path: os.PathLike[str] | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for result in results:
            for event in result.events:
                handle.write(json.dumps(_jsonable(event), ensure_ascii=False) + "\n")


def save_initial_camera_frames(
    seeds: Iterable[int],
    output_dir: os.PathLike[str] | str,
    camera_height: int = 64,
    camera_width: int = 64,
) -> List[str]:
    """Save one synchronized four-view RGB bundle at reset for each seed.

    Keeping camera capture in a short-lived reset-only environment avoids the
    OSMesa context teardown issue in robosuite 1.5.2 after repeated direct state
    edits.  The event-level JSONL benchmark remains physics-only unless a
    caller explicitly supplies a render configuration to :class:`L0Episode`.
    """

    from imageio.v2 import imwrite

    root = Path(output_dir)
    paths: List[str] = []
    config = L0Config(render=True, camera_height=camera_height, camera_width=camera_width, frame_dir=None)
    for seed in seeds:
        env = TwoArmL0Env(seed=int(seed), config=config)
        try:
            env.reset()
            directory = root / f"seed_{int(seed):04d}" / "reset"
            directory.mkdir(parents=True, exist_ok=True)
            for camera, frame in env.camera_frames().items():
                path = directory / f"{camera}.png"
                imwrite(path, frame)
                paths.append(str(path))
        finally:
            env.close()
    return paths


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def smoke_test(render: bool = False, output_dir: Optional[os.PathLike[str] | str] = None) -> Dict[str, Any]:
    """Run the focused L0 smoke checks and return a JSON-safe report."""

    started = time.time()
    frame_dir = str(Path(output_dir) / "frames") if (render and output_dir is not None) else None
    config = L0Config(render=render, frame_dir=frame_dir)
    env = TwoArmL0Env(seed=0, config=config)
    try:
        initial_obs = env.reset()
        action_dim = env.action_dim
        initial_predicate = OracleVerifier().evaluate(env)
        if not initial_predicate["giver_holds"]:
            raise AssertionError(f"giver did not hold after reset: {initial_predicate}")
        checkpoint = env.checkpoint()
        qpos_before = checkpoint.qpos.copy()
        env.place_object_on_table()
        env.restore(checkpoint)
        restore_error = float(np.max(np.abs(env.sim.data.qpos - qpos_before)))
        if restore_error > 1e-10:
            raise AssertionError(f"checkpoint restore error {restore_error}")

        camera_shapes: Dict[str, List[int]] = {}
        if render:
            camera_shapes = {name: list(frame.shape) for name, frame in env.camera_frames().items()}
            if len(camera_shapes) != 4:
                raise AssertionError(f"expected 4 cameras, got {camera_shapes}")
    finally:
        env.close()

    replay_errors: Dict[str, float] = {}
    for replay_seed in range(5):
        replay_env = TwoArmL0Env(seed=replay_seed, config=L0Config(render=False))
        try:
            replay_env.reset()
            replay_state = replay_env.checkpoint()
            for _ in range(50):
                replay_env.step(np.zeros(replay_env.action_dim, dtype=np.float64))
            first_qpos = replay_env.sim.data.qpos.copy()
            first_qvel = replay_env.sim.data.qvel.copy()
            replay_env.restore(replay_state)
            for _ in range(50):
                replay_env.step(np.zeros(replay_env.action_dim, dtype=np.float64))
            replay_errors[str(replay_seed)] = float(
                max(
                    np.max(np.abs(replay_env.sim.data.qpos - first_qpos)),
                    np.max(np.abs(replay_env.sim.data.qvel - first_qvel)),
                )
            )
        finally:
            replay_env.close()

    # The render flag is validated above with all four cameras.  Keep the
    # matrix itself physics-only in the smoke command; a full rendered matrix
    # is available through ``ae-l0 --render`` and is intentionally slower.
    benchmark_config = L0Config(render=False) if render else config
    results = run_benchmark(
        seeds=[0],
        perturbations=L0Episode.PERTURBATIONS,
        conditions=L0Episode.CONDITIONS,
        config=benchmark_config,
    )
    metrics = aggregate_metrics(results)
    checks = {
        "nominal_no_gate_success": next(
            r.success for r in results if r.perturbation == "nominal" and r.condition == "no_gate"
        ),
        "late_no_gate_false_pass": next(
            r.unsupported_release for r in results if r.perturbation == "late" and r.condition == "no_gate"
        ),
        "late_gate_r1_success": next(
            r.success for r in results if r.perturbation == "late" and r.condition == "oracle_gate_r1"
        ),
        "checkpoint_restore_max_abs_error": restore_error,
        "replay_50_step_max_abs_error": max(replay_errors.values()),
    }
    if not all(
        [
            checks["nominal_no_gate_success"],
            checks["late_no_gate_false_pass"],
            checks["late_gate_r1_success"],
            checks["replay_50_step_max_abs_error"] <= 1e-5,
        ]
    ):
        raise AssertionError(f"L0 smoke assertions failed: {checks}")

    report: Dict[str, Any] = {
        "schema": "auditembodied.l0.smoke.v1",
        "duration_s": None,
        "action_dim": int(action_dim),
        "initial_observation_keys": len(initial_obs),
        "initial_predicate": initial_predicate,
        "reset_repaired": bool(getattr(env, "reset_repaired", False)),
        "reset_repair_reason": getattr(env, "reset_repair_reason", None),
        "camera_shapes": camera_shapes,
        "checks": checks,
        "replay_50_step_max_abs_error_by_seed": replay_errors,
        "replay_50_step_max_abs_error": max(replay_errors.values()),
        "metrics": metrics,
    }
    report["duration_s"] = round(time.time() - started, 3)
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        events_path = directory / "l0_smoke_events.jsonl"
        summary_path = directory / "l0_smoke_summary.json"
        report["events_path"] = str(events_path)
        report["summary_path"] = str(summary_path)
        write_jsonl(results, events_path)
        summary_path.write_text(
            json.dumps(_jsonable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return report


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AuditEmbodied L0 oracle closed-loop benchmark")
    parser.add_argument("--seeds", type=int, default=1, help="number of seeds starting at 0")
    parser.add_argument("--render", action="store_true", help="enable four-camera OSMesa RGB observations")
    parser.add_argument("--smoke", action="store_true", help="run assertions and a one-seed smoke benchmark")
    parser.add_argument("--output-dir", default="outputs/l0", help="directory for JSONL and summary outputs")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    started = time.time()
    if args.smoke:
        report = smoke_test(render=args.render, output_dir=args.output_dir)
    else:
        frame_dir = str(Path(args.output_dir) / "frames") if args.render else None
        if args.render:
            save_initial_camera_frames(
                range(max(args.seeds, 1)),
                frame_dir,
                camera_height=64,
                camera_width=64,
            )
        # Run the matrix physics-only after the reset camera bundle.  This
        # keeps the 50-seed benchmark from accumulating OSMesa contexts.
        config = L0Config(render=False)
        results = run_benchmark(range(max(args.seeds, 1)), config=config)
        write_jsonl(results, Path(args.output_dir) / "l0_events.jsonl")
        report = {
            "schema": "auditembodied.l0.benchmark.v1",
            "episodes": len(results),
            "metrics": aggregate_metrics(results),
            "events_path": str(Path(args.output_dir) / "l0_events.jsonl"),
            "initial_frames_dir": frame_dir if args.render else None,
        }
    report["duration_s"] = round(time.time() - started, 3)
    print(json.dumps(_jsonable(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
