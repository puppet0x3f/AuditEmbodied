"""Actuator-only recovery skills using registered robot waypoints."""

from __future__ import annotations

import numpy as np

from .continuous import ContinuousR1Recovery


def actual_gripper_positions(env):
    """Four actual robot finger joint positions; no object/contact inputs."""
    return np.stack([np.asarray(env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes["right"]]).copy()
                     for robot in env.raw.robots])


def actual_eef_rotations(env):
    """World orientations from the robot's measured end-effector sites."""
    return np.stack([np.asarray(env.sim.data.site_xmat[robot.eef_site_id["right"]]).reshape(3, 3).copy()
                     for robot in env.raw.robots])


class PoseRetargetedR1Recovery:
    """Keep the giver's R1 hold; reconstruct its target from robot SE(3).

    The registered giver-to-receiver transform is carried by the actual giver
    pose. Neither object pose nor fault labels enter this reconstruction.
    """

    def run(self, env, expert, episode, capture, registered_positions, registered_rotations,
            *, height_offset=0.0, steps=38, approach_steps=16):
        positions = np.asarray(registered_positions)
        rotations = np.asarray(registered_rotations)
        giver_position = env._eef_pos(0).copy()
        current_rotations = actual_eef_rotations(env)
        rotation_delta = current_rotations[0] @ rotations[0].T
        receiver_position = giver_position + rotation_delta @ (positions[1] - positions[0])
        receiver_position[2] += float(height_offset)
        # Change the target only. Reuse exactly the frozen R1's actuator
        # controller, including its wrist behavior, timing and gripper signs.
        result = ContinuousR1Recovery(use_contact_anchor=False, approach_steps=approach_steps).run(
            env, expert, episode, capture, steps=steps, receiver_target=receiver_position)
        return {**result, "skill": "R1_retarget",
                "target_source": "registered robot relative position transformed by actual giver SE3",
                "receiver_position_target": receiver_position.tolist(),
                "giver_position_held": giver_position.tolist(), "height_offset": float(height_offset),
                "wrist_control": "unchanged frozen R1; zero rotational action deltas"}


def _phase(env, expert, episode, capture, name, targets, grippers, steps, gain=0.8):
    expert.execute_phase(env, episode, name, targets, contact_mode=None,
                         grippers=grippers, steps=steps, gain=gain, capture=capture)


def _move(env, expert, episode, capture, name, targets, grippers, steps):
    start = np.stack([env._eef_pos(i) for i in range(2)])
    targets = np.asarray(targets)
    for i in range(steps):
        waypoint = start + (targets - start) * ((i + 1) / steps)
        _phase(env, expert, episode, capture, name, waypoint, grippers, 1)


class ContinuousR2Recovery:
    """Reapproach/close the giver while the receiver keeps its grasp."""

    def run(self, env, expert, episode, capture, giver_target, *, steps=38, approach_steps=16):
        receiver_hold = env._eef_pos(1).copy()
        targets = (np.asarray(giver_target), receiver_hold)
        _phase(env, expert, episode, capture, "r2_open_approach", targets, (-1.0, 1.0), approach_steps, 1.0)
        _phase(env, expert, episode, capture, "r2_close_settle", targets, (1.0, 1.0), steps - approach_steps, 1.0)
        return {"skill": "R2", "control_steps": steps,
                "target_source": "registered giver robot waypoint; current receiver robot position"}


class ContinuousR3Recovery:
    """Put down, release onto the table, pick up and repeat the handover.

    The carrier and pickup completion estimate use actual finger positions.
    The placement/pickup locations come from registered robot waypoints and
    a calibrated table plane. Object pose/contact truth is never queried.
    """

    def __init__(self, search_offsets=(0.0, -0.02, 0.02, -0.04, 0.04, -0.06, 0.06),
                 placement_steps=40, pickup_height_adjustment=0.0, receiver_height_offset=-0.025):
        self.search_offsets = tuple(search_offsets)
        self.placement_steps = int(placement_steps)
        self.pickup_height_adjustment = float(pickup_height_adjustment)
        self.receiver_height_offset = float(receiver_height_offset)

    def run(self, env, expert, episode, capture, registered_targets, registered_gaps, table_position):
        registered = np.asarray(registered_targets).copy()
        table_position = np.asarray(table_position)
        table_height = float(table_position[2])
        table_shift = table_position[:2] - registered[:, :2].mean(axis=0)
        gaps = np.mean(np.abs(actual_gripper_positions(env)), axis=1)
        references = np.asarray(registered_gaps)
        plausible = (gaps > 0.006) & (gaps < 0.03)
        deviation = np.where(plausible, np.abs(gaps - references), np.inf)
        carrier = int(np.argmin(deviation)) if np.any(plausible) else None
        park = registered.copy()
        park[:, 2] += 0.10
        # This task's real table is on the giver side. The receiver cannot
        # reach it while maintaining its current grasp. Transfer back using
        # actuator-only R2 if the robot finger readings identify that holder.
        if plausible[0] and abs(gaps[0] - references[0]) < 0.01:
            carrier = 0
        elif carrier == 1:
            ContinuousR2Recovery().run(env, expert, episode, capture, registered[0])
            giver_gap = float(np.mean(np.abs(actual_gripper_positions(env)[0])))
            if 0.006 < giver_gap < 0.03 and abs(giver_gap - references[0]) < 0.01:
                carrier = 0
            else:
                return {"skill": "R3", "carrier_from_actual_fingers": 1,
                        "pickup_estimated_from_actual_fingers": False,
                        "pickup_attempts": [], "robot_estimate_failure": "giver_transfer_not_acquired"}
        if carrier is not None:
            other = 1 - carrier
            targets = np.stack([env._eef_pos(i) for i in range(2)])
            grippers = [-1.0, -1.0]
            grippers[carrier] = 1.0
            _phase(env, expert, episode, capture, "r3_free_other_gripper", targets, grippers, 12)
            targets[other] = park[other]
            _move(env, expert, episode, capture, "r3_park_other_arm", targets, grippers, 16)
            targets[carrier] = registered[carrier]
            _move(env, expert, episode, capture, "r3_recenter_carrier", targets, grippers, 24)
            targets[carrier, :2] += table_shift
            _move(env, expert, episode, capture, "r3_transport_above_real_table", targets, grippers, 60)
            targets[carrier, 2] = table_height + references[carrier] - 0.0036
            _move(env, expert, episode, capture, "r3_lower_to_table", targets, grippers, self.placement_steps)
            _phase(env, expert, episode, capture, "r3_table_loaded", targets, grippers, 8)
            _phase(env, expert, episode, capture, "r3_table_release", targets, (-1.0, -1.0), 14)
            _phase(env, expert, episode, capture, "r3_table_settle", targets, (-1.0, -1.0), 6)
            targets[carrier, 2] += 0.10
            _move(env, expert, episode, capture, "r3_clear_table", targets, (-1.0, -1.0), 16)
        else:
            _move(env, expert, episode, capture, "r3_clear_for_table_restart", park, (-1.0, -1.0), 24)
            _phase(env, expert, episode, capture, "r3_table_settle", park, (-1.0, -1.0), 10)

        attempts = []
        pickup = False
        targets = park.copy()
        for offset in self.search_offsets:
            pickup_target = registered[0].copy()
            pickup_target[:2] += table_shift
            pickup_target[0] += offset
            pickup_target[2] = table_height + references[0] - 0.0036 + self.pickup_height_adjustment
            targets[0] = pickup_target + [0.0, 0.0, 0.10]
            _move(env, expert, episode, capture, "r3_pickup_hover", targets, (-1.0, -1.0), 20)
            targets[0] = pickup_target
            _move(env, expert, episode, capture, "r3_pickup_descend", targets, (-1.0, -1.0), 20)
            _phase(env, expert, episode, capture, "r3_pickup_close", targets, (1.0, -1.0), 22)
            targets[0] = pickup_target + [0.0, 0.0, 0.12]
            _move(env, expert, episode, capture, "r3_pickup_lift", targets, (1.0, -1.0), 28)
            gap = float(np.mean(np.abs(actual_gripper_positions(env)[0])))
            pickup = bool(0.006 < gap < 0.03 and abs(gap - references[0]) < 0.01)
            attempts.append({"x_offset": offset, "actual_finger_gap": gap, "estimated_grasp": pickup})
            if pickup:
                break
            _phase(env, expert, episode, capture, "r3_retry_open", targets, (-1.0, -1.0), 12)
        if pickup:
            targets[0] = registered[0]
            _move(env, expert, episode, capture, "r3_return_to_handover", targets, (1.0, -1.0), 60)
            _phase(env, expert, episode, capture, "r3_handover_settle", targets, (1.0, -1.0), 24)
            # The single-arm table pickup lets the far end sag. This fixed
            # training-calibrated approach offset catches it before closing;
            # it does not change the standalone frozen R1 or read object pose.
            receiver_target = registered[1].copy()
            receiver_target[2] += self.receiver_height_offset
            ContinuousR1Recovery(use_contact_anchor=False, approach_steps=16).run(
                env, expert, episode, capture, steps=38, receiver_target=receiver_target)
        return {"skill": "R3", "carrier_from_actual_fingers": carrier,
                "pickup_estimated_from_actual_fingers": pickup, "pickup_attempts": attempts,
                "receiver_height_offset": self.receiver_height_offset,
                "target_source": "registered robot positions, initial finger gaps and calibrated table plane"}
