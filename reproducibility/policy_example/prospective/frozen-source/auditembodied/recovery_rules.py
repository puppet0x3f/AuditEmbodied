"""Sequential recovery selection using measured robot finger positions."""

import numpy as np

from .recovery_skills import PoseRetargetedR1Recovery, actual_gripper_positions


class LocalThenTableRule:
    """Try frozen local R1, then escalate if receiver closure is inconsistent.

    This rule receives no fault label, phase label, object pose or contact
    truth. The return value from R1 contains evaluator diagnostics; it is
    deliberately discarded before choosing the recovery scope.
    """

    def __init__(self, relative_gap_tolerance=0.05):
        self.relative_gap_tolerance = float(relative_gap_tolerance)

    def run(self, env, expert, episode, capture, registered_positions, registered_rotations,
            reference_gaps, table_position, global_recovery):
        PoseRetargetedR1Recovery().run(
            env, expert, episode, capture, registered_positions, registered_rotations)
        fingers = np.abs(actual_gripper_positions(env)[1])
        reference = float(reference_gaps[1])
        error = float(np.max(np.abs(fingers - reference)) / max(reference, 1e-6))
        escalate = not bool(np.all(np.isfinite(fingers)) and error <= self.relative_gap_tolerance)
        if escalate:
            global_recovery.run(env, expert, episode, capture, registered_positions,
                                reference_gaps, table_position)
        return {"skill": "rule", "used_global_recovery": escalate,
                "sequence": ["R1_retarget", "R3"] if escalate else ["R1_retarget"],
                "receiver_actual_finger_displacements": fingers.tolist(),
                "receiver_registered_finger_displacement": reference,
                "relative_gap_error": error, "relative_gap_tolerance": self.relative_gap_tolerance,
                "decision_source": "actual robot finger joints relative to registered robot grasp"}
