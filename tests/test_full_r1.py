from auditembodied.continuous import ContinuousR1Recovery
from auditembodied.r1_validation import run_physical_trial, validate_full_r1_trial


def test_full_r1_requires_physical_release_and_stable_hold():
    row = validate_full_r1_trial(0)
    assert row["failure_created"]
    assert row["giver_preserved_during_fault"]
    assert row["success"]
    assert row["giver_released"]
    assert row["hold_seconds"] >= 1.0 - 1e-9
    assert row["post_release_receiver_stable"]
    assert row["giver_stayed_released"]
    assert row["contact_anchor_count"] == 0
    assert row["state_edit_count_after_initialization"] == 0
    assert row["blocked_calls"] == []
    assert row["recovery"]["target_source"] == "registered_robot_waypoint"


def test_full_r1_rejects_a_hidden_state_edit(monkeypatch):
    def edited_recovery(self, env, *args, **kwargs):
        env._set_hammer_pose([0.0, 0.0, 1.05])

    monkeypatch.setattr(ContinuousR1Recovery, "run", edited_recovery)
    row = validate_full_r1_trial(0)
    assert not row["success"]
    assert row["failure_reason"] == "forbidden_state_edit"
    assert row["blocked_calls"] == [{"stage": "recovery", "method": "_set_hammer_pose"}]


def test_online_gate_has_only_robot_stream_and_counts_all_consumed_frames():
    calls = []
    def gate(stream, index, allow_stale):
        assert set(stream) == {"rgb", "proprio", "actions", "stages"}
        assert len(stream["stages"]) == index + 1
        calls.append((index, allow_stale))
        return not allow_stale
    row, _ = run_physical_trial(0, gate=gate, validation_only=True)
    assert row["success"]
    assert row["gate_checks"] == 2
    assert row["observations"] == calls[-1][0] + 1
    assert row["extra_observations"] == 39
