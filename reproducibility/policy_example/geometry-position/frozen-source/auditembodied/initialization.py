"""Zero-force, no-anchor acceptance check for the continuous initialization."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

from .l0 import L0Config, OracleVerifier, TwoArmL0Env


def audit_initialization(
    seeds: Iterable[int] = range(10),
    hold_seconds: float = 1.0,
    output: str | Path = "outputs/initialization-stability.json",
) -> Dict[str, Any]:
    rows = []
    for seed in seeds:
        env = TwoArmL0Env(int(seed), L0Config(render=False, horizon=300, expert_mode="continuous"))
        try:
            import mujoco

            env.reset()
            before = np.array([warning.number for warning in env.sim.data.warning], dtype=int)
            env.bootstrap_handover()
            start_time = float(env.sim.data.time)
            verifier = OracleVerifier()
            trace = [{"time": start_time, **verifier.evaluate(env)}]
            action = np.zeros(env.action_dim)
            action[6] = action[13] = 1.0
            hold_steps = int(math.ceil(hold_seconds * env.config.control_freq))
            for _ in range(hold_steps):
                env.step(action)
                trace.append({"time": float(env.sim.data.time), **verifier.evaluate(env)})
            after = np.array([warning.number for warning in env.sim.data.warning], dtype=int)
            bad_state_warnings = {
                name: int(after[int(getattr(mujoco.mjtWarning, name))] - before[int(getattr(mujoco.mjtWarning, name))])
                for name in ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC", "mjWARN_BADCTRL")
            }
            finite = bool(np.all(np.isfinite(env.sim.data.qpos)) and np.all(np.isfinite(env.sim.data.qvel)))
            receiver_stable = all(point["receiver_supports"] for point in trace)
            giver_stable = all(point["giver_holds"] for point in trace)
            duration = float(env.sim.data.time) - start_time
            passed = bool(
                receiver_stable and giver_stable and finite
                and duration >= hold_seconds - 1e-9
                and not any(bad_state_warnings.values())
            )
            rows.append({
                "seed": int(seed), "passed": passed,
                "receiver_stable": receiver_stable, "giver_stable": giver_stable,
                "hold_steps": hold_steps, "observed_seconds": duration,
                "bad_state_warnings": bad_state_warnings,
                "bootstrap": env.bootstrap_info,
                "max_relative_speed": max(point["relative_speed"] for point in trace),
                "min_clearance": min(point["clearance"] for point in trace),
                "trace": trace,
            })
        except Exception as exc:
            rows.append({"seed": int(seed), "passed": False, "error": repr(exc)})
        finally:
            env.close()
    passed_count = sum(row["passed"] for row in rows)
    report = {
        "schema": "auditembodied.initialization.zero_force.v1",
        "criteria": {"min_success_rate": 0.9, "hold_seconds": float(hold_seconds),
                     "all_sampled_supports_true": True, "external_force": [0.0, 0.0, 0.0],
                     "contact_anchor_steps": 0, "post_close_state_edits": 0},
        "seeds": [row["seed"] for row in rows], "episodes": len(rows),
        "passed_count": passed_count, "success_rate": passed_count / max(1, len(rows)),
        "acceptance_passed": bool(rows and passed_count / len(rows) >= 0.9),
        "rows": rows,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", default="outputs/initialization-stability.json")
    args = parser.parse_args()
    result = audit_initialization(range(args.seeds), output=args.output)
    print(json.dumps({key: result[key] for key in ("episodes", "passed_count", "success_rate", "acceptance_passed")}))
