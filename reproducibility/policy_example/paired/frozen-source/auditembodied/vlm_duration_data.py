"""Capture matched pre-recovery RGB and build observed-outcome VLM examples.

Rendering is enabled before reset in a wrapper around the unchanged lift env.
The default capture executes fixed continue; an explicit config/candidate can
capture during another registered skill. Images are joined to executed
candidate outcomes only after exact physical-prefix comparison.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .l0 import CAMERAS, _import_robosuite
from . import l4_lift
from .duration_pilot import LOCAL_FIELDS


ROOTS = ("outputs/l4-duration-pilot-20261108", "outputs/l4-duration-relevel-20261108")
CLASSES = {"A": "task completed and every process guard passed",
           "B": "task completed but a process guard was violated",
           "C": "task not completed, process guards passed",
           "D": "task not completed and a process guard was violated",
           "E": "required keeper grasp absent before a local recovery"}


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def capture_one(job, config=None, candidate="continue"):
    seed, scenario, folder = job
    _import_robosuite(True)  # Reuse the existing context-binding fix.
    original = l4_lift.TwoArmLiftEnv
    holder, recorded = {}, {}

    def factory(*args, **kwargs):
        env = original(*args, **kwargs)
        env.raw.has_offscreen_renderer = True
        holder["env"] = env
        return env

    def observe(port):
        from robosuite import macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
        from OpenGL import GL
        env = holder["env"]
        before = l4_lift.snapshot(env)
        convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
        rgb = np.stack([env.sim.render(camera_name=c, width=64, height=64)[::convention].copy() for c in CAMERAS])
        after = l4_lift.snapshot(env)
        if before != after or any(float(v.std()) <= 1 for v in rgb):
            raise ValueError("render modified physical state or returned an invalid camera")
        recorded.update(prefix_state=before, port=port,
                        gpu_renderer=GL.glGetString(GL.GL_RENDERER).decode(),
                        image_std=[float(v.std()) for v in rgb])
        destination = Path(folder) / f"{seed}-{scenario}.npz"
        np.savez_compressed(destination, rgb=rgb)
        recorded["observation_file"] = str(destination.resolve())
        return {"candidate": candidate}

    with patch.object(l4_lift, "TwoArmLiftEnv", factory):
        row = l4_lift.run_trial(seed, scenario, config=config, selector=observe)
    if not recorded or row.get("error"):
        raise RuntimeError(row.get("error", "capture was not reached"))
    row["capture"] = recorded
    return row


def plan(row):
    cfg = l4_lift.LiftConfig()
    steps = {key: getattr(cfg, key) for key in LOCAL_FIELDS}
    steps.update(row.get("local_steps", {}))
    action = row["candidate"]
    if action == "continue":
        return {"kind": "continue", "instruction": "Keep both measured end-effector poses and both grippers closed; check the final one-second hold."}
    if action == "R_global":
        return {"kind": "global", "instruction": "Lower to the registered table, release onto the table, regrasp both handles and lift again; verify one-second hold.",
                "table_lower_steps": cfg.table_lower_steps, "lift_steps": cfg.lift_steps}
    repair, keeper = ("left", "right") if action == "R_left" else ("right", "left")
    return {"kind": "local", "repair_arm": repair, "keeper_arm": keeper,
            "instruction": f"Keep the {keeper} gripper holding. Recover the {repair} handle using the registered relative robot pose and the fixed gravity-arc search, then relevel and verify one-second hold.",
            "arc_search_degrees": list(cfg.local_search_angles_deg), **steps, "final_settle_steps": 8}


def outcome(row):
    action = row["candidate"]
    if action in ("R_left", "R_right"):
        keeper = 1 if action == "R_left" else 0
        if not row["post_fault"]["holds"][keeper]:
            return "E"
    if row["success"]:
        return "B" if row["protection_violation"] else "A"
    return "D" if row["protection_violation"] else "C"


def build(output):
    output = Path(output)
    captures = read_rows(output / "captures.jsonl")
    observed = {(r["seed"], r["scenario"]): r for r in captures}
    executed = list(captures)
    for root in ROOTS:
        for stage in ("pilot", "confirmation"):
            executed.extend(read_rows(Path(root) / (stage+"-trials.jsonl")))
    # The original matrix supplies actually executed wrong-side and continue
    # branches for old training seeds; unknown branches remain absent.
    executed.extend(r for r in read_rows("outputs/l4-lift-20261107/matrix/trials.jsonl")
                    if (r["seed"], r["scenario"]) in observed)
    examples, seen = [], set()
    for row in executed:
        key = row["seed"], row["scenario"]
        capture = observed[key]["capture"]
        if row["prefix_state"] != capture["prefix_state"]:
            raise ValueError(f"observation/outcome physical prefix differs for {key}")
        action = plan(row)
        uid = (key, json.dumps(action, sort_keys=True))
        label = outcome(row)
        if uid in seen:
            previous = next(x for x in examples if (x["seed"], x["scenario"]) == key and x["plan"] == action)
            if previous["label"] != label:
                raise ValueError("repeated executed action has conflicting outcomes")
            continue
        seen.add(uid)
        examples.append({"seed": key[0], "scenario": key[1], "image": capture["observation_file"],
                         "robot": capture["port"], "plan": action, "label": label})
    counts = {}
    for name, allowed in (("train", set(range(8))), ("development", set(range(20, 24)))):
        split = [r for r in examples if r["seed"] in allowed]
        with (output / (name+".jsonl")).open("w") as stream:
            for r in split:
                stream.write(json.dumps(r)+"\n")
        counts[name] = {"examples": len(split), "initialization_seeds": sorted({r["seed"] for r in split}),
                        "physical_states": len({(r["seed"], r["scenario"]) for r in split}),
                        "classes": {c: sum(r["label"] == c for r in split) for c in CLASSES}}
    report = {"counts": counts, "classes": CLASSES, "cameras": list(CAMERAS),
              "rgb_shape": [4, 64, 64, 3], "capture_continuations": len(captures),
              "all_physical_prefixes_identical": True, "unknown_branches_used": 0,
              "note": "Small readiness dataset. Development seeds participated in the physical mechanism diagnostic; they are not a final policy test."}
    (output / "data-report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not args.build_only:
        destination = output / "captures.jsonl"
        existing = {(r["seed"], r["scenario"]) for r in read_rows(destination)} if destination.exists() else set()
        reference = read_rows(Path(ROOTS[0]) / "pilot-trials.jsonl") + read_rows(Path(ROOTS[0]) / "confirmation-trials.jsonl")
        cases = sorted({(r["seed"], r["scenario"]) for r in reference} - existing)
        if args.limit:
            cases = cases[:args.limit]
        with destination.open("a") as stream:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                for row in pool.map(capture_one, [(s, q, str(output)) for s, q in cases]):
                    stream.write(json.dumps(row)+"\n")
                    stream.flush()
                    print(json.dumps({"seed": row["seed"], "scenario": row["scenario"], **{k: row["capture"][k] for k in ("gpu_renderer", "image_std")}}), flush=True)
    if not args.limit:
        build(output)


if __name__ == "__main__":
    main()
