"""Training/development RGB collection and physical visual recovery trials."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing
from pathlib import Path

import numpy as np

from .continuous import _proprio
from .l0 import CAMERAS, _jsonable
from .l3_matrix import MatrixConfig, run_trial, _summary
from .recovery_skills import actual_eef_rotations, actual_gripper_positions


SCENARIOS = ("object_sham", "object_rotation_mild", "object_rotation", "giver_loss", "coupled_offset")
TARGETS = ("giver_supports", "receiver_supports", "coordinated_support")


def support_targets(predicate, giver_relative_speed, deep_collision):
    giver = bool(predicate["giver_holds"] and predicate["clearance"] > predicate["height_threshold"]
                 and giver_relative_speed <= predicate["relative_speed_threshold"])
    receiver = bool(predicate["receiver_supports"])
    return np.array([giver, receiver, giver and receiver and not deep_collision], dtype=np.int8)


class RGBRecorder:
    """Robot observations and evaluator labels are stored in separate files.

    Rendering is explicit: disabling automatic image observables avoids
    rendering four views at every physics step. It does not change actions,
    simulator state, or the 20 Hz protection monitor.
    """

    def __init__(self, stride=8):
        self.stride = int(stride)
        self.rgb, self.robot, self.labels, self.times, self.stages = [], [], [], [], []
        self.decision_points = []
        self.policy_observations = 0
        self.steps, self.previous_stage = 0, None
        self.cached_time, self.cached = None, None

    def after_reset(self, env):
        for camera in CAMERAS:
            env.raw.modify_observable(f"{camera}_image", "enabled", False)

    def register(self, reference_gaps):
        self.reference_gaps = np.asarray(reference_gaps).copy()

    def _robot_rgb(self, env):
        now = float(env.sim.data.time)
        if self.cached_time != now:
            from robosuite import macros
            from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
            convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
            images = np.stack([env.sim.render(camera_name=c, width=64, height=64)[::convention].copy()
                               for c in CAMERAS])
            fingers = actual_gripper_positions(env)
            errors = np.max(np.abs(np.abs(fingers) - self.reference_gaps[:, None]), axis=1) / self.reference_gaps
            robot = np.r_[_proprio(env), fingers.reshape(-1), self.reference_gaps,
                          actual_eef_rotations(env).reshape(-1), errors].astype(np.float32)
            self.cached = {"rgb": images, "robot": robot}
            self.cached_time = now
        return self.cached

    def observe(self, env):
        """Only these two arrays cross the online policy input boundary."""
        self.policy_observations += 1
        return self._robot_rgb(env)

    def capture(self, env, packet, force=False):
        if not force:
            self.steps += 1
        changed = packet["stage"] != self.previous_stage
        self.previous_stage = packet["stage"]
        if not (force or changed or self.steps % self.stride == 0):
            return
        if self.times and self.times[-1] == float(env.sim.data.time):
            self.decision_points[-1] = self.decision_points[-1] or force
            return
        observed = self._robot_rgb(env)
        predicate = packet["evaluator"]
        handle_name = env.sim.model.geom_id2name(env.raw.hammer_handle_geom_id)
        site_name = env.sim.model.site_id2name(env.raw.robots[0].eef_site_id["right"])
        speed = float(np.linalg.norm(env.sim.data.get_geom_xvelp(handle_name) - env.sim.data.get_site_xvelp(site_name)))
        collision = False
        for contact in env.sim.data.contact[:env.sim.data.ncon]:
            a, b = (env.sim.model.geom_id2name(int(g)) for g in (contact.geom1, contact.geom2))
            if contact.dist < -.01 and ((a.startswith("robot0") and b.startswith("robot1")) or
                                        (a.startswith("robot1") and b.startswith("robot0"))):
                collision = True
                break
        self.rgb.append(observed["rgb"])
        self.robot.append(observed["robot"])
        self.labels.append(support_targets(predicate, speed, collision))
        self.times.append(float(env.sim.data.time))
        self.stages.append(packet["stage"])
        self.decision_points.append(force)

    def save(self, folder, name):
        folder = Path(folder)
        for sub in ("robot", "evaluator"):
            (folder / sub).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "robot" / f"{name}.npz", rgb=np.asarray(self.rgb, dtype=np.uint8),
                            robot=np.asarray(self.robot, dtype=np.float32), time=np.asarray(self.times))
        np.savez_compressed(folder / "evaluator" / f"{name}.npz", labels=np.asarray(self.labels, dtype=np.int8),
                            stages=np.asarray(self.stages), decision=np.asarray(self.decision_points), time=np.asarray(self.times))


def visual_job(job):
    seed, scenario, method, config, folder, model_path = job
    recorder = RGBRecorder()
    policy = None
    if method == "visual":
        from .visual_recovery import MultiSupportVerifier, VisualRecoverySelector
        policy = VisualRecoverySelector(MultiSupportVerifier.load(model_path))
    row = run_trial(seed, scenario, method, config, observer=recorder, recovery_policy=policy)
    recorder.save(folder, f"{seed}-{scenario}-{method}")
    return row


def collect(output, split, workers=4, visual=False):
    output = Path(output)
    baseline = json.loads((output / "frozen-baseline.json").read_text())
    config = MatrixConfig(**baseline["config"])
    if split not in ("pilot", "train", "dev"):
        raise ValueError("Only training and development collection is permitted")
    seeds = baseline["dev_seeds"] if split == "dev" else baseline["train_seeds"]
    methods = ("visual",) if visual else ("rule",) if split == "dev" else ("R1_retarget", "R3", "rule")
    folder = output / ("online-" + split if visual else split)
    folder.mkdir(exist_ok=True)
    if (folder / "trials.jsonl").exists():
        raise ValueError(f"collection already exists: {folder}")
    jobs = [(s, q, k, config, str(folder), str(output / "visual-support.npz"))
            for s in seeds for q in SCENARIOS for k in methods]
    if split == "pilot" and not visual:
        jobs = [(0, "object_rotation", "R1_retarget", config, str(folder), None),
                (0, "object_rotation", "R3", config, str(folder), None),
                (0, "giver_loss", "rule", config, str(folder), None)]
    elif split == "pilot":
        jobs = [job for job in jobs if job[0] == 0]
    with (folder / "trials.jsonl").open("w") as log:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for row in pool.map(visual_job, jobs):
                log.write(json.dumps(_jsonable(row)) + "\n"); log.flush()
                print(json.dumps({k: row.get(k) for k in ("seed", "scenario", "candidate", "success", "failure_reason", "recorded_rgb_bundles")}), flush=True)


def summarize(output):
    output = Path(output)
    results, all_rows = {}, []
    for split in ("train", "dev"):
        groups = {}
        for folder in (output / split, output / ("online-" + split)):
            for line in (folder / "trials.jsonl").open():
                row = json.loads(line)
                trace = row.pop("trace")
                row["trace_monotonic"] = bool(np.all(np.diff([f["time"] for f in trace]) > 0))
                row["split"] = split
                group = groups.setdefault((row["seed"], row["scenario"]), {})
                if row["candidate"] in group:
                    raise ValueError("duplicate trial")
                group[row["candidate"]] = row
                all_rows.append(row)
        baseline = [g["rule"] for g in groups.values()]
        visual = [g["visual"] for g in groups.values()]
        checks = [{"seed": s, "scenario": q, **{key + "_identical": g["rule"].get(key) is not None and
                   g["rule"].get(key) == g["visual"].get(key)
                   for key in ("initialization_state", "initialization_end_state", "prefix_state")}}
                  for (s, q), g in groups.items()]
        pairs = [(g["rule"], g["visual"]) for g in groups.values()
                 if all(g[k]["success"] and not g[k]["protection_violation"] for k in ("rule", "visual"))]
        costs = {"scenes": len(pairs)}
        if pairs:
            for key in ("recovery_seconds", "recovery_path_m"):
                a = float(np.mean([p[0][key] for p in pairs])); b = float(np.mean([p[1][key] for p in pairs]))
                costs[key] = {"rule": a, "visual": b, "reduction_fraction": 1-b/a if a else None}
        results[split] = {"by_method": {k: _summary([g[k] for g in groups.values() if k in g])
                                       for k in ("rule", "visual", "R1_retarget", "R3") if any(k in g for g in groups.values())},
            "by_scenario": {q: {k: _summary([g[k] for (s, scene), g in groups.items() if scene == q])
                                 for k in ("rule", "visual")} for q in SCENARIOS},
            "prefix_checks": checks, "all_prefixes_identical": all(all(v for k, v in c.items() if k.endswith("_identical")) for c in checks),
            "paired_costs": costs,
            "sequences": {k: dict(Counter(tuple(r.get("recovery", {}).get("sequence", []))
                              for r in (baseline if k == "rule" else visual))) for k in ("rule", "visual")},
            "visual_mean_policy_observations": float(np.mean([r["policy_observations"] for r in visual])),
            "failures": [{k: r.get(k) for k in ("seed", "scenario", "candidate", "failure_reason", "failed_stage",
                "recovery", "post_fault_predicate", "post_recovery_predicate", "table_cycle_verified", "protection")}
                         for r in baseline + visual if not r["success"] or r["protection_violation"]]}
        # JSON keys cannot be tuples; keep executed action sequences legible.
        results[split]["sequences"] = {k: {" -> ".join(sequence): n for sequence, n in v.items()}
                                        for k, v in results[split]["sequences"].items()}
    validity = {"actual_episodes": len(all_rows),
        "all_prefixes_identical": all(r["all_prefixes_identical"] for r in results.values()),
        "numerical_instabilities": sum(r["failure_reason"] == "numerical_instability" or
            any(r.get("bad_state_warnings", {}).values()) or not r["trace_monotonic"] for r in all_rows),
        "execution_errors": sum(r["failure_reason"] in ("execution_error", "initialization_error") for r in all_rows),
        "contact_anchors": sum(r["contact_anchor_count"] for r in all_rows),
        "manual_state_edits": sum(r["state_edit_count_after_initialization"] for r in all_rows),
        "forbidden_calls": sum(len(r["blocked_calls"]) for r in all_rows),
        "wrench_leaks": sum(r.get("external_wrench_leak_steps", 0) or bool(r.get("external_wrench_at_recovery_start", 0)) for r in all_rows)}
    baseline = json.loads((output / "frozen-baseline.json").read_text())
    import hashlib
    previous = baseline["existing_test_results"]
    # The original repository remains the location of preserved held-out data.
    test_folder = Path(previous["path"])
    if not test_folder.exists():
        test_folder = Path("/data2/zh/AuditEmbodied") / previous["path"]
    validity["previous_test_files_unchanged"] = all(hashlib.sha256((test_folder/n).read_bytes()).hexdigest() == h
                                                   for n, h in previous["files_sha256"].items())
    validity["baseline_controllers_unchanged"] = all((Path(__file__).parent/n).read_bytes() ==
        (output/"baseline-source"/n).read_bytes() for n in ("continuous.py", "recovery_skills.py", "recovery_rules.py"))
    report = {"scope": "training/development only; visual selector not yet evaluated on new test seeds",
              "validity": validity, **results}
    fields = ("split", "seed", "scenario", "candidate", "success", "protection_violation", "recovery_seconds",
              "recovery_path_m", "policy_observations", "failure_reason")
    with (output/"physical-results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({k:r.get(k) for k in fields} for r in all_rows)
    (output/"physical-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("pilot", "train", "dev"), default="pilot")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        print(json.dumps(summarize(args.output_dir), ensure_ascii=False))
    else:
        collect(args.output_dir, args.split, args.workers, args.visual)


if __name__ == "__main__":
    main()
