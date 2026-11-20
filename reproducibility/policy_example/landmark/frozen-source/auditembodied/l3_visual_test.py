"""Frozen proprioceptive control and paired held-out visual L3 evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import re

import numpy as np

from .continuous import _proprio
from .l0 import _jsonable
from .l3_matrix import MatrixConfig, run_trial, _summary
from .l3_visual import RGBRecorder
from .recovery_skills import actual_eef_rotations, actual_gripper_positions
from .visual_recovery import MultiSupportVerifier, VisualRecoverySelector


METHODS = ("rule", "proprio", "visual", "R3")
WARNING_NAMES = ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC", "mjWARN_BADCTRL")


class DecisionObserver(RGBRecorder):
    """Acquire only policy inputs; the shared runner audits physics at 20 Hz.

    The robot model receives an empty RGB port. All methods initialize the
    same camera-enabled environment, with automatic camera sensors disabled.
    Thus only the visual selector consumes RGB after initialization.
    """

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def after_reset(self, env):
        super().after_reset(env)
        import mujoco
        self.env = env
        self.warning_before = {k: int(env.sim.data.warning[int(getattr(mujoco.mjtWarning, k))].number)
                               for k in WARNING_NAMES}

    def register(self, reference_gaps):
        super().register(reference_gaps)
        import mujoco
        self.initialization_warnings = {
            k: int(self.env.sim.data.warning[int(getattr(mujoco.mjtWarning, k))].number) - self.warning_before[k]
            for k in WARNING_NAMES}
        if any(self.initialization_warnings.values()):
            raise FloatingPointError("numerical instability during physical bootstrap")

    def capture(self, env, packet, force=False):
        pass  # Truth stays in the shared runner's evaluator trace.

    def observe(self, env):
        if self.mode == "image_robot":
            observed = super().observe(env)
            self.rgb.append(observed["rgb"])
        elif self.mode == "robot":
            self.policy_observations += 1
            fingers = actual_gripper_positions(env)
            errors = np.max(np.abs(np.abs(fingers) - self.reference_gaps[:, None]), axis=1) / self.reference_gaps
            robot = np.r_[_proprio(env), fingers.reshape(-1), self.reference_gaps,
                          actual_eef_rotations(env).reshape(-1), errors].astype(np.float32)
            observed = {"rgb": np.empty(0, dtype=np.uint8), "robot": robot}
        else:
            raise ValueError("fixed candidate has no learned observation callback")
        self.robot.append(observed["robot"])
        self.times.append(float(env.sim.data.time))
        return observed

    def save_inputs(self, path):
        if self.policy_observations:
            np.savez_compressed(path, rgb=np.asarray(self.rgb, dtype=np.uint8),
                                robot=np.asarray(self.robot), time=np.asarray(self.times), mode=self.mode)


def paired_prefix(group):
    rows = list(group.values())
    equal = lambda key: all(r.get(key) is not None and r[key] == rows[0].get(key) for r in rows)
    reached = ["prefix_state" in r for r in rows]
    uniform_failure = not any(reached) and all(r.get("failure_reason") in
        ("initialization_error", "initialization_not_supported", "numerical_instability") for r in rows)
    return {"initialization_identical": equal("initialization_state"),
            "initialization_end_identical": equal("initialization_end_state"),
            "fault_prefix_reached": all(reached),
            "fault_prefix_identical": equal("prefix_state") if all(reached) else None,
            "uniform_initialization_failure": uniform_failure,
            "comparable": equal("initialization_state") and equal("initialization_end_state")
                          and (equal("prefix_state") or uniform_failure)}


def safe(row):
    return bool(row["success"] and not row["protection_violation"])


def paired_cost(groups, reference):
    pairs = [(g["visual"], g[reference]) for g in groups.values() if safe(g["visual"]) and safe(g[reference])]
    out = {"scenes": len(pairs)}
    if pairs:
        for key in ("recovery_seconds", "recovery_path_m"):
            a = float(np.mean([r[key] for r, _ in pairs])); b = float(np.mean([r[key] for _, r in pairs]))
            out[key] = {"visual": a, reference: b, "reduction_fraction": 1-a/b if b else None}
    return out


def intervals(groups, draws=2000):
    seeds = sorted({s for s, _ in groups})
    indices = np.random.default_rng(20261107).integers(0, len(seeds), (draws, len(seeds)))
    result = {"resampling_unit": "initialization seed; all five faults stay together", "draws": draws}
    for method in ("rule", "proprio", "R3"):
        delta = np.array([np.mean([int(safe(g["visual"]))-int(safe(g[method]))
                                 for (s, _), g in groups.items() if s == seed]) for seed in seeds])
        result[f"safe_success_delta_vs_{method}"] = {
            "point": float(delta.mean()), "ci95": np.quantile(delta[indices].mean(1), [.025, .975]).tolist()}
        for key in ("recovery_seconds", "recovery_path_m"):
            totals = np.array([(sum(g["visual"][key] for (s, _), g in groups.items()
                                   if s == seed and safe(g["visual"]) and safe(g[method])),
                               sum(g[method][key] for (s, _), g in groups.items()
                                   if s == seed and safe(g["visual"]) and safe(g[method]))) for seed in seeds])
            a, b = totals[indices, 0].sum(1), totals[indices, 1].sum(1)
            ratios = 1-a[b > 0]/b[b > 0]
            result[f"paired_{key}_reduction_vs_{method}"] = {
                "ci95": np.quantile(ratios, [.025, .975]).tolist()} if len(ratios) else None
    return result


def verify_frozen(output, manifest):
    for name in manifest["source_files"]:
        if (Path(__file__).parent/name).read_bytes() != (output/"frozen-source"/name).read_bytes():
            raise ValueError(f"frozen source changed: {name}")
    for name in ("visual-support.npz", "support-robot.npz"):
        if (output/name).read_bytes() != (output/"frozen-source"/name).read_bytes():
            raise ValueError(f"frozen model changed: {name}")
    for reference in manifest["preserved_results"]:
        root = Path(reference["root"])
        if not root.exists():
            root = Path("/data2/zh/AuditEmbodied")/root
        for name, expected in reference["sha256"].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"previous result changed: {root/name}")


def trial_job(job):
    seed, scenario, method, protocol, output, phase = job
    output = Path(output)
    mode = {"visual": "image_robot", "proprio": "robot"}.get(method)
    observer = DecisionObserver(mode)
    policy = None
    if mode:
        path = output/("visual-support.npz" if method == "visual" else "support-robot.npz")
        model = MultiSupportVerifier.load(path)
        if model.mode != mode:
            raise ValueError("wrong modality checkpoint")
        policy = VisualRecoverySelector(model)
    row = run_trial(seed, scenario, "visual" if mode else method, MatrixConfig(**protocol),
                    observer=observer, recovery_policy=policy)
    row["candidate"] = method
    row["model_mode"] = mode
    row["policy_rgb_bundles"] = len(observer.rgb)
    row["initialization_bad_warnings"] = getattr(observer, "initialization_warnings", {})
    if mode:
        observer.save_inputs(output/phase/"inputs"/f"{seed}-{scenario}-{method}.npz")
    return row


def read_rows(path):
    rows = []
    for line in path.open():
        row = json.loads(line); trace = row.pop("trace")
        row["trace_monotonic"] = bool(np.all(np.diff([f["time"] for f in trace]) > 0))
        rows.append(row)
    return rows


def validity(rows, groups, source_frozen=True):
    checks = [{"seed": s, "scenario": q, **paired_prefix(g)} for (s, q), g in groups.items()]
    seeds = {s for s, _ in groups}
    initialized = [s for s in seeds if all(r.get("initialization_supported", False)
                    for (seed, _), g in groups.items() if seed == s for r in g.values())]
    return {"actual_episodes": len(rows), "scenes": len(groups), "all_sources_frozen": source_frozen,
        "all_cases_comparable": all(r["comparable"] for r in checks), "prefix_checks": checks,
        "initialized_seeds": sorted(initialized), "initialized_seed_rate": len(initialized)/len(seeds),
        "numerical_instabilities": sum(r["failure_reason"] == "numerical_instability" or
            any(r.get("bad_state_warnings", {}).values()) or any(r.get("initialization_bad_warnings", {}).values())
            or not r["trace_monotonic"] for r in rows),
        "execution_errors": sum(r["failure_reason"] == "execution_error" for r in rows),
        "contact_anchors": sum(r["contact_anchor_count"] for r in rows),
        "manual_state_edits": sum(r["state_edit_count_after_initialization"] for r in rows),
        "forbidden_calls": sum(len(r["blocked_calls"]) for r in rows),
        "wrench_leaks": sum(r.get("external_wrench_leak_steps", 0) or bool(r.get("external_wrench_at_recovery_start", 0)) for r in rows)}


def verdict(report, criteria):
    p, v = report["by_method"], report["validity"]
    rate = lambda k: p[k]["safe_successes"]/p[k]["episodes"]
    lower = lambda method: bool(report["paired_costs"][method]["scenes"] and all(
        report["paired_costs"][method][key]["reduction_fraction"] is not None and
        report["paired_costs"][method][key]["reduction_fraction"] > 0
        for key in ("recovery_seconds", "recovery_path_m")))
    checks = {"valid_protocol": v["all_cases_comparable"] and v["all_sources_frozen"] and not any(v[k]
        for k in ("numerical_instabilities", "execution_errors", "contact_anchors", "manual_state_edits", "forbidden_calls", "wrench_leaks")),
        "initialization": v["initialized_seed_rate"] >= criteria["minimum_initialization_seed_rate"],
        "higher_safe_success_than_rule": rate("visual") > rate("rule"),
        "not_lower_safe_success_than_proprio": rate("visual") >= rate("proprio"),
        "independent_visual_benefit": rate("visual") > rate("proprio") or lower("proprio"),
        "no_more_protection_violations": all(p["visual"]["protection_violation_rate"] <= p[k]["protection_violation_rate"] for k in ("rule", "proprio")),
        "near_global_success": rate("visual") >= rate("R3")-criteria["global_success_margin"],
        "lower_paired_cost_than_global": lower("R3")}
    return {"decision": "Go" if all(checks.values()) else "No-Go", "checks": checks,
            "failed_checks": [k for k, value in checks.items() if not value]}


def analyze(output, phase):
    output = Path(output); m = json.loads((output/"frozen-experiment.json").read_text())
    verify_frozen(output, m)
    rows = read_rows(output/phase/"trials.jsonl")
    methods = METHODS if phase == "test" else ("rule", "proprio", "visual")
    if phase == "dev":
        ref = Path(m["reference_root"])
        rows.extend(r for r in read_rows(ref/"dev/trials.jsonl") if r["candidate"] == "rule")
        rows.extend(read_rows(ref/"online-dev/trials.jsonl"))
    groups = {}
    for row in rows:
        group = groups.setdefault((row["seed"], row["scenario"]), {})
        if row["candidate"] in group:
            raise ValueError("duplicate paired episode")
        group[row["candidate"]] = row
    expected = {(s, q) for s in m[phase+"_seeds"] for q in m["scenarios"]}
    if set(groups) != expected or any(set(g) != set(methods) for g in groups.values()):
        raise ValueError("paired run is incomplete")
    report = {"scope": phase, "seeds": m[phase+"_seeds"], "methods": list(methods),
        "by_method": {k:_summary([g[k] for g in groups.values()]) for k in methods},
        "by_scenario": {q:{k:_summary([g[k] for (s, scene),g in groups.items() if scene == q]) for k in methods} for q in m["scenarios"]},
        "validity": validity(rows, groups),
        "paired_costs": {k:paired_cost(groups,k) for k in methods if k != "visual"},
        "sequences": {k:dict(Counter(" -> ".join(g[k].get("recovery", {}).get("sequence", [k])) for g in groups.values())) for k in methods},
        "mean_policy_observations": {k:float(np.mean([g[k].get("policy_observations", 0) for g in groups.values()])) for k in methods},
        "mean_policy_rgb_bundles": {k:float(np.mean([g[k].get("policy_rgb_bundles", g[k].get("policy_observations", 0) if k == "visual" else 0) for g in groups.values()])) for k in methods},
        "failures": [{k:r.get(k) for k in ("seed", "scenario", "candidate", "failure_reason", "error", "failed_stage", "recovery", "post_fault_predicate", "post_recovery_predicate", "table_cycle_verified", "protection")} for r in rows if not safe(r)]}
    report["new_physical_episodes"] = len(m[phase+"_seeds"])*len(m["scenarios"])*(4 if phase == "test" else 1)
    report["validity"]["previous_results_unchanged"] = True
    report["validity"]["proprio_uses_zero_rgb_bundles"] = all(r.get("policy_rgb_bundles") == 0 for r in rows if r["candidate"] == "proprio")
    if phase == "test":
        report["seed_bootstrap"] = intervals(groups)
        report["go_no_go"] = verdict(report, m["criteria"])
    log = output/f"{phase}.log"
    warning_times = [float(s) for s in re.findall(r"WARNING: .*?Time = ([0-9.]+)\.",log.read_text())] if log.exists() else []
    report["native_setup_qacc_log_events"] = len(warning_times)
    report["native_warning_simulation_times_seconds"] = dict(Counter(warning_times))
    report["numerical_scope"] = "formal monitor starts after env.reset; bootstrap warnings abort, task warnings/time rollback abort; native construction/reset log warnings are separate"
    (output/phase/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    fields = ("seed", "scenario", "candidate", "success", "protection_violation", "recovery_seconds", "recovery_path_m", "policy_observations", "policy_rgb_bundles", "failure_reason")
    with (output/phase/"cases.csv").open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r.get(k) for k in fields} for r in rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", choices=("dev", "test"), required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(); output = Path(args.output_dir)
    m = json.loads((output/"frozen-experiment.json").read_text()); verify_frozen(output,m)
    if not args.analyze_only:
        if args.phase == "test" and not (output/"dev/report.json").exists():
            raise ValueError("complete and report the proprio development control first")
        folder = output/args.phase; (folder/"inputs").mkdir(parents=True,exist_ok=True)
        if (folder/"trials.jsonl").exists():
            raise ValueError("data already exists; analyze only, never overwrite held-out trials")
        methods = ("proprio",) if args.phase == "dev" else METHODS
        jobs = [(s,q,k,m["protocol"],str(output),args.phase) for s in m[args.phase+"_seeds"] for q in m["scenarios"] for k in methods]
        with (folder/"trials.jsonl").open("w") as log:
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
                for row in pool.map(trial_job,jobs):
                    log.write(json.dumps(_jsonable(row))+"\n");log.flush()
                    print(json.dumps({k:row.get(k) for k in ("seed","scenario","candidate","success","failure_reason")}),flush=True)
        if json.loads((output/"frozen-experiment.json").read_text()) != m:
            raise ValueError("experiment definition changed during evaluation")
    r = analyze(output,args.phase)
    print(json.dumps({"by_method":r["by_method"],"go_no_go":r.get("go_no_go")}),flush=True)


if __name__ == "__main__":
    main()
