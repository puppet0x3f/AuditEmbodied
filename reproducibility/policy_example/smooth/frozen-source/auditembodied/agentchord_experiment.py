"""Fresh-seed comparison of the adapted AgentChord runtime and frozen baselines."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import itertools
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from .admissible_mlp import StartGatedMLP, verify
from .agentchord_style import AgentChordStyle, graph_spec
from .duration_pilot import file_hash
from .l4_lift import SCENARIOS, LiftConfig, run_trial
from .l4_selector import ProprioRule, safe
from .p0 import method_metrics, outcome, paired_statistics, same_prefix, save_json


METHODS = ['protected', 'gated_direct', 'agentchord_style', 'global']
HEALTHY_METHODS = METHODS[:-1]


def prepare(output, parent, seeds, audit, runtime_check):
    output, parent, seeds = Path(output), Path(parent), list(seeds)
    if output.exists():
        raise ValueError('output exists; preserve observed results')
    if audit['used_overlap'] or audit['requested_seeds'] != seeds:
        raise ValueError('cohort does not match fresh-seed audit')
    prior = json.loads((parent / 'protocol.json').read_text())
    if file_hash(prior['model_path']) != prior['model_sha256']:
        raise ValueError('parent frozen model changed')
    changed = [p for p, digest in prior['source_sha256'].items() if file_hash(p) != digest]
    if changed != ['auditembodied/l4_lift.py'] or not runtime_check['passed']:
        raise ValueError('only the checked optional graph hook may change the parent runtime')
    if runtime_check['changed_sources']['auditembodied/l4_lift.py']['current'] != file_hash('auditembodied/l4_lift.py'):
        raise ValueError('runtime check does not cover the current source')
    sources = [*prior['source_sha256'], 'auditembodied/agentchord_graph.py',
               'auditembodied/agentchord_style.py', 'auditembodied/agentchord_experiment.py']
    protocol = {'scope': 'AgentChord-style graph recovery; fixed template and shared physical skills; no test tuning',
        'parent': str(parent.resolve()), 'model_path': prior['model_path'],
        'model_sha256': prior['model_sha256'], 'threshold': prior['threshold'],
        'rule': prior['rule'], 'test_config': prior['test_config'],
        'test_seeds': seeds, 'scenarios': list(SCENARIOS), 'methods': METHODS,
        'healthy_methods': HEALTHY_METHODS, 'expected_episodes': len(seeds) * 15,
        'graph_spec': graph_spec(prior['rule']),
        'observations': {'keys': ['positions', 'fingers', 'reference_positions'],
            'control_hz': 20, 'rgb_frames': 0,
            'policy_access': 'same robot stream permitted to every method; frozen one-shot policies do not revise choices',
            'excluded': ['contact truth', 'object state', 'scenario label', 'seed', 'future frames'],
            'graph_monitors': 'registered encoder load bounds and relative robot height only'},
        'skills': 'unchanged continue/R_left/R_right/R_global; local relevel40; same geometry and actuator settings',
        'cost': {'initial_decision': 'measured callback rounded up to physical20Hz hold for every method',
            'online_monitor': 'measured callback within shared50ms control period; excess replayed as physical holds',
            'total': 'all attempted skills + all physical waits + final1s hold, in one continuous environment',
            'budget': 'upstream graph max_transitions8; shared environment horizon2000 control steps',
            'protection': 'cumulative across every attempted edge; never reset after recovery switching'},
        'seed_audit': audit, 'default_runtime_check': runtime_check,
        'source_sha256': {p: file_hash(p) for p in sources}}
    output.mkdir(parents=True)
    save_json(output / 'protocol.json', protocol)
    save_json(output / 'graph.json', protocol['graph_spec'])
    return protocol


def trial_job(job):
    seed, scenario, method, config, rule, values, spec = job
    graph = AgentChordStyle(spec) if method == 'agentchord_style' else None
    selector = (graph if graph is not None else ProprioRule(**rule) if method == 'protected'
        else StartGatedMLP(values) if method == 'gated_direct' else lambda port: {'candidate': 'R_global'})
    started = time.perf_counter()
    row = run_trial(seed, scenario, config=LiftConfig(**config), selector=selector,
                    account_selector_latency=True, recovery_graph=graph)
    row.update(method=method, wall_seconds=time.perf_counter() - started)
    return row


def run(output, workers=4):
    output = Path(output)
    protocol = json.loads((output / 'protocol.json').read_text())
    verify(protocol)
    values = json.loads(Path(protocol['model_path']).read_text())['mlp']
    jobs = [(s, q, m, protocol['test_config'], protocol['rule'], values, protocol['graph_spec'])
        for s in protocol['test_seeds'] for q in (*SCENARIOS, 'sham')
        for m in (METHODS if q != 'sham' else HEALTHY_METHODS)]
    with (output / 'trials.jsonl').open('x') as stream:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for i, row in enumerate(pool.map(trial_job, jobs), 1):
                stream.write(json.dumps(row) + '\n');stream.flush()
                print(json.dumps({'completed': i, 'total': len(jobs),
                    **{k: row.get(k) for k in ('seed', 'scenario', 'method', 'candidate', 'success', 'protection_violation', 'failure_reason')},
                    'skills': row.get('graph', {}).get('executed_skills')}), flush=True)
    verify(protocol)


def analyze(output):
    output = Path(output)
    protocol = json.loads((output / 'protocol.json').read_text())
    verify(protocol)
    rows = [json.loads(line) for line in (output / 'trials.jsonl').read_text().splitlines()]
    if len(rows) != protocol['expected_episodes']:
        raise ValueError('incomplete cohort')
    groups = {}
    for row in rows:
        groups.setdefault((row['seed'], row['scenario']), []).append(row)
    faults = [r for r in rows if r['scenario'] != 'sham']
    graph_rows = [r for r in faults if r['method'] == 'agentchord_style']
    gated = [r for r in rows if r['method'] == 'gated_direct']
    values = json.loads(Path(protocol['model_path']).read_text())['mlp']
    model = StartGatedMLP(values)
    def observed_losses(row):
        return sum(not f['evaluator']['holds'][1 if f['skill'] == 'R_left' else 0]
            for f in row['trace'] if f['phase'] == 'recovery' and f.get('skill') in ('R_left', 'R_right'))
    validity = {'episodes': len(rows), 'expected_episodes': protocol['expected_episodes'],
        'complete_method_groups': set(groups) == {(s, q) for s in protocol['test_seeds'] for q in (*SCENARIOS, 'sham')}
            and all({r['method'] for r in group} == set(HEALTHY_METHODS if key[1] == 'sham' else METHODS)
                    for key, group in groups.items()),
        'all_paired_prefixes_equal': all(same_prefix(group) for group in groups.values()),
        'initialization_failures': sum(not r.get('nominal_ready', False) for r in rows),
        'invalid_faults': sum(not r.get('fault_class_valid', False) for r in rows),
        'execution_errors': sum(bool(r.get('error')) for r in rows),
        'numerical_instabilities': sum(r.get('failure_reason') == 'numerical_instability' for r in rows),
        'contact_anchors': sum(r['contact_anchor_count'] for r in rows),
        'state_edits': sum(r['state_edit_count_after_initialization'] for r in rows),
        'forbidden_calls': sum(len(r['blocked_calls']) for r in rows),
        'complete_costs': sum('total_post_fault_seconds' in r for r in rows),
        'cost_identity': all('total_post_fault_seconds' in r and abs(r['total_post_fault_seconds'] -
            r['skill_seconds'] - r['inference_wait_seconds'] - r['hold_seconds']) < 1e-8 for r in rows),
        'wait_steps_match_trace': all(sum(f['stage'] in ('inference_wait', 'monitor_overrun_wait')
            for f in r['trace']) == r['inference_wait_steps'] for r in rows),
        'frozen_mlp_decisions_reproduced': all('selection' in r and
            model(r['selection']['input']) == r['selection']['decision'] for r in gated),
        'graph_keeper_losses_cumulative': all(observed_losses(r) == r['protection']['keeper_grasp_lost_steps'] for r in graph_rows),
        'graph_monitor_ports_only': all(set(e['input']) == {'positions', 'fingers', 'reference_positions'}
            for r in graph_rows for e in r.get('skill_events', [])),
        'frozen_model_and_sources_unchanged': True}
    common = {key for key, group in groups.items() if key[1] != 'sham' and all(safe(r) for r in group)}
    pairs = list(itertools.combinations(METHODS, 2))
    graph_summary = {'episodes': len(graph_rows),
        'actual_skill_sequences': dict(Counter(' -> '.join(r.get('graph', {}).get('executed_skills', [])) or 'continue' for r in graph_rows)),
        'multi_skill_episodes': sum(len(r.get('graph', {}).get('executed_skills', [])) > 1 for r in graph_rows),
        'used_global': sum('R_global' in r.get('graph', {}).get('executed_skills', []) for r in graph_rows),
        'keeper_monitor_interruptions': sum(e['monitor_index'] is not None and any('keeper_missing' in n for n in e['monitors'])
            for r in graph_rows for e in r.get('graph', {}).get('events', [])),
        'completion_monitor_interruptions': sum(e['monitor_index'] is not None and e['monitors'] == ['incomplete']
            for r in graph_rows for e in r.get('graph', {}).get('events', [])),
        'budget_exhaustions': sum(not r.get('graph', {}).get('completed', False) for r in graph_rows),
        'terminal_but_unsafe': sum(r.get('task_complete', False) and r['protection_violation'] for r in graph_rows),
        'monitor_calls': sum(r.get('monitor_calls', 0) for r in graph_rows),
        'monitor_wall_seconds': sum(r.get('monitor_wall_seconds', 0.) for r in graph_rows),
        'max_monitor_wall_seconds': max((r.get('max_monitor_wall_seconds', 0.) for r in graph_rows), default=0.),
        'monitor_wait_seconds': sum(r.get('monitor_wait_seconds', 0.) for r in graph_rows)}
    report = {'scope': protocol['scope'], 'test_seeds': protocol['test_seeds'], 'validity': validity,
        'fault_results': {m: method_metrics([r for r in faults if r['method'] == m]) for m in METHODS},
        'healthy_controls': {m: method_metrics([r for r in rows if r['scenario'] == 'sham' and r['method'] == m]) for m in HEALTHY_METHODS},
        'by_scenario': {q: {m: method_metrics([r for r in faults if r['scenario'] == q and r['method'] == m]) for m in METHODS} for q in SCENARIOS},
        'outcomes': {m: dict(Counter(outcome(r) for r in faults if r['method'] == m)) for m in METHODS},
        'paired': {f'{a}_vs_{b}': paired_statistics(faults, a, b) for a, b in pairs},
        'all_four_common_safe': {'states': len(common),
            'metrics': {m: method_metrics([r for r in faults if r['method'] == m and (r['seed'], r['scenario']) in common]) for m in METHODS},
            'paired': {f'{a}_vs_{b}': paired_statistics([r for r in faults if (r['seed'], r['scenario']) in common], a, b) for a, b in pairs}},
        'graph': graph_summary,
        'failures': [{**{k: r.get(k) for k in ('seed', 'scenario', 'method', 'candidate', 'success', 'task_complete',
            'protection_violation', 'protection', 'failure_reason', 'graph', 'error')},
            'initial_decision': r.get('selection', {}).get('decision')} for r in faults if not safe(r)]}
    save_json(output / 'report.json', report)
    fields = ['split', 'method', 'episodes', 'safe_successes', 'terminal_successes', 'protection_violations',
              'mean_total_seconds', 'mean_skill_seconds', 'mean_wait_seconds', 'mean_recovery_path_m']
    with (output / 'main-results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields);writer.writeheader()
        for split, results in [('fault', report['fault_results']), ('healthy', report['healthy_controls']),
                                ('four_common_safe', report['all_four_common_safe']['metrics'])]:
            for method, result in results.items():
                writer.writerow({'split': split, 'method': method, **{k: result[k] for k in fields[2:]}})
    fields = ['seed', 'scenario', 'method', 'candidate', 'success', 'protection_violation',
              'total_post_fault_seconds', 'skill_seconds', 'inference_wait_seconds', 'selector_wall_seconds',
              'monitor_wall_seconds', 'recovery_path_m', 'failure_reason']
    with (output / 'cases.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields);writer.writeheader()
        writer.writerows({k: r.get(k) for k in fields} for r in rows)
    figures(output, report)
    return report


def figures(output, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Protected rule', 'Frozen MLP + start gate', 'AgentChord-style', 'Fixed global']
    colors = ['#3973ac', '#d07825', '#258672', '#899097']
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for i, (method, label, color) in enumerate(zip(METHODS, labels, colors)):
        r = report['fault_results'][method]
        value = 100 * r['safe_successes'] / r['episodes']
        axes[0].barh(label, value, color=color)
        axes[0].text(value + .6, i, f"{r['safe_successes']}/{r['episodes']}", va='center')
        value = report['all_four_common_safe']['metrics'][method]['mean_total_seconds']
        if value is not None:
            axes[1].barh(label, value, color=color)
            axes[1].text(value + .1, i, f'{value:.3f}', va='center')
    axes[0].set_xlim(0, 115);axes[0].set_xlabel('Safe completion (%)')
    axes[1].set_xlim(0, 15);axes[1].set_xlabel('Total post-fault seconds\nSame ' + str(report['all_four_common_safe']['states']) + ' common-safe states')
    for axis in axes:
        axis.invert_yaxis();axis.grid(axis='x', alpha=.2)
    fig.suptitle('Shared robot observations and skills; fresh paired initialization seeds')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'comparison.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=['prepare', 'run', 'analyze'])
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--parent', default='outputs/startgate-mlp-20261109')
    parser.add_argument('--seed-start', type=int, default=12000)
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--seed-audit');parser.add_argument('--runtime-check')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.phase == 'prepare':
        protocol = prepare(args.output_dir, args.parent, range(args.seed_start, args.seed_start + args.seeds),
            json.loads(Path(args.seed_audit).read_text()), json.loads(Path(args.runtime_check).read_text()))
        print(json.dumps({'prepared': True, 'episodes': protocol['expected_episodes']}))
    elif args.phase == 'run':
        run(args.output_dir, args.workers)
    else:
        report = analyze(args.output_dir)
        print(json.dumps({'validity': report['validity'], 'fault_results': report['fault_results'], 'graph': report['graph']}))


if __name__ == '__main__':
    main()
