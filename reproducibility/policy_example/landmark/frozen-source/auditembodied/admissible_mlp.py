"""Frozen candidate MLP with observable start prerequisites, on fresh seeds."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from .duration_pilot import file_hash
from .l4_lift import CANDIDATES, SCENARIOS, LiftConfig, run_trial
from .l4_offline_selection import features
from .l4_selector import ProprioRule, safe
from .p0 import CandidateMLP, method_metrics, paired_statistics, same_prefix, save_json


def start_mask(x):
    """No fragility or completion test: only the already registered load proxy."""
    left, right = bool(x[2]), bool(x[3])
    return np.array([left and right, right, left, True], dtype=bool)


class StartGatedMLP:
    def __init__(self, values):
        self.model = CandidateMLP(values)

    def __call__(self, port):
        model = self.model
        x = features(port, model.rule)
        probabilities = model.probabilities(x)[0]
        mask = start_mask(x)
        order = np.argsort(model.values['action_cost_seconds'], kind='stable')
        threshold = model.values['threshold']
        choose = lambda gated: next((CANDIDATES[j] for j in order
            if probabilities[j] >= threshold and (not gated or mask[j])), 'abstain')
        return {'candidate': choose(True), 'ungated_candidate': choose(False),
                'safety_scores': probabilities.tolist(), 'threshold': threshold,
                'start_admissible': mask.tolist(), 'loaded': x[2:4].astype(bool).tolist()}


def prepare(output, parent, seeds, audit, runtime_check=None):
    output, parent = Path(output), Path(parent)
    seeds = list(seeds)
    if output.exists():
        raise ValueError('output exists; never replace observed test data')
    if audit['used_overlap'] or seeds != audit['requested_seeds']:
        raise ValueError('fresh-seed audit does not match requested cohort')
    prior = json.loads((parent / 'protocol.json').read_text())
    model_path = (parent / 'direct-selected.json').resolve()
    direct = json.loads(model_path.read_text())
    if direct['kind'] != 'candidate_mlp' or file_hash(model_path) != prior['direct_model_sha256']:
        raise ValueError('parent selected MLP changed')
    current_sources = {path: file_hash(path) for path in prior['physics_source_sha256']}
    changed = {path: {'parent': digest, 'current': current_sources[path]}
               for path, digest in prior['physics_source_sha256'].items() if current_sources[path] != digest}
    if changed:
        if (set(changed) != {'auditembodied/l4_lift.py'} or not runtime_check or
                not runtime_check['passed'] or changed != runtime_check['changed_sources']):
            raise ValueError('parent runtime differs without a matching default-path compatibility check')
    source = str(Path(__file__).resolve())
    protocol = {'scope': 'fresh-state frozen-MLP start-prerequisite control; no fitting or calibration',
        'parent': str(parent.resolve()), 'model_path': str(model_path),
        'model_sha256': file_hash(model_path), 'threshold': direct['mlp']['threshold'],
        'rule': prior['rule'], 'test_config': prior['test_config'],
        'test_seeds': seeds, 'methods': ['protected', 'gated_direct', 'global'],
        'scenarios': list(SCENARIOS), 'expected_episodes': len(seeds) * 11,
        'gate': {'load_bounds_m': [direct['rule']['loaded_lower'], direct['rule']['loaded_upper']],
            'continue': 'both encoder load proxies present',
            'R_left': 'right keeper load proxy present',
            'R_right': 'left keeper load proxy present', 'R_global': 'unmasked',
            'excluded_rules': ['no keeper asymmetry limit', 'no height-completion rule', 'no contact truth'],
            'selection': 'unchanged MLP probabilities, threshold and nominal-cost order; mask only'},
        'unchanged': ['weights', 'normalization', 'threshold', 'nominal action costs', 'skills',
                      'perturbation amplitudes', 'success and protection conditions'],
        'cost': 'actual selector callback time rounded up to physical20Hz hold + feedback motion + final1s hold; wall time separate',
        'seed_audit': audit, 'parent_source_changes': changed, 'parent_runtime_check': runtime_check,
        'source_sha256': {**current_sources, source: file_hash(source)}}
    output.mkdir(parents=True)
    save_json(output / 'protocol.json', protocol)
    return protocol


def verify(protocol):
    if file_hash(protocol['model_path']) != protocol['model_sha256']:
        raise ValueError('frozen parent model changed')
    if any(file_hash(path) != digest for path, digest in protocol['source_sha256'].items()):
        raise ValueError('frozen runtime changed')


def trial_job(job):
    seed, scenario, method, config, rule, values = job
    selector = (ProprioRule(**rule) if method == 'protected' else StartGatedMLP(values)
                if method == 'gated_direct' else lambda port: {'candidate': 'R_global'})
    started = time.perf_counter()
    row = run_trial(seed, scenario, config=LiftConfig(**config), selector=selector,
                    account_selector_latency=True)
    row.update(method=method, wall_seconds=time.perf_counter() - started)
    return row


def run(output, workers=4):
    output = Path(output)
    protocol = json.loads((output / 'protocol.json').read_text())
    verify(protocol)
    values = json.loads(Path(protocol['model_path']).read_text())['mlp']
    jobs = [(s, q, m, protocol['test_config'], protocol['rule'], values)
            for s in protocol['test_seeds'] for q in (*SCENARIOS, 'sham')
            for m in (protocol['methods'] if q != 'sham' else ['protected', 'gated_direct'])]
    with (output / 'trials.jsonl').open('x') as stream:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for i, row in enumerate(pool.map(trial_job, jobs), 1):
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                print(json.dumps({'completed': i, 'total': len(jobs),
                    **{k: row.get(k) for k in ('seed', 'scenario', 'method', 'candidate', 'success', 'protection_violation', 'failure_reason')}}), flush=True)
    verify(protocol)


def local_precondition(row):
    if row['candidate'] not in ('R_left', 'R_right'):
        return None
    keeper = 1 if row['candidate'] == 'R_left' else 0
    present = row.get('post_fault', {}).get('holds', [False, False])[keeper]
    losses = [f for f in row['trace'] if f['phase'] == 'recovery' and not f['evaluator']['holds'][keeper]]
    return {'keeper_present_at_start': present,
        'keeper_interrupted_after_valid_start': bool(present and losses),
        'first_loss_stage': losses[0]['stage'] if losses else None,
        'keeper_loss_steps': len(losses)}


def analyze(output):
    output = Path(output)
    protocol = json.loads((output / 'protocol.json').read_text())
    verify(protocol)
    rows = [json.loads(s) for s in (output / 'trials.jsonl').read_text().splitlines()]
    if len(rows) != protocol['expected_episodes']:
        raise ValueError('incomplete test cohort')
    groups = {}
    for row in rows:
        groups.setdefault((row['seed'], row['scenario']), []).append(row)
    expected = {(s, q) for s in protocol['test_seeds'] for q in (*SCENARIOS, 'sham')}
    faults = [r for r in rows if r['scenario'] != 'sham']
    gated = [r for r in rows if r['method'] == 'gated_direct']
    gated_faults = [r for r in faults if r['method'] == 'gated_direct']
    values = json.loads(Path(protocol['model_path']).read_text())['mlp']
    gate = StartGatedMLP(values)
    raw = CandidateMLP(values)
    scores_equal, decisions_equal = True, True
    for row in gated:
        if 'selection' not in row:
            scores_equal = decisions_equal = False
            continue
        port = row['selection']['input']
        decision = gate(port)
        decisions_equal &= decision == row['selection']['decision']
        scores_equal &= np.array_equal(decision['safety_scores'], raw(port)['safety_scores'])
    validity = {'episodes': len(rows), 'expected_episodes': protocol['expected_episodes'],
        'all_expected_states': set(groups) == expected,
        'complete_method_groups': all({r['method'] for r in g} ==
            (set(protocol['methods']) if key[1] != 'sham' else {'protected', 'gated_direct'})
            for key, g in groups.items()),
        'all_paired_prefixes_equal': all(same_prefix(g) for g in groups.values()),
        'initialization_failures': sum(not r.get('nominal_ready', False) for r in rows),
        'invalid_faults': sum(not r.get('fault_class_valid', False) for r in rows),
        'execution_errors': sum(bool(r.get('error')) for r in rows),
        'numerical_instabilities': sum(r.get('failure_reason') == 'numerical_instability' for r in rows),
        'state_edits': sum(r['state_edit_count_after_initialization'] for r in rows),
        'contact_anchors': sum(r['contact_anchor_count'] for r in rows),
        'forbidden_calls': sum(len(r['blocked_calls']) for r in rows),
        'complete_costs': sum('total_post_fault_seconds' in r for r in rows),
        'cost_identity': all('total_post_fault_seconds' in r and abs(r['total_post_fault_seconds'] -
            r['skill_seconds'] - r['inference_wait_seconds'] - r['hold_seconds']) < 1e-8 for r in rows),
        'wait_steps_match_trace': all(len([f for f in r['trace'] if f['stage'] == 'inference_wait']) ==
            r['inference_wait_steps'] for r in rows),
        'all_frozen_decisions_reproduced': bool(decisions_equal), 'raw_mlp_probabilities_exact': bool(scores_equal),
        'threshold_unchanged': all(r['selection']['decision']['threshold'] == protocol['threshold']
            for r in gated if 'selection' in r), 'frozen_model_and_sources_unchanged': True}
    failure_cases = [{**{k: r.get(k) for k in ('seed', 'scenario', 'method', 'candidate', 'success',
        'task_complete', 'protection_violation', 'failure_reason', 'protection', 'error')},
        'local_start': local_precondition(r), 'selection': r.get('selection')} for r in rows if not safe(r)]
    counts = {}
    for method in protocol['methods']:
        locals_ = [local_precondition(r) for r in faults if r['method'] == method]
        locals_ = [x for x in locals_ if x is not None]
        counts[method] = {'local_selected': len(locals_),
            'missing_keeper_precondition': sum(not x['keeper_present_at_start'] for x in locals_),
            'keeper_interrupted_after_valid_start': sum(x['keeper_interrupted_after_valid_start'] for x in locals_)}
    changed = [r for r in gated_faults if 'selection' in r and
               r['selection']['decision']['candidate'] != r['selection']['decision']['ungated_candidate']]
    report = {'scope': protocol['scope'], 'test_seeds': protocol['test_seeds'],
        'threshold': protocol['threshold'], 'test_config': protocol['test_config'], 'validity': validity,
        'fault_results': {m: method_metrics([r for r in faults if r['method'] == m]) for m in protocol['methods']},
        'healthy_controls': {m: method_metrics([r for r in rows if r['scenario'] == 'sham' and r['method'] == m])
            for m in ('protected', 'gated_direct')},
        'by_scenario': {q: {m: method_metrics([r for r in faults if r['method'] == m and r['scenario'] == q])
            for m in protocol['methods']} for q in SCENARIOS},
        'paired': {f'{a}_vs_{b}': paired_statistics(faults, a, b) for a, b in
            [('protected', 'gated_direct'), ('protected', 'global'), ('gated_direct', 'global')]},
        'local_start_counts': counts,
        'changed_decisions': {'faults': len(changed),
            'from_to': dict(Counter(r['selection']['decision']['ungated_candidate'] + '->' + r['candidate'] for r in changed)),
            'note': 'Ungated proposals are observed decisions only; their unexecuted outcomes are unknown.'},
        'failures': failure_cases}
    save_json(output / 'report.json', report)
    fields = ['seed', 'scenario', 'method', 'candidate', 'success', 'protection_violation',
        'total_post_fault_seconds', 'skill_seconds', 'inference_wait_seconds', 'selector_wall_seconds',
        'recovery_path_m', 'wall_seconds', 'failure_reason']
    with (output / 'cases.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in fields} for r in rows)
    fields = ['split', 'method', 'episodes', 'safe_successes', 'terminal_successes', 'protection_violations',
              'mean_total_seconds', 'mean_skill_seconds', 'mean_wait_seconds', 'mean_recovery_path_m']
    with (output / 'main-results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for split, results in [('fault', report['fault_results']), ('healthy', report['healthy_controls'])]:
            for method, result in results.items():
                writer.writerow({'split': split, 'method': method, **{k: result[k] for k in fields[2:]}})
    figures(output, report)
    return report


def figures(output, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods = ['protected', 'gated_direct', 'global']
    labels = ['Protected rule', 'Frozen MLP + start gate', 'Fixed global']
    colors = ['#3973ac', '#d07825', '#8b9299']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
        result = report['fault_results'][method]
        percent = 100 * result['safe_successes'] / result['episodes']
        axes[0].barh(label, percent, color=color)
        axes[0].text(percent + 1, i, f"{result['safe_successes']}/{result['episodes']}", va='center')
    axes[0].set_xlim(0, 116)
    axes[0].set_xlabel('Safe completion (%)')
    pair = report['paired']['protected_vs_gated_direct']
    for i, (key, color) in enumerate([('first_mean', colors[0]), ('second_mean', colors[1])]):
        result = pair['total_post_fault_seconds']
        if result is not None:
            value = result[key]
            axes[1].barh(labels[i], value, color=color)
            axes[1].text(value + .1, i, f'{value:.3f}', va='center')
    axes[1].set_xlabel(f"Total post-fault seconds\nSame {pair['both_safe_scenes']} common-safe states")
    axes[1].set_xlim(0, max(report['fault_results'][m]['mean_total_seconds'] or 0 for m in methods) * 1.1)
    for axis in axes:
        axis.invert_yaxis()
        axis.grid(axis='x', alpha=.2)
    fig.suptitle('Frozen weights and threshold; new initialization seeds')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'comparison.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--phase', required=True, choices=['prepare', 'run', 'analyze'])
    parser.add_argument('--parent', default='outputs/p0-20261109')
    parser.add_argument('--seed-start', type=int, default=11000)
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--seed-audit')
    parser.add_argument('--runtime-check')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.phase == 'prepare':
        if not args.seed_audit:
            parser.error('prepare requires a fresh-seed audit')
        protocol = prepare(args.output_dir, args.parent, range(args.seed_start, args.seed_start + args.seeds),
                           json.loads(Path(args.seed_audit).read_text()),
                           json.loads(Path(args.runtime_check).read_text()) if args.runtime_check else None)
        print(json.dumps({'prepared': True, 'threshold': protocol['threshold'], 'episodes': protocol['expected_episodes']}))
    elif args.phase == 'run':
        run(args.output_dir, args.workers)
    else:
        report = analyze(args.output_dir)
        print(json.dumps({'validity': report['validity'], 'fault_results': report['fault_results']}))


if __name__ == '__main__':
    main()
