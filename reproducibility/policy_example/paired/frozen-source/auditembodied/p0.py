"""P0: observable candidate learning, fresh closed loops and paired waiting."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import csv
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from .duration_pilot import STATE_KEYS, file_hash, prefix_frames
from .l4_lift import CANDIDATES, SCENARIOS, LiftConfig, run_trial
from .l4_offline_selection import features, load_cases, predict, summarize
from .l4_selector import ProprioRule, safe


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


class CandidateMLP:
    """Masked multi-label safety prediction; every observed safe action is positive."""

    def __init__(self, values):
        self.values = values
        self.rule = ProprioRule(**values['rule'])
        self.mean = np.array(values['mean'])
        self.scale = np.array(values['scale'])
        self.w1, self.b1, self.w2, self.b2 = [np.array(values[k]) for k in ('w1', 'b1', 'w2', 'b2')]

    def probabilities(self, x):
        z = (np.atleast_2d(x) - self.mean) / self.scale
        logits = np.tanh(z @ self.w1 + self.b1) @ self.w2 + self.b2
        return 1 / (1 + np.exp(-np.clip(logits, -35, 35)))

    def choose_features(self, x, threshold=None):
        threshold = self.values['threshold'] if threshold is None else threshold
        p = self.probabilities(x)
        order = np.argsort(self.values['action_cost_seconds'], kind='stable')
        return [next((CANDIDATES[j] for j in order if row[j] >= threshold), 'abstain') for row in p]

    def __call__(self, port):
        x = features(port, self.rule)
        return {'candidate': self.choose_features(x)[0],
                'safety_scores': self.probabilities(x)[0].tolist(),
                'threshold': self.values['threshold']}


def training_arrays(cases):
    x = np.stack([c['x'] for c in cases])
    y = np.zeros((len(cases), len(CANDIDATES)))
    mask = np.zeros_like(y)
    costs = [[] for _ in CANDIDATES]
    for i, case in enumerate(cases):
        for j, action in enumerate(CANDIDATES):
            if action in case['outcomes']:
                row = case['outcomes'][action]
                y[i, j], mask[i, j] = safe(row), 1
                costs[j].append(row['recovery_seconds'])
    return x, y, mask, [float(np.mean(v)) for v in costs]


def fit_mlp(cases, rule, steps=4000, seed=20261109):
    x, y, mask, costs = training_arrays(cases)
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1
    z = (x - mean) / scale
    rng = np.random.default_rng(seed)
    params = [rng.normal(0, 1 / np.sqrt(x.shape[1]), (x.shape[1], 32)), np.zeros(32),
              rng.normal(0, 1 / np.sqrt(32), (32, len(CANDIDATES))), np.zeros(len(CANDIDATES))]
    weights = mask.copy()
    for j in range(len(CANDIDATES)):
        positives = int(np.sum(mask[:, j] * y[:, j]))
        negatives = int(np.sum(mask[:, j] * (1 - y[:, j])))
        if positives and negatives:
            weights[:, j] *= y[:, j] / (2 * positives) + (1 - y[:, j]) / (2 * negatives)
        else:
            weights[:, j] /= max(positives + negatives, 1)
    weights /= weights.sum()
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    for step in range(1, steps + 1):
        w1, b1, w2, b2 = params
        hidden = np.tanh(z @ w1 + b1)
        logits = hidden @ w2 + b2
        p = 1 / (1 + np.exp(-np.clip(logits, -35, 35)))
        error = (p - y) * weights
        dh = (error @ w2.T) * (1 - hidden ** 2)
        gradients = [z.T @ dh + 1e-4 * w1, dh.sum(0),
                     hidden.T @ error + 1e-4 * w2, error.sum(0)]
        for j, gradient in enumerate(gradients):
            first[j] = .9 * first[j] + .1 * gradient
            second[j] = .999 * second[j] + .001 * gradient ** 2
            params[j] -= .01 * (first[j] / (1 - .9 ** step)) / (np.sqrt(second[j] / (1 - .999 ** step)) + 1e-8)
    values = {'rule': asdict(rule), 'mean': mean.tolist(), 'scale': scale.tolist(),
              **{k: p.tolist() for k, p in zip(('w1', 'b1', 'w2', 'b2'), params)},
              'threshold': .5, 'action_cost_seconds': costs, 'steps': steps, 'seed': seed,
              'architecture': '7 observable features -> 32 tanh -> 4 independent safety logits',
              'supervision': 'all actually executed candidate outcomes; unknown healthy alternatives masked'}
    return CandidateMLP(values)


def choice_summary(cases, choices):
    faults = [i for i, c in enumerate(cases) if not c['healthy']]
    healthy = [i for i, c in enumerate(cases) if c['healthy']]
    return {'faults': summarize([cases[i] for i in faults], [choices[i] for i in faults]),
            'healthy_scenes': len(healthy),
            'healthy_continue': sum(choices[i] == 'continue' for i in healthy),
            'choices': choices}


def choose_working_point(model, development):
    x = np.stack([c['x'] for c in development])
    scores = np.unique(model.probabilities(x))
    thresholds = np.unique(np.r_[0., (scores[:-1] + scores[1:]) / 2, 1.])
    records = []
    for threshold in thresholds:
        result = choice_summary(development, model.choose_features(x, float(threshold)))
        result['threshold'] = float(threshold)
        records.append(result)
    eligible = [r for r in records if r['healthy_continue'] == r['healthy_scenes'] and
                r['faults']['unobserved_outcomes'] == 0]
    if not eligible:
        raise ValueError('no development working point has observed fault outcomes and healthy continue')
    key = lambda r: (-r['faults']['safe_successes'], r['faults']['protection_violations'],
                     r['faults']['mean_recovery_seconds'], abs(r['threshold'] - .5))
    selected = min(eligible, key=key)
    model.values['threshold'] = selected['threshold']
    return selected, records


def prepare(output, seeds, seed_audit):
    output = Path(output)
    if output.exists():
        raise ValueError('P0 output already exists')
    if seed_audit['used_overlap'] or list(seeds) != seed_audit['requested_seeds']:
        raise ValueError('seed audit does not establish a fresh requested cohort')
    rule = ProprioRule(**json.loads(Path('outputs/l4-proprio-20261107/frozen.json').read_text())['rule'])
    training = load_cases('train', rule)
    development = load_cases('dev', rule)
    started = time.perf_counter()
    model = fit_mlp(training, rule)
    selected, grid = choose_working_point(model, development)
    tree = json.loads(Path('outputs/l4-offline-attribution-20261108/direct-tree.json').read_text())['tree']
    tree_dev = choice_summary(development, predict(tree, np.stack([c['x'] for c in development])))
    candidates = [('candidate_mlp', selected), ('direct_tree', tree_dev)]
    name, result = min(candidates, key=lambda v: (-v[1]['faults']['safe_successes'],
        v[1]['faults']['protection_violations'], v[1]['faults']['mean_recovery_seconds'], v[0]))
    direct = {'kind': name, 'mlp': model.values, 'tree': tree, 'rule': asdict(rule)}
    output.mkdir(parents=True)
    save_json(output / 'direct-selected.json', direct)
    save_json(output / 'training-report.json', {
        'train_seeds': list(range(10)), 'development_seeds': list(range(60, 68)),
        'training': choice_summary(training, model.choose_features(np.stack([c['x'] for c in training]))),
        'candidate_mlp_development': selected, 'cart_development': tree_dev,
        'selected_direct': name, 'selection': 'development safe success, protection, recovery cost; never new test outcomes',
        'threshold_scan': grid, 'fitting_seconds': time.perf_counter() - started})
    base = LiftConfig()
    held = replace(base, fault_withdrawal=.06, coupled_height_offset=.05)
    sources = ['l4_lift.py', 'l0.py', 'continuous.py', 'recovery_skills.py', 'l4_selector.py', 'l4_offline_selection.py', 'p0.py']
    protocol = {'scope': 'P0 observable selector comparison and held-out perturbation/wait intervention',
        'rule': asdict(rule), 'training_config': asdict(base), 'test_config': asdict(held),
        'test_seeds': list(seeds), 'wait_subset_seeds': list(seeds)[:8],
        'methods': ['protected', 'direct', 'global'], 'scenarios': list(SCENARIOS),
        'shift': {'fault_withdrawal': [.04, .06], 'coupled_height_offset': [.04, .05]},
        'shift_selection': 'fixed before new-state execution; no test-outcome tuning',
        'selected_direct': name, 'core_episodes': len(seeds) * 3 * 3 + len(seeds) * 2,
        'intervention_episodes': min(8, len(seeds)) * 2 * 3 * 2,
        'intervention_wait_steps': [0, 19], 'intervention_durations': [30, 40, 60],
        'wait_interpretation': '0.95s registered scope-decision budget as controlled intervention; supplied repair role, not measured model latency',
        'cost': 'core: actual measured selector time rounded up to 20Hz physical waits, feedback motion and final 1s hold; wall clock separate',
        'fixed_global_information': 'constant candidate, no privileged repair role',
        'seed_audit': seed_audit,
        'physics_source_sha256': {str(Path('auditembodied') / p): file_hash(Path('auditembodied') / p) for p in sources},
        'direct_model_sha256': file_hash(output / 'direct-selected.json')}
    save_json(output / 'protocol.json', protocol)
    return protocol


def direct_selector(direct):
    if direct['kind'] == 'candidate_mlp':
        return CandidateMLP(direct['mlp'])
    rule = ProprioRule(**direct['rule'])
    return lambda port: {'candidate': predict(direct['tree'], [features(port, rule)])[0]}


def trial_job(job):
    spec, config, rule_values, direct = job
    seed, scenario, method, wait_steps, duration = spec
    started = time.perf_counter()
    config = LiftConfig(**config)
    if method == 'intervention':
        candidate = 'R_left' if scenario == 'left_loss' else 'R_right'
        selector = lambda port: {'candidate': candidate, 'local_relevel_steps': duration}
        row = run_trial(seed, scenario, config=config, selector=selector, fixed_selector_wait_steps=wait_steps)
    else:
        selector = ProprioRule(**rule_values) if method == 'protected' else direct_selector(direct) if method == 'direct' else lambda port: {'candidate': 'R_global'}
        row = run_trial(seed, scenario, config=config, selector=selector, account_selector_latency=True)
    row.update(method=method, planned_wait_steps=wait_steps, planned_duration_steps=duration,
               wall_seconds=time.perf_counter() - started)
    return row


def run_phase(output, phase, workers=4):
    output = Path(output)
    protocol = json.loads((output / 'protocol.json').read_text())
    if any(file_hash(p) != h for p, h in protocol['physics_source_sha256'].items()):
        raise ValueError('physical sources changed after freeze')
    if file_hash(output / 'direct-selected.json') != protocol['direct_model_sha256']:
        raise ValueError('direct model changed after freeze')
    direct = json.loads((output / 'direct-selected.json').read_text())
    if phase == 'core':
        specs = [(s, q, m, None, None) for s in protocol['test_seeds']
                 for q in (*SCENARIOS, 'sham') for m in (protocol['methods'] if q != 'sham' else ['protected', 'direct'])]
    else:
        specs = [(s, q, 'intervention', w, n) for s in protocol['wait_subset_seeds']
                 for q in ('left_loss', 'right_loss') for w in protocol['intervention_wait_steps']
                 for n in protocol['intervention_durations']]
    destination = output / f'{phase}-trials.jsonl'
    jobs = [(spec, protocol['test_config'], protocol['rule'], direct) for spec in specs]
    with destination.open('x') as stream:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for i, row in enumerate(pool.map(trial_job, jobs), 1):
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                print(json.dumps({'phase': phase, 'completed': i, 'total': len(specs),
                    **{k: row.get(k) for k in ('seed', 'scenario', 'method', 'candidate', 'success', 'protection_violation', 'failure_reason')}}), flush=True)
    if any(file_hash(p) != h for p, h in protocol['physics_source_sha256'].items()):
        raise ValueError('physical sources changed during execution')


def method_metrics(rows):
    def mean(key):
        return float(np.mean([r[key] for r in rows])) if rows and all(key in r for r in rows) else None
    return {'episodes': len(rows), 'safe_successes': sum(safe(r) for r in rows),
        'terminal_successes': sum(r.get('task_complete', False) for r in rows),
        'protection_violations': sum(r['protection_violation'] for r in rows),
        'mean_total_seconds': mean('total_post_fault_seconds'), 'mean_skill_seconds': mean('skill_seconds'),
        'mean_wait_seconds': mean('inference_wait_seconds'), 'mean_selector_wall_seconds': mean('selector_wall_seconds'),
        'mean_recovery_path_m': mean('recovery_path_m'), 'mean_episode_wall_seconds': mean('wall_seconds'),
        'complete_cost_observations': sum('total_post_fault_seconds' in r for r in rows),
        'choices': dict(Counter(r['candidate'] for r in rows)),
        'failure_reasons': dict(Counter(r.get('failure_reason') or 'protection_condition_failed' for r in rows if not safe(r)))}


def paired_statistics(rows, first, second, draws=2000):
    groups = {}
    for row in rows:
        groups.setdefault((row['seed'], row['scenario']), {})[row['method']] = row
    groups = {key: value for key, value in groups.items() if first in value and second in value}
    seeds = sorted({s for s, _ in groups})
    rng = np.random.default_rng(20261109)
    indices = rng.integers(0, len(seeds), (draws, len(seeds)))
    differences = np.array([np.mean([int(safe(g[first])) - int(safe(g[second]))
        for (s, _), g in groups.items() if s == seed]) for seed in seeds])
    result = {'first': first, 'second': second, 'paired_scenes': len(groups),
        'safe_success_difference': float(differences.mean()),
        'safe_success_difference_ci95': np.quantile(differences[indices].mean(1), [.025, .975]).tolist()}
    paired = [(key, g[first], g[second]) for key, g in groups.items() if safe(g[first]) and safe(g[second])]
    result['both_safe_scenes'] = len(paired)
    for metric in ('total_post_fault_seconds', 'skill_seconds', 'recovery_path_m'):
        if not paired:
            result[metric] = None
            continue
        a, b = [float(np.mean([pair[j][metric] for pair in paired])) for j in (1, 2)]
        totals = np.array([[sum(pair[1][metric] for pair in paired if pair[0][0] == seed),
                            sum(pair[2][metric] for pair in paired if pair[0][0] == seed)] for seed in seeds])
        sampled = totals[indices].sum(1)
        ratios = 1 - sampled[sampled[:, 1] > 0, 0] / sampled[sampled[:, 1] > 0, 1]
        result[metric] = {'first_mean': a, 'second_mean': b,
                         'reduction_fraction': 1 - a / b if b else None,
                         'ci95': np.quantile(ratios, [.025, .975]).tolist() if len(ratios) else None}
    return result


def same_prefix(rows):
    reference = rows[0]
    return all(all(row.get(k) == reference.get(k) for k in STATE_KEYS) and
               prefix_frames(row) == prefix_frames(reference) for row in rows)


def outcome(row):
    return ('B' if row['protection_violation'] else 'A') if row['success'] else ('D' if row['protection_violation'] else 'C')


def analyze(output):
    output = Path(output)
    protocol = json.loads((output / 'protocol.json').read_text())
    read = lambda phase: [json.loads(s) for s in (output / f'{phase}-trials.jsonl').read_text().splitlines()]
    core, wait = read('core'), read('intervention')
    faults = [r for r in core if r['scenario'] != 'sham']
    groups, wait_groups = {}, {}
    for r in core:
        groups.setdefault((r['seed'], r['scenario']), []).append(r)
    for r in wait:
        wait_groups.setdefault((r['seed'], r['scenario']), []).append(r)
    all_rows = core + wait
    valid = {'physical_episodes': len(all_rows), 'core_episodes': len(core), 'intervention_episodes': len(wait),
        'expected_counts': len(core) == protocol['core_episodes'] and len(wait) == protocol['intervention_episodes'],
        'all_paired_prefixes_equal': all(same_prefix(g) for g in [*groups.values(), *wait_groups.values()]),
        'initialization_failures': sum(not r.get('nominal_ready', False) for r in all_rows),
        'invalid_faults': sum(not r.get('fault_class_valid', False) for r in all_rows),
        'execution_errors': sum(bool(r.get('error')) for r in all_rows),
        'numerical_instabilities': sum(r.get('failure_reason') == 'numerical_instability' for r in all_rows),
        'contact_anchors': sum(r['contact_anchor_count'] for r in all_rows),
        'state_edits': sum(r['state_edit_count_after_initialization'] for r in all_rows),
        'forbidden_calls': sum(len(r['blocked_calls']) for r in all_rows),
        'wait_cost_identity': all('total_post_fault_seconds' in r and abs(r['total_post_fault_seconds'] -
            r['inference_wait_seconds'] - r['skill_seconds'] - r['hold_seconds']) < 1e-8 for r in all_rows),
        'wait_step_accounting': all(len([f for f in r['trace'] if f['stage'] == 'inference_wait']) == r['inference_wait_steps'] for r in all_rows),
        'model_unchanged': file_hash(output / 'direct-selected.json') == protocol['direct_model_sha256'],
        'physics_unchanged': all(file_hash(p) == h for p, h in protocol['physics_source_sha256'].items())}
    transitions, envelopes, wait_results = Counter(), {}, {}
    for w in (0, 19):
        selected = [r for r in wait if r['planned_wait_steps'] == w]
        wait_results[str(w)] = {str(n): method_metrics([r for r in selected if r['planned_duration_steps'] == n]) for n in (30, 40, 60)}
        envelopes[str(w)] = {'safe_candidates': sum(safe(r) for r in selected),
            'recoverable_states': sum(any(safe(r) for r in group if r['planned_wait_steps'] == w) for group in wait_groups.values())}
    contrasts = []
    for (seed, scenario), group in wait_groups.items():
        lookup = {(r['planned_wait_steps'], r['planned_duration_steps']): r for r in group}
        for n in (30, 40, 60):
            transitions[outcome(lookup[0, n]) + '->' + outcome(lookup[19, n])] += 1
        contrasts.append({'seed': seed, 'scenario': scenario,
            'without_wait': {str(n): outcome(lookup[0, n]) for n in (30, 40, 60)},
            'with_wait': {str(n): outcome(lookup[19, n]) for n in (30, 40, 60)}})
    report = {'scope': protocol['scope'], 'selected_direct': protocol['selected_direct'], 'test_config': protocol['test_config'],
        'test_seeds': protocol['test_seeds'], 'validity': valid,
        'fault_results': {m: method_metrics([r for r in faults if r['method'] == m]) for m in protocol['methods']},
        'by_scenario': {q: {m: method_metrics([r for r in faults if r['method'] == m and r['scenario'] == q])
            for m in protocol['methods']} for q in SCENARIOS},
        'healthy_controls': {m: method_metrics([r for r in core if r['method'] == m and r['scenario'] == 'sham']) for m in ('protected', 'direct')},
        'paired': {f'{a}_vs_{b}': paired_statistics(faults, a, b) for a, b in [('protected', 'direct'), ('protected', 'global'), ('direct', 'global')]},
        'wait_intervention': {'states': len(wait_groups), 'results': wait_results, 'envelopes': envelopes,
            'transitions': dict(transitions), 'cases': contrasts,
            'interpretation': protocol['wait_interpretation']},
        'failures': [{k: r.get(k) for k in ('seed', 'scenario', 'method', 'candidate', 'planned_wait_steps',
            'planned_duration_steps', 'success', 'protection_violation', 'failure_reason', 'protection', 'error')} for r in all_rows if not safe(r)]}
    save_json(output / 'report.json', report)
    fields = ['seed', 'scenario', 'method', 'candidate', 'planned_wait_steps', 'planned_duration_steps',
        'success', 'protection_violation', 'total_post_fault_seconds', 'skill_seconds', 'inference_wait_seconds',
        'selector_wall_seconds', 'recovery_path_m', 'wall_seconds', 'failure_reason']
    with (output / 'cases.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in fields} for r in all_rows)
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
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    methods = ['protected', 'direct', 'global']
    labels = ['Protected rule', 'Direct candidate MLP' if report['selected_direct'] == 'candidate_mlp' else 'Direct CART', 'Fixed global']
    colors = ['#3973ac', '#d07825', '#8b9299']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    values = [100 * report['fault_results'][m]['safe_successes'] / report['fault_results'][m]['episodes'] for m in methods]
    axes[0].barh(labels, values, color=colors)
    axes[0].set_xlim(0, 115)
    for i, m in enumerate(methods):
        r = report['fault_results'][m]
        axes[0].text(values[i] + 1, i, f"{r['safe_successes']}/{r['episodes']}", va='center')
    axes[0].set_xlabel('Safe completion (%)')
    for i, m in enumerate(methods[:2]):
        result = report['paired'][f'{m}_vs_global']['total_post_fault_seconds']
        if result is None:
            continue
        value = result['reduction_fraction'] * 100
        low, high = [v * 100 for v in result['ci95']]
        axes[1].barh(labels[i], value, color=colors[i])
        axes[1].errorbar(value, i, xerr=[[value - low], [high - value]], fmt='none', ecolor='black', capsize=4)
    axes[1].axvline(0, color='black', linewidth=.7)
    axes[1].set_xlabel('Total time reduction vs global (%)\nCommon safe scenes; seed bootstrap 95% CI')
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Fresh states and held-out perturbation amplitudes')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'p0-main-results.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    cases = report['wait_intervention']['cases']
    matrix = np.array([['ABCD'.index(r[key][str(n)]) for key in ('without_wait', 'with_wait') for n in (30, 40, 60)] for r in cases])
    palette = ['#6ba88b', '#e7a65d', '#c5cbd1', '#cc6677']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(matrix, cmap=ListedColormap(palette), vmin=-.5, vmax=3.5, aspect='auto')
    for i in range(len(cases)):
        for j in range(6):
            ax.text(j, i, 'ABCD'[matrix[i, j]], ha='center', va='center')
    ax.set_yticks(range(len(cases)), [f"{r['seed']} {r['scenario'].split('_')[0]}" for r in cases])
    ax.set_xticks(range(6), ['30 / W0', '40 / W0', '60 / W0', '30 / W19', '40 / W19', '60 / W19'])
    ax.axvline(2.5, color='white', linewidth=3)
    ax.set_xlabel('Relevel steps / forced pre-skill wait at 20 Hz')
    ax.set_title('Actually executed local candidates on matched fault states')
    ax.legend(handles=[Patch(color=c, label=s) for c, s in zip(palette, ['A safe complete', 'B unsafe complete', 'C safe incomplete', 'D unsafe incomplete'])],
              loc='upper center', bbox_to_anchor=(.5, -.12), ncol=2)
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'p0-wait-intervention.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--phase', choices=['prepare', 'core', 'intervention', 'analyze'], required=True)
    parser.add_argument('--seed-start', type=int, default=10000)
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--seed-audit')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.phase == 'prepare':
        if not args.seed_audit:
            parser.error('prepare requires the completed seed metadata audit')
        audit = json.loads(Path(args.seed_audit).read_text())
        result = prepare(args.output_dir, range(args.seed_start, args.seed_start + args.seeds), audit)
        print(json.dumps({'prepared': True, 'selected_direct': result['selected_direct']}))
    elif args.phase == 'analyze':
        result = analyze(args.output_dir)
        print(json.dumps({'validity': result['validity'], 'fault_results': result['fault_results'], 'wait_envelopes': result['wait_intervention']['envelopes']}))
    else:
        run_phase(args.output_dir, args.phase, args.workers)


if __name__ == '__main__':
    main()
