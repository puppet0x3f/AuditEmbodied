"""Post-hoc paired method ranking and explicit first-closing phase timestamps.

No new physical runs or revised historical scores. Requires only NumPy.
"""
from collections import Counter, defaultdict
import numpy as np


def interval(values):
    x = np.asarray(values, float)
    assert len(x) and np.isfinite(x).all()
    draws = np.random.default_rng(20261122).integers(0, len(x), (10000, len(x)))
    return {'estimate': float(x.mean()),
            'ci95': np.quantile(x[draws].mean(1), [.025, .975]).tolist(),
            'independent_seeds': len(x)}


def ranking(rows):
    by = {(r['seed'], r['scenario'], r['method'], r['hold_mode'], r['wait_steps']): r for r in rows}
    seeds = sorted({r['seed'] for r in rows})
    scenarios = ('left_loss', 'right_loss')
    cells = []
    gaps = {}
    for hold in ('registered', 'measured_position'):
        for wait in (1, 20):
            values = []; episode_counts = Counter(); seed_counts = Counter(); seed_values = []
            for seed in seeds:
                outcomes = [[bool(by[seed, s, m, hold, wait]['native_Y']) for m in
                             ('closed_loop', 'visual_servo')] for s in scenarios]
                delta = float(np.mean([int(a)-int(b) for a, b in outcomes]))
                values.append(delta)
                seed_values.append({'seed': seed, 'CAD_minus_Landmark': delta})
                seed_counts['CAD_win' if delta > 0 else 'Landmark_win' if delta < 0 else 'tie'] += 1
                for a, b in outcomes:
                    episode_counts['both' if a and b else 'CAD_only' if a else 'Landmark_only' if b else 'neither'] += 1
            gaps[hold, wait] = values
            cells.append({'hold_mode': hold, 'wait_steps': wait, 'Y500_CAD_minus_Landmark': interval(values),
                          'episode_pairs': dict(episode_counts), 'seed_comparison': dict(seed_counts),
                          'seed_values': seed_values})
    long_change = np.asarray(gaps['measured_position', 20])-gaps['registered', 20]
    changes = []
    for i, seed in enumerate(seeds):
        first = gaps['registered', 20][i]; second = gaps['measured_position', 20][i]
        changes.append({'seed': seed, 'registered_gap': first, 'position_gap': second,
                        'gap_change': float(long_change[i])})
    return {'cells': cells, 'long_wait_gap_change': interval(long_change), 'long_wait_seed_transitions': changes,
            'note': 'Registered-wrist slice. Gap change differs from the original wrist-averaged three-way contrast. '
                    'All fault sides and conditions remain together in a resampled seed block; new analysis is post hoc.'}


def closure(row):
    trace = row['trace']; log = row['global_execution']
    end = next((a['close_time'] for a in log['pickup_attempts'] if a.get('close_time') is not None), None)
    result = {k: row[k] for k in ('seed', 'scenario', 'method', 'hold_mode', 'wait_steps')}
    if end is None:
        return dict(result, available=False)
    last = next(i for i, f in enumerate(trace) if abs(f['time']-end) < 1e-8)
    first = last
    while first > 0 and trace[first-1]['stage'] == 'global_close':
        first -= 1
    assert trace[first]['stage'] == trace[last]['stage'] == 'global_close'
    before = first-1
    obs = [d for d in log['observations'] if d['estimate']['valid'] and
           d['observed_time'] <= trace[before]['time']+1e-8][-1]
    target = np.asarray(obs['estimate']['targets'])
    h0 = np.asarray(obs.get('observation_truth', {}).get('handle_positions',
                    next(f['evaluator']['handle_positions'] for f in trace
                         if abs(f['time']-obs['observed_time']) < 1e-8)))
    action = np.asarray(trace[first]['robot']['action']).reshape(2, 7)
    expected = np.clip(.8*(target-np.asarray(trace[before]['robot']['positions']))/.05, -1., 1.)
    assert np.allclose(expected, action[:, :3], atol=1e-8, rtol=0)
    keeper = 1 if row['scenario'] == 'left_loss' else 0 if row['scenario'] == 'right_loss' else None
    moments = {}
    for name, index in (('before_first_closing_command', before), ('after_first_closing_step', first), ('end_of_first_closing_phase', last)):
        f = trace[index]; pos = np.asarray(f['robot']['positions']); handle = np.asarray(f['evaluator']['handle_positions'])
        components = [target-h0, -(handle-h0), pos-target]
        residual = sum(components)-(pos-handle)
        assert np.max(abs(residual)) < 1e-12
        arms = []
        for arm in range(2):
            desired = np.asarray(obs['estimate']['rotation_targets'])[arm]
            current = np.asarray(f['robot']['rotations'])[arm]
            wrist = np.degrees(np.arccos(np.clip((np.trace(desired@current.T)-1)/2, -1, 1)))
            arms.append({'role': 'healthy' if keeper is None else 'keeper' if keeper == arm else 'free',
                         'target_mm': float(np.linalg.norm(components[0][arm])*1000),
                         'drift_mm': float(np.linalg.norm(components[1][arm])*1000),
                         'tracking_mm': float(np.linalg.norm(components[2][arm])*1000),
                         'eef_handle_mm': float(np.linalg.norm(pos[arm]-handle[arm])*1000),
                         'wrist_deg': float(wrist), 'registered_grasp': bool(f['evaluator']['holds'][arm])})
        moments[name] = {'time': f['time'], 'stage': f['stage'], 'arms': arms}
    return dict(result, available=True, moments=moments, closing_samples=last-first+1,
                closing_phase_seconds=float(trace[last]['time']-trace[before]['time']),
                first_sample_to_stored_close_seconds=float(trace[last]['time']-trace[first]['time']),
                first_command_target_verified=True, joint_positions_recorded='qpos' in trace[before]['robot'])


def summarize(rows, timed):
    groups = defaultdict(list)
    for r in timed:
        groups[r['method'], r['hold_mode'], r['wait_steps'], r['scenario'] == 'sham'].append(r)
    conditions = []
    for (method, hold, wait, healthy), group in sorted(groups.items()):
        d = {'method': method, 'hold_mode': hold, 'wait_steps': wait, 'healthy': healthy,
             'episodes': len(group), 'available': sum(r['available'] for r in group), 'moments': {}}
        for moment in ('before_first_closing_command', 'after_first_closing_step', 'end_of_first_closing_phase'):
            roles = {}
            for role in ('keeper', 'free', 'healthy'):
                by_seed = defaultdict(list)
                for r in group:
                    if r['available']:
                        arms=[a for a in r['moments'][moment]['arms'] if a['role'] == role]
                        if arms:by_seed[r['seed']].extend(arms)
                if not by_seed: continue
                roles[role] = {key: interval([np.mean([a[key] for a in arms]) for s, arms in sorted(by_seed.items())])
                               for key in ('target_mm', 'drift_mm', 'tracking_mm', 'eef_handle_mm', 'wrist_deg')}
            d['moments'][moment] = roles
        conditions.append(d)
    return {'records': len(rows), 'new_physical_executions': 0, 'ranking': ranking(rows),
            'closure': {'records': len(timed), 'action_target_checks': sum(r.get('first_command_target_verified', False) for r in timed),
                        'closing_samples': sorted({r['closing_samples'] for r in timed if r['available']}),
                        'joint_positions_available': sum(r.get('joint_positions_recorded', False) for r in timed),
                        'conditions': conditions},
            'zero_discordance_note': {'independent_seeds': 20, 'confidence': .95,
                'one_sided_upper_bound_on_seed_discordance_probability': 1-.05**(1/20),
                'meaning': 'For a comparison with zero discordant seeds only; not an interaction-effect interval or an equivalence test.'}}


def tables(report):
    lines = [r'Hold & Wait (s) & CAD--Landmark (pp) & 95\% interval & Seed W/T/L\\', r'\midrule']
    for c in report['ranking']['cells']:
        d = c['Y500_CAD_minus_Landmark']; w = c['seed_comparison']; lo, hi = np.asarray(d['ci95'])*100
        lines.append(f"{'Registered' if c['hold_mode']=='registered' else 'Position only'} & {c['wait_steps']*.05:.2f} & {d['estimate']*100:+.1f} & [{lo:+.1f}, {hi:+.1f}] & {w.get('CAD_win',0)}/{w.get('tie',0)}/{w.get('Landmark_win',0)}"+r'\\')
    rank = '\n'.join([r'\begin{tabular}{llrrr}', r'\toprule', *lines, r'\bottomrule', r'\end{tabular}'])+'\n'
    lines = [r'Executor & Hold & Role & Before close & First step & End close\\', r'\midrule']
    for c in report['closure']['conditions']:
        if c['healthy'] or c['wait_steps'] != 20 or c['hold_mode'] not in ('registered','measured_position'): continue
        for role in ('keeper', 'free'):
            vals = [c['moments'][m][role]['tracking_mm']['estimate'] for m in ('before_first_closing_command','after_first_closing_step','end_of_first_closing_phase')]
            lines.append(f"{'CAD' if c['method']=='closed_loop' else 'Landmark'} & {'Registered' if c['hold_mode']=='registered' else 'Position only'} & {role.capitalize()} & "+' & '.join(f'{v:.2f}' for v in vals)+r'\\')
    timing = '\n'.join([r'\begin{tabular}{lllrrr}',r'\toprule',*lines,r'\bottomrule',r'\end{tabular}'])+'\n'
    return {'paired-ranking':rank, 'closure-timing':timing}
