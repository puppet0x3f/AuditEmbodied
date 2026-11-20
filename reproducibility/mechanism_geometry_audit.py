"""Post-hoc first-approach diagnostics and the paired geometry extension.

Only NumPy is required. Joint proximity and normalized action clipping are
diagnostics, not changes to the frozen completion or protection predicates.
"""
from collections import Counter, defaultdict
import numpy as np


def interval(values):
    values = np.asarray(values, dtype=float)
    assert len(values) and np.isfinite(values).all()
    draws = np.random.default_rng(20261113).integers(0, len(values), (10000, len(values)))
    return dict(estimate=float(values.mean()),
                ci95=np.quantile(values[draws].mean(1), [.025, .975]).tolist(),
                independent_seeds=len(values))


def seed_mean(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        if row.get(field) is not None:
            grouped[row['seed']].append(float(row[field]))
    return interval([np.mean(v) for _, v in sorted(grouped.items())]) if grouped else None


def mechanism_input(row):
    """Retain recorded inputs sufficient to recompute every new diagnostic."""
    trace = row['trace']
    first = next(i for i, f in enumerate(trace) if f['stage'] == 'global_close')
    last = first
    while last + 1 < len(trace) and trace[last + 1]['stage'] == 'global_close':
        last += 1
    frames = [f for f in trace[:last + 1] if f['phase'] == 'recovery']
    before = trace[first - 1]
    observation = [d for d in row['global_execution']['observations']
                   if d['estimate']['valid'] and d['observed_time'] <= before['time'] + 1e-8][-1]
    times = {round(f['time'], 8) for f in frames}
    return {**{k: row[k] for k in ('seed', 'scenario', 'hold_mode', 'wait_steps', 'task_complete')},
            'first_arrival': row['global_execution']['arrival_checks'][0],
            'target': observation['estimate']['targets'],
            'rotation_target': observation['estimate']['rotation_targets'],
            'frames': frames,
            'joints': [d for d in row['robot_joint_diagnostics'] if round(d['time'], 8) in times]}


def joint_state(record, arm):
    data = record['arms'][arm]
    positions = np.asarray(data['qpos'])
    limits = np.asarray(data['limits'])
    margins = np.minimum(positions - limits[:, 0], limits[:, 1] - positions)
    assert abs(float(margins.min()) - data['nearest_limit_radians']) < 1e-10
    index = int(margins.argmin())
    return dict(nearest_limit_rad=float(margins[index]), nearest_joint=index + 1,
                nearest_side='lower' if positions[index] - limits[index, 0]
                < limits[index, 1] - positions[index] else 'upper',
                joint7_limit_rad=float(margins[6]), positions=positions.tolist())


def mechanism_case(row):
    frames = row['frames']
    joint_by_time = {round(d['time'], 8): d for d in row['joints']}
    first = next(i for i, f in enumerate(frames) if f['stage'] == 'global_close')
    before = frames[first - 1]
    target = np.asarray(row['target'])
    actual_action = np.asarray(frames[first]['robot']['action']).reshape(2, 7)
    expected = np.clip(.8 * (target - np.asarray(before['robot']['positions'])) / .05, -1, 1)
    assert np.allclose(actual_action[:, :3], expected, rtol=0, atol=1e-8)
    check = row['first_arrival']
    errors = np.linalg.norm(target - np.asarray(before['robot']['positions']), axis=1) * 1000
    rotations = np.asarray(before['robot']['rotations'])
    angles = np.degrees(np.arccos(np.clip(
        (np.einsum('aij,aij->a', np.asarray(row['rotation_target']), rotations) - 1) / 2, -1, 1)))
    assert np.allclose(errors, check['translation_error_mm'], atol=1e-7, rtol=0)
    assert np.allclose(angles, check['rotation_error_deg'], atol=1e-7, rtol=0)
    assert check['arrived'] == bool(max(errors) <= 12 and max(angles) <= 5)
    assert check['bounded_best_effort_close'] == (check['extra_steps'] == 40)
    free = 0 if row['scenario'] == 'left_loss' else 1 if row['scenario'] == 'right_loss' else None
    result = {k: row[k] for k in ('seed', 'scenario', 'hold_mode', 'wait_steps')}
    result.update(C=bool(row['task_complete']), arrived=bool(check['arrived']),
                  capped=bool(check['bounded_best_effort_close']), extra_seconds=check['extra_steps'] * .05,
                  max_position_error_mm=float(errors.max()), max_wrist_error_deg=float(angles.max()),
                  position_arrived=bool(max(errors) <= 12), wrist_arrived=bool(max(angles) <= 5),
                  first_close_dual=bool(all(frames[-1]['evaluator']['holds'])),
                  first_command_target_verified=True, stages={})
    if free is None:
        return result
    result.update(free_position_error_mm=float(errors[free]), free_wrist_error_deg=float(angles[free]))
    joint = joint_state(joint_by_time[round(before['time'], 8)], free)
    result.update(free_nearest_limit_rad=joint['nearest_limit_rad'],
                  free_nearest_joint=joint['nearest_joint'], free_nearest_side=joint['nearest_side'],
                  free_joint7_limit_rad=joint['joint7_limit_rad'],
                  free_near_limit=bool(joint['nearest_limit_rad'] <= .02),
                  free_joint7_near_limit=bool(joint['joint7_limit_rad'] <= .02))
    start = next(i for i, f in enumerate(frames) if f['stage'] == 'global_align_wrist')
    approach = frames[start:first]
    actions = np.asarray([f['robot']['action'] for f in approach]).reshape(-1, 2, 7)[:, free]
    clipped = np.abs(actions) >= 1 - 1e-9
    result.update(approach_control_steps=len(approach),
                  translation_clipped_steps=int(clipped[:, :3].any(1).sum()),
                  rotation_clipped_steps=int(clipped[:, 3:6].any(1).sum()),
                  translation_clipped_fraction=float(clipped[:, :3].any(1).mean()),
                  rotation_clipped_fraction=float(clipped[:, 3:6].any(1).mean()))
    stages = ('inference_wait', 'global_table_settle', 'global_align_wrist',
              'global_hover', 'global_descend', 'global_arrival_settle', 'global_close')
    for stage in stages:
        matching = [f for f in frames if f['stage'] == stage]
        if not matching:
            continue
        frame = matching[-1]
        joints = joint_state(joint_by_time[round(frame['time'], 8)], free)
        result['stages'][stage] = dict(time=frame['time'],
            free_grasp_target_distance_mm=float(np.linalg.norm(np.asarray(frame['robot']['positions'])[free] - target[free]) * 1000),
            free_nearest_limit_rad=joints['nearest_limit_rad'], free_joint7_limit_rad=joints['joint7_limit_rad'],
            free_nearest_joint=joints['nearest_joint'], free_nearest_side=joints['nearest_side'],
            registered_free_grasp=bool(frame['evaluator']['holds'][free]))
    return result


def mechanism_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['hold_mode'], row['wait_steps'], row['scenario'] == 'sham'].append(row)
    cells = []
    fields = ('C', 'arrived', 'capped', 'extra_seconds', 'max_position_error_mm', 'max_wrist_error_deg',
              'position_arrived', 'wrist_arrived', 'first_close_dual', 'free_position_error_mm',
              'free_wrist_error_deg', 'free_nearest_limit_rad', 'free_joint7_limit_rad',
              'free_near_limit', 'free_joint7_near_limit', 'translation_clipped_fraction', 'rotation_clipped_fraction')
    counts = ('C', 'arrived', 'capped', 'position_arrived', 'wrist_arrived', 'first_close_dual',
              'free_near_limit', 'free_joint7_near_limit')
    for (hold, wait, healthy), group in sorted(groups.items()):
        cell = dict(hold_mode=hold, wait_steps=wait, healthy=healthy, episodes=len(group),
                    metrics={f: seed_mean(group, f) for f in fields},
                    counts={f: sum(bool(r.get(f)) for r in group) for f in counts}, stages={})
        cell['nearest_joint_counts'] = dict(Counter(f"J{r['free_nearest_joint']}-{r['free_nearest_side']}"
                                                   for r in group if 'free_nearest_joint' in r))
        cell['completion_by_near_limit'] = {str(near): dict(
            episodes=sum(r.get('free_near_limit') == near for r in group),
            C=sum(r['C'] for r in group if r.get('free_near_limit') == near)) for near in (False, True)} if not healthy else {}
        stages = sorted({s for r in group for s in r['stages']})
        for stage in stages:
            records = [dict(seed=r['seed'], **r['stages'][stage]) for r in group if stage in r['stages']]
            cell['stages'][stage] = dict(episodes=len(records), **{f: seed_mean(records, f) for f in
                ('free_grasp_target_distance_mm', 'free_nearest_limit_rad', 'free_joint7_limit_rad')})
        cells.append(cell)
    return dict(records=len(rows), faults=sum(r['scenario'] != 'sham' for r in rows),
                healthy=sum(r['scenario'] == 'sham' for r in rows), new_physical_executions=0,
                independent_seeds=len({r['seed'] for r in rows}),
                action_target_checks=sum(r['first_command_target_verified'] for r in rows), cells=cells,
                interpretation='Post-hoc diagnostic associations, not identified causal mediation. '
                'Near limit means <=0.02 rad from a declared joint range. Normalized action clipping '
                'does not establish actuator torque saturation. No contact-force or Jacobian evidence is added.')


def mechanism_table(report):
    lines = [r'Hold & Wait & $C$ & Arrived & Capped & Free error (mm) & Near limit & Translation clip (\%)\\', r'\midrule']
    for cell in report['cells']:
        if cell['healthy']:
            continue
        count, metrics, n = cell['counts'], cell['metrics'], cell['episodes']
        name = 'Registered' if cell['hold_mode'] == 'registered' else 'Position only'
        lines.append(f"{name} & {cell['wait_steps']*.05:.2f} & {count['C']}/{n} & {count['arrived']}/{n} & "
                     f"{count['capped']}/{n} & {metrics['free_position_error_mm']['estimate']:.1f} & "
                     f"{count['free_near_limit']}/{n} & {metrics['translation_clipped_fraction']['estimate']*100:.1f} " + r'\\')
    return '\n'.join([r'\begin{tabular}{llrrrrrr}', r'\toprule', *lines, r'\bottomrule', r'\end{tabular}']) + '\n'


def geometry_contrast(rows, terms, field):
    groups = defaultdict(dict)
    for row in rows:
        if row['scenario'] == 'sham':
            continue
        key = row['hold_mode'], row['wait_steps'], row['method']
        groups[row['seed']].setdefault(key, []).append(float(row[field]))
    values = []
    for seed, group in sorted(groups.items()):
        assert all(len(group[key]) == 2 for key, _ in terms), seed
        values.append(sum(weight * np.mean(group[key]) for key, weight in terms))
    return interval(values)


def geometry_summary(rows, waiting):
    assert len(rows) == 480 and len({r['seed'] for r in rows}) == 20
    groups = defaultdict(list)
    for row in rows:
        groups[row['hold_mode'], row['wait_steps'], row['method'], row['scenario'] == 'sham'].append(row)
    report = dict(records=480, new_formal_executions=240, reused_registered_executions=240,
                  independent_seeds=20, cells=[], contrasts={},
                  native_samples=sum(r['native_samples'] for r in rows))
    for (hold, wait, method, healthy), group in sorted(groups.items()):
        assert len(group) == (20 if healthy else 40)
        metrics = waiting.metrics(group)
        metrics['final_balance_failures'] = sum(r['final_balance_failure'] for r in group)
        report['cells'].append(dict(hold_mode=hold, wait_steps=wait, method=method, healthy=healthy,
                                   metrics=metrics))
    fields = ('C', 'Y', 'native_Y', 'total', 'wait', 'skill', 'pickup_attempts')
    def put(name, terms):
        report['contrasts'][name] = {f: geometry_contrast(rows, terms, f) for f in fields}
    effects = {}
    for method in ('retarget', 'closed_loop'):
        terms = [(('measured_position', 20, method), 1), (('registered', 20, method), -1),
                 (('measured_position', 1, method), -1), (('registered', 1, method), 1)]
        effects[method] = terms
        put(method + '_hold_wait', terms)
        for wait in (1, 20):
            put(method + f'_position_minus_registered_{wait}',
                [(('measured_position', wait, method), 1), (('registered', wait, method), -1)])
    put('executor_hold_wait', effects['closed_loop'] + [(key, -weight) for key, weight in effects['retarget']])
    for hold in ('registered', 'measured_position'):
        for wait in (1, 20):
            put(hold + f'_CAD_minus_Target_{wait}',
                [((hold, wait, 'closed_loop'), 1), ((hold, wait, 'retarget'), -1)])
    report['scope'] = ('Position-only extension of the previously observed geometry cohort. '
                       'CAD/Target, not CAD/Landmark. Shapes use different seed cohorts; '
                       'no between-shape causal contrast or equivalence claim.')
    return report


def geometry_score(row, native_path, legacy, controller, waiting, fit):
    result, fits = waiting.score(row, native_path, legacy, controller, fit)
    frames = [f for f in row['trace'] if f['stage'] == 'final_stable_hold']
    final = frames[-1]['evaluator'] if frames else row.get('final', {})
    result['final_balance_failure'] = bool(not result['C'] and all(final.get('holds', [False]))
        and final.get('clearance', 0) >= .1 and
        (final.get('tilt_deg', 0) > 15 or final.get('handle_height_difference', 0) > .03))
    return result, fits


def geometry_tables(report):
    lines = [r'Hold & Wait & Executor & $C$ & $Y_{20}$ & $Y_{500}$ & Healthy & Retry & Total (s)\\', r'\midrule']
    cells = {(c['hold_mode'], c['wait_steps'], c['method'], c['healthy']): c['metrics'] for c in report['cells']}
    for hold in ('registered', 'measured_position'):
        for wait in (1, 20):
            for method, label in (('retarget', 'Target'), ('closed_loop', 'CAD')):
                r, h = cells[hold, wait, method, False], cells[hold, wait, method, True]
                name = 'Registered' if hold == 'registered' else 'Position only'
                lines.append(f"{name} & {wait*.05:.2f} & {label} & {r['C']}/40 & {r['Y']}/40 & "
                             f"{r['native_Y']}/40 & {h['native_Y']}/20 & {r['repeated_pickups']} & {r['total']:.3f} " + r'\\')
    full = '\n'.join([r'\begin{tabular}{lllrrrrrr}', r'\toprule', *lines, r'\bottomrule', r'\end{tabular}']) + '\n'
    lines = [r'Executor & $\Delta_{h\times w}Y_{500}$ (pp) & 95\% interval & $\Delta_{h\times w}T$ (s) & 95\% interval\\', r'\midrule']
    for method, label in (('retarget', 'Target'), ('closed_loop', 'CAD')):
        d = report['contrasts'][method + '_hold_wait']
        y, t = d['native_Y'], d['total']
        lines.append(f"{label} & {y['estimate']*100:+.1f} & [{y['ci95'][0]*100:+.1f}, {y['ci95'][1]*100:+.1f}] & "
                     f"{t['estimate']:+.3f} & [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] " + r'\\')
    effects = '\n'.join([r'\begin{tabular}{lrrrr}', r'\toprule', *lines, r'\bottomrule', r'\end{tabular}']) + '\n'
    return {'geometry-holding': full, 'geometry-holding-effects': effects}


def geometry_state(row):
    return {k: row[k] for k in ('seed', 'scenario', 'method', 'hold_mode', 'wait_steps',
        'initialization_state', 'preclose_state', 'initialization_end_state', 'prefix_state', 'post_wait_state')}


def geometry_pairing(states):
    scenes, methods = defaultdict(list), defaultdict(list)
    for r in states:
        scenes[r['seed'], r['scenario']].append(r)
        methods[r['seed'], r['scenario'], r['hold_mode'], r['wait_steps']].append(r)
    assert len(scenes) == 60 and all(len(g) == 8 for g in scenes.values())
    before = ('initialization_state', 'preclose_state', 'initialization_end_state', 'prefix_state')
    for group in scenes.values():
        assert all({k: r[k] for k in before} == {k: group[0][k] for k in before} for r in group)
    assert len(methods) == 240 and all(len(g) == 2 for g in methods.values())
    for group in methods.values():
        assert {r['method'] for r in group} == {'closed_loop', 'retarget'}
        assert group[0]['post_wait_state'] == group[1]['post_wait_state']
    return dict(exact_prehold_scene_groups=60, exact_posthold_method_pairs=240,
                new_posthold_method_pairs=120, reused_registered_method_pairs=120)


def holding_commands(row):
    if row['hold_mode'] != 'measured_position':
        return 0
    reference = np.asarray(row['holding_input']['positions'])
    count = 0
    for index, frame in enumerate(row['trace']):
        if frame['phase'] == 'recovery' and frame['stage'] == 'inference_wait':
            before = np.asarray(row['trace'][index - 1]['robot']['positions'])
            actual = np.asarray(frame['robot']['action']).reshape(2, 7)
            expected = np.clip(.8 * (reference - before) / .05, -1, 1)
            assert np.allclose(actual[:, :3], expected, rtol=0, atol=1e-8)
            assert np.array_equal(actual[:, 6], row['holding_input']['gripper_commands'])
            count += 1
    assert count == row['wait_steps']
    return count
