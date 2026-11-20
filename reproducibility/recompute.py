"""Recompute the paper's main tables from logged samples, without the simulator.

Only Python and NumPy are required. Recorded scores and reference tables are
comparison targets, never inputs to scoring, candidate choice, or aggregation.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

# Frozen visual fits used single-threaded BLAS. Preserve that numerical path
# before NumPy loads; threaded reductions can change an iterative silhouette fit.
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
import numpy as np

HERE = Path(__file__).resolve().parent
DT = .05
SCOPE = ('protected', 'gated_direct', 'agentchord_style', 'global')
SCOPE_NAMES = ('Protected rule', 'MLP + start gate', 'AgentChord-style', 'Global restart')
SUPERVISION = ('fixed60', 'ce', 'terminal', 'joint')
SUPERVISION_NAMES = ('Fixed60', 'Plain SFT', 'Terminal only', 'Joint + set')
HANDOVER = ('visual', 'proprio', 'rule', 'R3')
HANDOVER_NAMES = ('Visual', 'Proprio', 'Finger rule', 'Global R3')
LOAD = ('protected', 'gated_direct', 'global')
LOAD_NAMES = ('Rule', 'Gated MLP', 'Global')
WAIT = ('decision_wait', 'controller_compute_wait')
LIFT_WAIT = ('inference_wait', 'controller_compute_wait', 'prefix_observation_wait', 'monitor_overrun_wait',
             'decision_budget_wait', 'decision_refresh_wait', 'regrasp_observation_wait')
BEFORE_TABLE = ('r3_free_other_gripper', 'r3_park_other_arm', 'r3_recenter_carrier',
                'r3_transport_above_real_table', 'r3_lower_to_table', 'r3_table_loaded',
                'r3_table_release', 'r3_table_settle')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message):
    check(abs(a - b) < 1e-7, message)


def longest(flags, keys=None):
    run = maximum = 0
    previous = None
    for i, flag in enumerate(flags):
        key = keys[i] if keys is not None else 0
        run = (run + 1 if key == previous else 1) if flag else 0
        maximum = max(maximum, run)
        previous = key
    return maximum


def read(cohort):
    with gzip.open(HERE / 'data' / f'{cohort}.jsonl.gz', 'rt') as f:
        for line in f:
            yield json.loads(line)


def frame_cost(row, phases):
    frames = [f for f in row['trace'] if f['phase'] in phases]
    previous = row['prefix_state']['time']
    for f in frames:
        close(f['time'] - previous, DT, 'Missing/duplicated physical sample')
        previous = f['time']
    check(bool(frames), 'No audited physical frames')
    return frames, frames[-1]['time'] - row['prefix_state']['time']


def lift_truth(e):
    support = [h and v <= .20 for h, v in zip(e['holds'], e['relative_speeds'])]
    ready = bool(all(support) and e['clearance'] >= .10 and e['tilt_deg'] <= 15.
                 and e['handle_height_difference'] <= .03)
    check(support == e['supports'] and ready == e['ready'], 'Lift truth predicate mismatch')
    return ready


def score_lift(row, cohort):
    frames, total = frame_cost(row, ('recovery', 'final_hold'))
    for frame in frames:
        if 'ready' in frame['evaluator']:
            lift_truth(frame['evaluator'])
    held = [f for f in frames if f['stage'] == 'final_stable_hold']
    complete = bool(len(held) >= 20 and all(lift_truth(f['evaluator']) for f in held))
    table = any(f['stage'] == 'global_table_settle' and f['evaluator']['table_contact']
                and not any(f['evaluator']['holds']) for f in frames)
    lifted = any(f['stage'] in ('global_lift', 'global_lift_settle')
                 and all(f['evaluator']['holds']) and f['evaluator']['clearance'] >= .10 for f in frames)
    cycle = table and lifted
    accepted = bool(complete and row['candidate'] != 'abstain' and not row.get('abstained', False)
                    and (row['candidate'] != 'R_global' or cycle))
    flags, roles = [], []
    for frame in frames:
        skill = frame.get('skill', row['candidate'])
        keeper = {'R_left': 1, 'R_right': 0}.get(skill) if frame['phase'] == 'recovery' else None
        roles.append(keeper)
        flags.append(keeper is not None and not frame['evaluator']['holds'][keeper])
    unsupported = longest([not (any(f['evaluator']['holds']) or f['evaluator']['table_contact']) for f in frames])
    collision = row['protection']['deep_interarm_contact_steps']
    shared_bad = unsupported > 2 or bool(collision)
    hard_bad = bool(row['blocked_calls'] or row.get('failure_reason') == 'numerical_instability')
    keeper_run = longest(flags, roles)
    bad = bool(shared_bad or any(flags) or hard_bad)
    keeper = {'R_left': 1, 'R_right': 0}.get(row['candidate'])
    v_obs = bool(row['post_fault']['holds'][keeper]) if keeper is not None else True
    v_entry = bool(row['post_wait_evaluator']['holds'][keeper]) if keeper is not None else True
    valid = v_entry if cohort in ('load', 'wait_decision', 'interface') else v_obs
    label = 'E' if not valid else 'A' if accepted and not bad else 'B' if accepted else 'D' if bad else 'C'
    waiting = sum(f['stage'] in LIFT_WAIT or cohort=='controller_pilot' and f['stage'].startswith('vision_wait_') for f in frames) * DT
    hold = sum(f['phase'] == 'final_hold' for f in frames) * DT
    skill = (len(frames) * DT) - waiting - hold
    for key, measured in [('total_post_fault_seconds', total), ('inference_wait_seconds', waiting),
                          ('skill_seconds', skill), ('hold_seconds', hold)]:
        close(row[key], measured, f'Lift cost mismatch: {key}')
    check(complete == row['task_complete'] and accepted == row['success'], 'Lift completion mismatch')
    check(table == row['table_setdown_verified'] and cycle == row['table_cycle_verified'], 'Lift interface mismatch')
    check(bad == row['protection_violation'], 'Lift protection mismatch')
    check(sum(flags) == row['protection']['keeper_grasp_lost_steps'], 'Lift keeper counter mismatch')
    close(unsupported * DT, row['protection']['max_unsupported_seconds'], 'Lift support counter mismatch')
    if 'label' in row.get('recorded', {}):
        check(label == row['recorded']['label'], 'Load label mismatch')
    result = base(row, complete, valid, bad, label, total, waiting, skill, 0., hold)
    result.update(shared_bad=bool(shared_bad), role_only=bool(any(flags) and not shared_bad and not hard_bad),
                  keeper_gap_ms=keeper_run * 50, shared_guard_completion=bool(accepted and not shared_bad and not hard_bad),
                  safe50=bool(valid and accepted and not shared_bad and not hard_bad and keeper_run <= 1),
                  safe100=bool(valid and accepted and not shared_bad and not hard_bad and keeper_run <= 2),
                  accepted=accepted, table=table, cycle=cycle)
    if cohort == 'wait_decision':
        result['initial_candidate'] = row['wait_initial_decision']['candidate']
    if cohort == 'load' and row['method'] == 'global' and row['scenario'] in ('left_loss', 'right_loss'):
        settled = [f for f in frames if f['stage'] == 'global_table_settle'][-1]
        closed = [f for f in frames if f['stage'] == 'global_close'][-1]
        handles = np.asarray(closed['evaluator']['handle_positions'])
        eef = np.asarray(closed['robot']['positions'])
        result.update(settle_tilt_deg=settled['evaluator']['tilt_deg'],
                      close_miss_m=float(np.linalg.norm(handles - eef, axis=1).max()),
                      ever_dual_regrasp=any(all(f['evaluator']['holds']) for f in frames if f['stage'] in
                          ('global_close', 'global_lift', 'global_lift_settle', 'final_stable_hold')))
    return result


def interface_case(row, result, geometry):
    frames = [f for f in row['trace'] if f['phase'] == 'recovery']
    prefix = [f for f in row['trace'] if f['time'] <= row['prefix_state']['time'] + 1e-8][-1]
    positions = np.array([prefix['robot']['positions']] + [f['robot']['positions'] for f in frames])
    path = float(np.linalg.norm(np.diff(positions,axis=0),axis=2).sum())
    close(path,row['recovery_path_m'],'Interface path length mismatch')
    closed = [f for f in frames if f['stage']=='global_close'][-1]
    hover=next(f['time'] for f in frames if f['stage']=='global_hover')
    prior=[f for f in frames if f['time']<hover-1e-8]
    result.update(path_m=path, close_holds=closed['evaluator']['holds'],
        pretarget_support_violation=longest([not(any(f['evaluator']['holds']) or f['evaluator']['table_contact']) for f in prior])>2,
        close_miss_m=float(np.linalg.norm(np.asarray(closed['robot']['positions'])-closed['evaluator']['handle_positions'],axis=1).max()),
        ever_dual_regrasp=any(all(f['evaluator']['holds']) for f in row['trace'] if f['stage'] in
            ('global_close','global_lift','global_lift_settle','final_stable_hold')))
    if row['method']=='fixed':
        check('regrasp_decision' not in row,'Fixed reference unexpectedly inspected')
        return None
    d=row['regrasp_decision'];p=d['input'];estimate=geometry.fit(p)
    check(estimate['valid']==d['estimate']['valid'],'RGB-fit validity mismatch')
    for k in ('center','rotation','targets','optimization_vector'):
        check(np.allclose(estimate[k],d['estimate'][k],atol=1e-5,rtol=0),'RGB-fit mismatch: '+k)
    check(d['apply']==(row['method']=='retarget'),'Target use mismatch')
    check(d['service_wait_steps']==40 and d['service_wall_seconds']<=2.,'Inspection cost slot mismatch')
    at=[f for f in frames if f['stage']=='global_clear'][-1]
    close(at['time'],d['observed_time'],'RGB observation-time mismatch')
    held=[f for f in frames if f['stage']=='regrasp_observation_wait']
    check(len(held)==40,'Missing inspection-hold samples')
    logged=d['estimate']
    handles=np.asarray(logged['center'])+np.asarray([[0.,.155,.06],[0.,-.155,.06]])@np.asarray(logged['rotation']).T
    truth=np.asarray(d['observation_truth']['handle_positions'])
    errors=np.linalg.norm(handles-truth,axis=1)
    check(np.allclose(errors,d['estimated_handle_error_m'],atol=1e-10,rtol=0),'RGB geometry diagnostic mismatch')
    return {'seed':row['seed'],'scenario':row['scenario'],'load':row['load'],'window':row['window'],'method':row['method'],
        'handle_error_mm':errors*1000,
        'entry_handle_error_mm':np.linalg.norm(handles-np.asarray(held[-1]['evaluator']['handle_positions']),axis=1)*1000,
        'fixed_miss_cm':float(np.linalg.norm(np.asarray(p['reference_targets'])-truth,axis=1).max())*100,
        'retarget_miss_cm':float(np.linalg.norm(np.asarray(logged['targets'])-truth,axis=1).max())*100}


def evaluation_layers(row, result, cohort, graph_calls):
    """Paired scoring views, retaining the cohort's original admission/acceptance.

    A pickup suffix is a stage restriction within one global call. It is never
    relabeled as the last recovery attempt. Exact graph-call boundaries are
    projected from the original execution events, rather than inferred from stages.
    """
    frames = [f for f in row['trace'] if f['phase'] in ('recovery', 'final_hold')]
    check(not row['protection']['deep_interarm_contact_steps'],
          'These layer witnesses require their recorded zero collision count')

    def accepted_suffix(start):
        tail = [f for f in frames if f['time'] > start + 1e-8]
        unsupported = longest([not (any(f['evaluator']['holds']) or f['evaluator']['table_contact']) for f in tail])
        role_bad = any(f['phase'] == 'recovery' and
                       (keeper := {'R_left': 1, 'R_right': 0}.get(f.get('skill', row['candidate']))) is not None
                       and not f['evaluator']['holds'][keeper] for f in tail)
        return bool(result['V'] and result['accepted'] and unsupported <= 2 and not role_bad
                    and not row['blocked_calls'] and row.get('failure_reason') != 'numerical_instability')

    calls = graph_calls[row['seed'], row['scenario']] if cohort == 'scope' else [
        {'candidate': row['candidate'], 'start_time': row['prefix_state']['time']}]
    nontrivial = [c for c in calls if c['candidate'] != 'continue']
    start = nontrivial[-1]['start_time'] if nontrivial else row['prefix_state']['time']
    pickup = next((f['time'] - DT for f in frames if f['stage'] == 'global_hover'), None)
    return {'cohort': cohort, 'seed': row['seed'], 'scenario': row['scenario'],
            'method': row['method'], 'load': row.get('load'), 'window': row.get('window'),
            'terminal': bool(result['V'] and result['C']),
            'accepted': bool(result['V'] and result['accepted']),
            'shared': bool(result['V'] and result['shared_guard_completion']),
            'full': bool(result['Y']), 'last_call': accepted_suffix(start),
            'pickup_suffix': accepted_suffix(pickup) if pickup is not None else None,
            'recovery_calls': len(nontrivial)}


def interface_metrics(rows):
    return {**metrics(rows),'setdown':sum(r['table'] for r in rows),
        'ever_dual_regrasp':sum(r['ever_dual_regrasp'] for r in rows),
        'close_dual_grasp':sum(all(r['close_holds']) for r in rows),
        'pretarget_support_violations':sum(r['pretarget_support_violation'] for r in rows),
        'completed_with_prior_violation':sum(r['C'] and r['pretarget_support_violation'] for r in rows),
        'close_grasp_histogram':dict(Counter(sum(r['close_holds']) for r in rows)),
        'path_m':float(np.mean([r['path_m'] for r in rows])),
        'close_worst_miss_cm':100*float(np.mean([r['close_miss_m'] for r in rows]))}


def interface_pair(rows, first, second):
    groups=defaultdict(dict)
    for r in rows:groups[r['seed'],r['scenario']][r['method']]=r
    seeds=sorted({k[0] for k in groups})
    delta=np.array([sum(int(g[first]['Y'])-int(g[second]['Y']) for k,g in groups.items() if k[0]==s) for s in seeds])
    draw=np.random.default_rng(20261111).integers(0,len(seeds),(2000,len(seeds)))
    sample=delta[draw].sum(1)/(2*len(seeds))
    common=[g for g in groups.values() if g[first]['Y'] and g[second]['Y']]
    return {'states':len(groups),'seeds':len(seeds),'first':first,'second':second,
        'Y_difference':float(delta.sum()/len(groups)),'Y_difference_ci95':np.quantile(sample,[.025,.975]).tolist(),
        'Y_gains':sum(g[first]['Y'] and not g[second]['Y'] for g in groups.values()),
        'Y_losses':sum(g[second]['Y'] and not g[first]['Y'] for g in groups.values()),
        'all_attempt_time_difference':float(np.mean([g[first]['total']-g[second]['total'] for g in groups.values()])),
        'common_protected_states':len(common),
        'common_protected_time_difference':float(np.mean([g[first]['total']-g[second]['total'] for g in common])) if common else None}


def inside(event, t):
    return event['start_time'] + 1e-8 < t <= event['end_time'] + 1e-8


def handover_roles(row, frames):
    # Reconstruct from physical intervals, including the already documented
    # final-packet annotation correction. Never consume a saved satisfaction flag.
    carrier, handoff = {}, {}
    previous = None
    for f in frames:
        role = None
        stage, truth, t = f['stage'], f['evaluator'], f['time']
        if f['phase'] in ('release', 'hold'):
            role = 'receiver_supports'
        elif stage in WAIT:
            decisions = [d for d in row['decision_events'] if inside(d, t)]
            if decisions:
                d = decisions[0]
                role = ('receiver_supports' if d['plan'] == 'continue' else 'giver_holds'
                        if d['plan'] == 'R1_retarget' or d['kind'] == 'post_local_check' else None)
            elif stage == 'controller_compute_wait':
                role = previous
            else:
                raise ValueError('Unmatched decision wait')
        else:
            calls = [c for c in row['skill_calls'] if inside(c, t)]
            call = min(calls, key=lambda c: c['end_time'] - c['start_time']) if calls else None
            if call:
                skill = call['skill']
                if skill in ('R1_retarget', 'R3_receiver_catch'):
                    role = 'giver_holds'
                elif skill == 'R3_transfer_R2':
                    role = 'receiver_holds'
                elif skill == 'R3':
                    idx = call['index']
                    if stage in ('r3_table_loaded', 'r3_table_release', 'r3_table_settle') and truth['table_contact']:
                        handoff[idx] = True
                    if stage in BEFORE_TABLE:
                        if idx not in carrier:
                            commands = f['robot'].get('gripper_commands')
                            if commands is None:
                                commands = [f['robot']['action14'][k] for k in (6, 13)]
                            positive = [i for i, command in enumerate(commands) if command > 0]
                            if len(positive) == 1:
                                carrier[idx] = positive[0]
                        if not handoff.get(idx, False) and idx in carrier:
                            role = 'giver_holds' if carrier[idx] == 0 else 'receiver_holds'
                    elif stage in ('r3_return_to_handover', 'r3_handover_settle'):
                        role = 'giver_holds'
        yield role
        previous = role


def score_handover(row):
    frames, total = frame_cost(row, ('recovery', 'release', 'hold'))
    for f in frames:
        e = f['evaluator']
        check(e['receiver_supports'] == bool(e['receiver_holds'] and e['clearance'] > .10
                                           and e['relative_speed'] <= .20), 'Receiver support mismatch')
    by_time = {f['time']: f['evaluator'] for f in row['trace']}
    valid = True
    for event in row['decision_events']:
        required = {'continue': 'receiver_supports', 'R1_retarget': 'giver_holds', 'R3': None}[event['plan']]
        check(required == event['required_entry_predicate'], 'Decision admission specification mismatch')
        observed = bool(by_time[event['start_time']][required]) if required else True
        entered = bool(by_time[event['end_time']][required]) if required else True
        check(observed == event['prerequisite_at_observation'] and entered == event['prerequisite_at_entry'],
              'Decision admission truth mismatch')
        valid &= entered
    for call in row['skill_calls']:
        required = {'R1_retarget': 'giver_holds', 'R3_receiver_catch': 'giver_holds',
                    'R3_transfer_R2': 'receiver_holds', 'R3': None}[call['skill']]
        entered = bool(call['start_evaluator'][required]) if required else True
        check(entered == call['valid_at_entry'], 'Skill admission truth mismatch')
        valid &= entered
    release = [f for f in frames if f['phase'] == 'release' and f['stage'] not in WAIT]
    held = [f for f in frames if f['phase'] == 'hold' and f['stage'] not in WAIT]
    last = [f for f in frames if f['phase'] == 'release'][-1]
    fingers = last['robot']['actual_finger_qpos4'][:2]
    released = not last['evaluator']['giver_holds'] and min(abs(x) for x in fingers) >= .03
    all_held = [f for f in frames if f['phase'] == 'hold']
    stable = bool(last['evaluator']['receiver_supports'] and len(all_held) >= 20
                  and all(f['evaluator']['receiver_supports'] and not f['evaluator']['giver_holds'] for f in all_held))
    complete = bool(released and stable and not any(row['bad_state_warnings'].values())
                    and row.get('failure_reason') != 'numerical_instability')
    table = any(f['stage'] == 'r3_table_settle' and f['evaluator']['table_contact']
                and not f['evaluator']['giver_holds'] and not f['evaluator']['receiver_holds'] for f in frames)
    lifted = any(f['stage'] in ('r3_pickup_lift', 'r3_return_to_handover')
                 and f['evaluator']['giver_holds'] and f['evaluator']['clearance'] > .10 for f in frames)
    cycle = not row['used_global_recovery'] or (table and lifted)
    unsupported = longest([not (f['evaluator']['giver_holds'] or f['evaluator']['receiver_holds']
                               or f['evaluator']['table_contact']) for f in frames])
    shared_bad = bool(unsupported > 2 or row['protection']['deep_robot_robot_contact_steps'])
    role_bad = [role is not None and not f['evaluator'][role] for f, role in zip(frames, handover_roles(row, frames))]
    hard_bad = bool(row['blocked_calls'] or row.get('failure_reason') == 'numerical_instability')
    bad = bool(shared_bad or any(role_bad) or hard_bad or not cycle)
    label = 'E' if not valid else 'A' if complete and not bad else 'B' if complete else 'D' if bad else 'C'
    waiting = sum(f['stage'] in WAIT for f in frames) * DT
    skill = sum(f['phase'] == 'recovery' and f['stage'] not in WAIT for f in frames) * DT
    release_time, hold = len(release) * DT, len(held) * DT
    recorded = row.get('recorded', row.get('audit'))
    check(complete == row['handover_completed'], 'Handover completion mismatch')
    check(label == recorded['label'], 'Handover label mismatch')
    for key, value in [('total_seconds', total), ('wait_seconds', waiting), ('recovery_motion_seconds', skill),
                       ('release_seconds', release_time), ('verification_seconds', hold)]:
        close(value, recorded[key], f'Handover cost mismatch: {key}')
    result = base(row, complete, bool(valid), bad, label, total, waiting, skill, release_time, hold)
    result.update(shared_bad=shared_bad, role_loss_samples=sum(role_bad), table=table, cycle=cycle,
                  local_then_global=row['recovery']['sequence'] == ['R1_retarget', 'R3'])
    return result


def base(row, complete, valid, bad, label, total, wait, skill, release, hold):
    result = {k: row[k] for k in ('seed', 'scenario', 'method', 'candidate', 'load', 'window',
                                  'planned_wait_steps', 'planned_duration_steps') if k in row}
    result.update(C=bool(complete), V=bool(valid), P=not bad, Y=label == 'A', label=label,
                  total=total, wait=wait, skill=skill, release=release, hold=hold)
    return result


def mlp_scores(port, values):
    check(set(port) == {'positions', 'fingers', 'reference_positions'}, 'MLP observation permission mismatch')
    fingers = np.abs(np.asarray(port['fingers']))
    widths = fingers.mean(1)
    loaded = (widths > values['rule']['loaded_lower']) & (widths < values['rule']['loaded_upper'])
    asym = np.abs(fingers[:, 0] - fingers[:, 1]) / np.maximum(fingers.sum(1), 1e-9)
    positions, ref = np.asarray(port['positions']), np.asarray(port['reference_positions'])
    height = abs((positions[0, 2] - positions[1, 2]) - (ref[0, 2] - ref[1, 2]))
    x = np.r_[widths, loaded.astype(float), asym, height]
    z = (np.atleast_2d(x) - values['mean']) / values['scale']
    logits = np.tanh(z @ values['w1'] + values['b1']) @ values['w2'] + values['b2']
    probabilities = (1 / (1 + np.exp(-np.clip(logits, -35, 35))))[0]
    mask = [bool(loaded.all()), bool(loaded[1]), bool(loaded[0]), True]
    return probabilities, np.asarray(mask, dtype=bool)


def mlp_decision(row, values):
    probabilities, mask = mlp_scores(row['selection']['input'], values)
    candidates = ('continue', 'R_left', 'R_right', 'R_global')
    chosen = next((candidates[j] for j in np.argsort(values['action_cost_seconds'], kind='stable')
                   if probabilities[j] >= values['threshold'] and mask[j]), 'abstain')
    expected = row['selection']['decision']
    check(chosen == expected['candidate'], 'Frozen MLP choice mismatch')
    check(np.allclose(probabilities, expected['safety_scores'], rtol=0, atol=1e-12), 'Frozen MLP score mismatch')


def waiting_decision(row, values):
    h = row['wait_decision']['history']
    mode = row['method']
    use = h[:1] if mode == 'cached' else h[-1:] if mode == 'refreshed' else h
    scores, masks = zip(*(mlp_scores(x['observation'], values) for x in use))
    score, mask = np.asarray(scores).min(0), np.asarray(masks).all(0)
    names = ('continue', 'R_left', 'R_right', 'R_global')
    chosen = next((names[j] for j in np.argsort(values['action_cost_seconds'], kind='stable')
                   if mask[j] and score[j] >= values['threshold']), 'abstain')
    recorded = row['selection']['decision']
    check(chosen == row['candidate'] == recorded['candidate'], 'Waiting decision mismatch')
    check(np.allclose(score, recorded['safety_scores'], atol=1e-12, rtol=0), 'Waiting score mismatch')
    check(mask.tolist() == recorded['start_admissible'], 'Waiting start-mask mismatch')
    check(all(x['elapsed_seconds'] <= row['wait_decision']['decision_time'] - row['prefix_state']['time'] + 1e-8 for x in h), 'Future observation used')
    check(all(f.get('skill') == 'continue' for f in row['trace'] if f['stage'] in ('inference_wait', 'decision_budget_wait')), 'Uncommitted wait role mismatch')


def metrics(rows):
    return {'episodes': len(rows), 'C': sum(r['C'] for r in rows), 'Y': sum(r['Y'] for r in rows),
            'violations': sum(not r['P'] for r in rows),
            **{k: float(np.mean([r[k] for r in rows])) for k in ('total', 'wait', 'skill', 'release', 'hold')},
            'labels': dict(Counter(r['label'] for r in rows)),
            'global_choices': sum(r['candidate'] == 'R_global' for r in rows),
            'local_then_global': sum(r.get('local_then_global', False) for r in rows)}


def paired(rows, first, second, allowed=None, seed=20261109):
    groups = defaultdict(dict)
    for r in rows:
        key = (r['seed'], r['scenario'])
        if allowed is None or key in allowed:
            groups[key][r['method']] = r
    groups = {k: g for k, g in groups.items() if first in g and second in g}
    seeds = sorted({k[0] for k in groups})
    draw = np.random.default_rng(seed).integers(0, len(seeds), (2000, len(seeds)))
    common = [(key, g[first], g[second]) for key, g in groups.items() if g[first]['Y'] and g[second]['Y']]
    check(bool(common), 'No common-protected pairs')
    totals = np.array([[sum(a['total'] for k, a, b in common if k[0] == s),
                        sum(b['total'] for k, a, b in common if k[0] == s)] for s in seeds])
    sample = totals[draw].sum(1)
    sample = sample[sample[:, 1] > 0]
    a, b = [float(np.mean([r[i]['total'] for r in common])) for i in (1, 2)]
    return {'states': len(common), 'seeds': len(seeds), 'first_mean': a, 'second_mean': b,
            'reduction_fraction': 1 - a / b,
            'ci95': np.quantile(1 - sample[:, 0] / sample[:, 1], [.025, .975]).tolist()}


def table(columns, lines):
    return '\n'.join([r'\begin{tabular}{' + columns + '}', r'\toprule', *lines,
                      r'\bottomrule', r'\end{tabular}']) + '\n'


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE / 'results')
    args = parser.parse_args()
    out = args.out.resolve()
    check(not out.exists(), 'Use a new output directory; existing results are never overwritten')
    out.mkdir(parents=True)
    (out / 'tables').mkdir()
    config = json.loads((HERE / 'data/config.json').read_text())
    model = json.loads((HERE / 'models/direct-selected.json').read_text())['mlp']
    scored = {}
    mlp_count = 0
    waiting_count = 0
    geometry_spec=importlib.util.spec_from_file_location('interface_geometry',HERE/'interface_geometry.py')
    geometry=importlib.util.module_from_spec(geometry_spec);geometry_spec.loader.exec_module(geometry)
    controller_spec=importlib.util.spec_from_file_location('controller_audit',HERE/'controller_audit.py')
    controller=importlib.util.module_from_spec(controller_spec);controller_spec.loader.exec_module(controller)
    prospective=None;prospective_geometry=None;prospective_fits=0
    if 'wait_hold' in config['episodes']:
        spec=importlib.util.spec_from_file_location('wait_geometry_audit',HERE/'wait_geometry_audit.py')
        prospective=importlib.util.module_from_spec(spec);spec.loader.exec_module(prospective)
        sys.modules['wait_geometry_audit']=prospective
        spec=importlib.util.spec_from_file_location('prospective_geometry',HERE/'prospective_geometry.py')
        prospective_geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(prospective_geometry)
    holding=None;landmark=None;holding_fits=0;landmark_fits=0;landmark_rgb_examples=0
    if 'holding_components' in config['episodes']:
        spec=importlib.util.spec_from_file_location('holding_components_audit',HERE/'holding_components_audit.py')
        holding=importlib.util.module_from_spec(spec);spec.loader.exec_module(holding)
        sys.modules['holding_components_audit']=holding
        spec=importlib.util.spec_from_file_location('landmark_visual',HERE/'landmark_visual.py')
        landmark=importlib.util.module_from_spec(spec);spec.loader.exec_module(landmark)
    paired_audit=None;paired_fits=0;paired_landmark_fits=0;paired_states={}
    if 'paired_holding' in config['episodes']:
        spec=importlib.util.spec_from_file_location('paired_holding_audit',HERE/'paired_holding_audit.py')
        paired_audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(paired_audit)
        sys.modules['paired_holding_audit']=paired_audit
    execution_error=None;execution_error_rows=[]
    if (HERE/'reference/execution-error-results.json').exists():
        spec=importlib.util.spec_from_file_location('execution_error_audit',HERE/'execution_error_audit.py')
        execution_error=importlib.util.module_from_spec(spec);spec.loader.exec_module(execution_error)
    lifting_smooth=None;lifting_smooth_states=[];lifting_smooth_fits=0
    if 'lifting_smooth' in config['episodes']:
        spec=importlib.util.spec_from_file_location('lifting_smooth_audit',HERE/'lifting_smooth_audit.py')
        lifting_smooth=importlib.util.module_from_spec(spec);spec.loader.exec_module(lifting_smooth)
    public_policy=None;public_policy_states=[]
    if 'public_transport' in config['episodes']:
        spec=importlib.util.spec_from_file_location('public_policy_audit',HERE/'public_policy_audit.py')
        public_policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(public_policy)
    handover_holding=None;handover_holding_states=[]
    if 'handover_holding' in config['episodes']:
        spec=importlib.util.spec_from_file_location('handover_holding_audit',HERE/'handover_holding_audit.py')
        handover_holding=importlib.util.module_from_spec(spec);spec.loader.exec_module(handover_holding)
    sampling_spec=importlib.util.spec_from_file_location('sampling_audit',HERE/'sampling_audit.py')
    sampling=importlib.util.module_from_spec(sampling_spec);sampling_spec.loader.exec_module(sampling)
    history_manifest=json.loads((HERE/'data/history-cases.json').read_text()) if (HERE/'data/history-cases.json').exists() else {'cases':[]}
    history_by_key={(r['seed'],r['scenario'],r['method'],r['load'],r['window']):r for r in history_manifest['cases']}
    history_rows=[];pilot_fits=0
    controller_fits=0
    image_example_count=0
    for file in sorted((HERE/'figures/interface').glob('input-*.json')):
        example=json.loads(file.read_text())
        with np.load(file.parent/example['data_file'],allow_pickle=False) as arrays:
            for i,camera in enumerate(geometry.CAMERAS):
                rgb=arrays[camera]
                check(hashlib.sha256(rgb.tobytes()).hexdigest()==example['rgb_sha256'][i],'RGB example digest mismatch')
                calculated=geometry.mask_features(rgb);expected=example['port']['views'][i]['colors']
                for a,b in zip(calculated,expected):
                    check(a['pixels']==b['pixels'],'RGB mask example pixel mismatch')
                    for key in ('support','center'):
                        check(a[key] is None and b[key] is None or a[key] is not None and b[key] is not None and np.allclose(a[key],b[key],atol=1e-10,rtol=0),'RGB silhouette example mismatch')
                image_example_count+=1
    check(image_example_count==8,'Incomplete policy-input RGB examples')
    prospective_rgb_examples=0
    if prospective is not None:
        for file in sorted((HERE/'figures/prospective').glob('input-*.json')):
            example=json.loads(file.read_text());prospective_geometry.configure(example['geometry'])
            with np.load(file.parent/example['data_file'],allow_pickle=False) as arrays:
                for i,camera in enumerate(prospective_geometry.CAMERAS):
                    rgb=arrays[camera]
                    check(hashlib.sha256(rgb.tobytes()).hexdigest()==example['rgb_sha256'][i],'Prospective RGB digest mismatch')
                    actual=prospective_geometry.mask_features(rgb);expected=example['port']['views'][i]['colors']
                    for a,b in zip(actual,expected):
                        check(a['pixels']==b['pixels'],'Prospective RGB mask mismatch')
                        for key in ('support','center'):
                            check(a[key] is None and b[key] is None or a[key] is not None and b[key] is not None and np.allclose(a[key],b[key],atol=1e-10,rtol=0),'Prospective RGB feature mismatch')
                    prospective_rgb_examples+=1
        check(prospective_rgb_examples==4*(1+int('geometry' in config['episodes'])),'Incomplete prospective RGB examples')
    if holding is not None:
        for file in sorted((HERE/'figures/landmark').glob('input-*.json')):
            example=json.loads(file.read_text())
            with np.load(file.parent/example['data_file'],allow_pickle=False) as arrays:
                for i,camera in enumerate(landmark.CAMERAS):
                    rgb=arrays[camera]
                    check(hashlib.sha256(rgb.tobytes()).hexdigest()==example['rgb_sha256'][i],'Landmark RGB digest mismatch')
                    actual=landmark.features(rgb);expected=example['port']['views'][i]['colors']
                    for a,b in zip(actual,expected):
                        check(a['pixels']==b['pixels'],'Landmark RGB pixel mismatch')
                        check(a['center'] is None and b['center'] is None or a['center'] is not None and b['center'] is not None and np.allclose(a['center'],b['center'],rtol=0,atol=1e-10),'Landmark RGB centroid mismatch')
                    landmark_rgb_examples+=1
        check(landmark_rgb_examples==8,'Incomplete landmark policy-input examples')
    interface_observations=[]
    graph_calls = {(r['seed'], r['scenario']): r['calls']
                   for r in json.loads((HERE / 'data/scope-calls.json').read_text())}
    layers = []
    for cohort in config['episodes']:
        scored[cohort] = []
        if cohort in ('paired_holding','paired_placement'):paired_states[cohort]=[]
        for row in read(cohort):
            if cohort=='public_transport':
                r=public_policy.score(row)
                public_policy_states.append(public_policy.pairing_record(row))
            elif cohort=='handover_holding':
                if row['holding_input'] is not None:
                    check(row['initialization_supported'] and row['external_wrench_at_recovery_start']==0,'Invalid handover holding prefix')
                r=handover_holding.score(row,HERE/'data/handover_holding-native'/row['native_file'],sys.modules[__name__])
                state={k:row[k] for k in ('seed','scenario','method','hold_mode','wait_steps',
                    'initialization_state','initialization_end_state','prefix_state','registered_robot_poses','holding_input') if k in row}
                state['decision_events']=row['decision_events'][:1]
                handover_holding_states.append(state)
            else:
                r = score_handover(row) if cohort == 'handover' else score_lift(row, cohort)
            if cohort=='lifting_smooth':
                prospective_geometry.configure(row['geometry'])
                r,n=lifting_smooth.score(row,HERE/'data/lifting_smooth-native'/row['native_file'],
                    score_lift,controller,prospective_geometry.fit)
                lifting_smooth_fits+=n
                lifting_smooth_states.append({k:row[k] for k in ('seed','scenario','hold_mode','wait_steps',
                    'initialization_state','preclose_state','initialization_end_state','prefix_state','post_wait_state')})
            if cohort in ('wait_hold','geometry'):
                prospective_geometry.configure(row['geometry'])
                r,n=prospective.score(row,HERE/'data'/(cohort+'-native')/row['native_file'],
                    score_lift,controller,prospective_geometry.fit)
                prospective_fits+=n
            if cohort in ('holding_components','independent_visual'):
                prospective_geometry.configure(row['geometry'])
                r,n=holding.score(row,HERE/'data'/(cohort+'-native')/row['native_file'],score_lift,controller,
                    prospective_geometry.fit if cohort=='holding_components' else None)
                holding_fits+=n
                if paired_audit is not None:
                    r.update(paired_audit.diagnostics(row,HERE/'data'/(cohort+'-native')/row['native_file']))
                if cohort=='independent_visual':
                    for observation in row['global_execution']['observations']:
                        estimate=landmark.estimate(observation['input']);landmark_fits+=1
                        check(estimate['valid']==observation['estimate']['valid'],'Landmark estimate validity mismatch')
                        if estimate['valid']:
                            for key in ('center','rotation','targets','rotation_targets','landmarks','reference_landmarks','rigid_residual_mm'):
                                check(np.allclose(estimate[key],observation['estimate'][key],atol=1e-8,rtol=0),'Landmark estimate mismatch: '+key)
            if cohort in ('paired_holding','paired_placement'):
                paired_states[cohort].append({k:row[k] for k in ('seed','scenario','method','hold_mode','wait_steps',
                    'initialization_state','preclose_state','initialization_end_state','prefix_state','post_wait_state')})
                prospective_geometry.configure(row['geometry'])
                r,n=paired_audit.score(row,HERE/'data'/(cohort+'-native')/row['native_file'],score_lift,controller,
                    prospective_geometry.fit if row['method']!='visual_servo' else None)
                paired_fits+=n
                if row['method']=='visual_servo':
                    for observation in row['global_execution']['observations']:
                        estimate=landmark.estimate(observation['input']);paired_landmark_fits+=1
                        check(estimate['valid']==observation['estimate']['valid'],'Paired landmark validity mismatch')
                        if estimate['valid']:
                            for key in ('center','rotation','targets','rotation_targets','landmarks','reference_landmarks','rigid_residual_mm'):
                                check(np.allclose(estimate[key],observation['estimate'][key],atol=1e-8,rtol=0),'Paired landmark estimate mismatch: '+key)
            if cohort == 'controller':
                r,observations=controller.score(row,score_lift)
                r.update(controller.native_view(HERE/'data/controller-native'/row['native_file'],row,r))
                for observation in observations:
                    estimate=geometry.fit(observation['input']);controller_fits+=1
                    check(estimate['valid']==observation['estimate']['valid'],'Controller RGB validity mismatch')
                    if estimate['valid']:
                        for key in ('center','rotation','targets','optimization_vector'):
                            check(np.allclose(estimate[key],observation['estimate'][key],atol=1e-5,rtol=0),'Controller RGB-fit mismatch: '+key)
            if cohort=='paired_holding' and execution_error is not None:
                execution_error_rows.append(execution_error.score(row,r))
            if cohort=='controller_pilot':
                r,observations=sampling.pilot(row,score_lift,controller.native_view,HERE/'data')
                for observation in observations:
                    estimate=geometry.fit(observation['input']);pilot_fits+=1
                    check(estimate['valid']==observation['estimate']['valid'],'Pilot RGB validity mismatch')
                    if estimate['valid']:
                        for key in ('center','rotation','targets','optimization_vector'):
                            check(np.allclose(estimate[key],observation['estimate'][key],atol=1e-5,rtol=0),'Pilot RGB fit mismatch: '+key)
            if cohort=='interface':
                observation=interface_case(row,r,geometry)
                if observation is not None:interface_observations.append(observation)
                key=(row['seed'],row['scenario'],row['method'],row['load'],row['window'])
                if key in history_by_key:history_rows.append(sampling.history_view(row,history_by_key[key],r,controller.native_view,HERE/'data'))
            if (cohort == 'scope' and row['method'] == 'agentchord_style' and row['scenario'] != 'sham'
                    or cohort == 'supervision' and row['method'] == 'ce'
                    or cohort == 'interface' and row['method'] == 'retarget'
                    and row['load'] == 'density150' and row['scenario'] != 'sham'):
                layers.append(evaluation_layers(row, r, cohort, graph_calls))
            if cohort == 'supervision':
                decision = row.get('duration_selection', {})
                r['admitted'] = bool(decision.get('decision', {}).get('model_admitted')) and not decision.get('baseline_fallback', False)
                r['fallback'] = bool(decision.get('baseline_fallback', False))
            scored[cohort].append(r)
            if cohort in ('scope', 'load') and row['method'] == 'gated_direct':
                mlp_decision(row, model)
                mlp_count += 1
            if cohort == 'wait_decision':
                waiting_decision(row, model)
                waiting_count += 1
        check(len(scored[cohort]) == config['episodes'][cohort], 'Incomplete cohort')
        identity = lambda r: tuple(r.get(k) for k in ('seed', 'scenario', 'method', 'load', 'window', 'planned_wait_steps', 'planned_duration_steps','hold_mode','geometry','wait_steps'))
        check(len({identity(r) for r in scored[cohort]}) == len(scored[cohort]), 'Duplicate execution')
        check(len({r['seed'] for r in scored[cohort]}) == (8 if cohort == 'waiting' else 20), 'Initialization-seed denominator mismatch')
    tables, aggregated = {}, []
    layer_lines = [r'Cohort & Goal & $+S$ & $+O$ & Last call & Pickup\\', r'\midrule']
    layer_groups = [('Graph', [r for r in layers if r['cohort'] == 'scope']),
                    ('SFT', [r for r in layers if r['cohort'] == 'supervision']),
                    ('RGB, actual', [r for r in layers if r['cohort'] == 'interface' and r['window'] == 'actual']),
                    ('RGB, 1 s', [r for r in layers if r['cohort'] == 'interface' and r['window'] == '1000ms'])]
    layer_summary = {}
    for name, rows in layer_groups:
        n = len(rows)
        values = {k: sum(r[k] for r in rows) for k in ('terminal', 'accepted', 'shared', 'full', 'last_call')}
        values.update(episodes=n, independent_seeds=len({r['seed'] for r in rows}),
                      pickup_suffix=sum(bool(r['pickup_suffix']) for r in rows) if all(r['pickup_suffix'] is not None for r in rows) else None,
                      last_call_only_accepts=sum(r['last_call'] and not r['full'] for r in rows),
                      pickup_only_accepts=sum(bool(r['pickup_suffix']) and not r['full'] for r in rows))
        layer_summary[name] = values
        text_cells = [f"{values[k]}/{n}" for k in ('terminal', 'shared', 'full', 'last_call')]
        text_cells.append('--' if values['pickup_suffix'] is None else f"{values['pickup_suffix']}/{n}")
        layer_lines.append(name + ' & ' + ' & '.join(text_cells) + r'\\')
    tables['evaluation-layers'] = table('lrrrrr', layer_lines)
    hand_lines = [r'Window & Policy & Fault $C$ & Fault $Y$ & Nominal $Y$ & Wait (s) & Total (s) & R1$\to$R3 \\', r'\midrule']
    for window, title in [('actual', 'Measured'), ('1000ms', '1 s min.'), ('2000ms', '2 s min.')]:
        for method, name in zip(HANDOVER, HANDOVER_NAMES):
            faults = [r for r in scored['handover'] if r['window'] == window and r['method'] == method and r['scenario'] != 'nominal']
            nominal = [r for r in scored['handover'] if r['window'] == window and r['method'] == method and r['scenario'] == 'nominal']
            m, n = metrics(faults), metrics(nominal)
            aggregated.append({'cohort': 'handover', 'load': '', 'window': window, 'method': method, **m})
            hand_lines.append(f"{title} & {name} & {m['C']}/{m['episodes']} & {m['Y']}/{m['episodes']} & {n['Y']}/{n['episodes']} & {m['wait']:.3f} & {m['total']:.3f} & {m['local_then_global']} " + r'\\')
    tables['handover-audit'] = table('llrrrrrr', hand_lines)
    if paired_audit is not None:
        tables['handover-focus']=paired_audit.handover_focus(hand_lines)
    scope_lines = [r'Policy & Safe & Complete & Total (s) \\', r'\midrule']
    for method, name in zip(SCOPE, SCOPE_NAMES):
        m = metrics([r for r in scored['scope'] if r['method'] == method and r['scenario'] != 'sham'])
        aggregated.append({'cohort': 'scope', 'load': '', 'window': 'actual', 'method': method, **m})
        scope_lines.append(f"{name} & {m['Y']}/{m['episodes']} & {m['C']}/{m['episodes']} & {m['total']:.3f} " + r'\\')
    tables['scope'] = table('lrrr', scope_lines)
    load_lines = [r'Wait & Policy & Fault $Y$ & Healthy $Y$ & Total (s) & Global \\', r'\midrule']
    for load in ('native', 'density150'):
        for window, title in [('actual', 'Actual'), ('1000ms', '1 s')]:
            if load == 'density150' and window != 'actual':
                load_lines.append(r'\midrule')
            for method, name in zip(LOAD, LOAD_NAMES):
                faults = [r for r in scored['load'] if (r['load'], r['window'], r['method']) == (load, window, method) and r['scenario'] != 'sham']
                healthy = [r for r in scored['load'] if (r['load'], r['window'], r['method']) == (load, window, method) and r['scenario'] == 'sham']
                m, n = metrics(faults), metrics(healthy)
                aggregated.append({'cohort': 'load', 'load': load, 'window': window, 'method': method, **m})
                if load == 'density150':
                    load_lines.append(f"{title if method == 'protected' else ''} & {name} & {m['Y']}/{m['episodes']} & {n['Y']}/{n['episodes']} & {m['total']:.3f} & {m['global_choices']} " + r'\\')
    tables['load-transfer'] = table('llrrrr', load_lines)
    protection_lines = [r'Policy & $S$ fail & $O$ only & $Y_0$ & $Y_{50,100}$ \\', r'\midrule']
    for cohort, methods, names, title in [('scope', SCOPE, SCOPE_NAMES, 'Scope: 60 faults per policy'),
                                         ('supervision', SUPERVISION, SUPERVISION_NAMES, 'Supervision: 40 faults per policy')]:
        protection_lines.append(r'\multicolumn{5}{l}{\textit{' + title + r'}}\\')
        for method, name in zip(methods, names):
            rows = [r for r in scored[cohort] if r['method'] == method and r['scenario'] != 'sham']
            y50, y100 = sum(r['safe50'] for r in rows), sum(r['safe100'] for r in rows)
            check(y50 == y100, 'Paper combines two equal secondary tolerance counts')
            protection_lines.append(f"{name} & {sum(r['shared_bad'] for r in rows)} & {sum(r['role_only'] for r in rows)} & {sum(r['Y'] for r in rows)} & {y50} " + r'\\')
    tables['protection-validity'] = table('lrrrr', protection_lines)
    supervision_lines = [r'Policy & Safe & Complete & Violations & Wait (s) & Total (s) & Admitted & Fallback \\', r'\midrule']
    for method, name in zip(SUPERVISION, SUPERVISION_NAMES):
        rows = [r for r in scored['supervision'] if r['method'] == method]
        m = metrics(rows)
        admitted = sum(r['admitted'] for r in rows)
        fallback = sum(r['fallback'] for r in rows)
        aggregated.append({'cohort': 'supervision', 'load': '', 'window': 'actual', 'method': method, **m})
        supervision_lines.append(f"{name} & {m['Y']}/{m['episodes']} & {m['C']}/{m['episodes']} & {m['violations']} & {m['wait']:.3f} & {m['total']:.3f} & {admitted} & {fallback} " + r'\\')
    tables['supervision'] = table('lrrrrrrr', supervision_lines)
    wait_groups = defaultdict(dict)
    for r in scored['waiting']:
        wait_groups[(r['seed'], r['scenario'])][(r['planned_wait_steps'], r['planned_duration_steps'])] = r
    selection_lines = [r'State & $W{=}0$ choice & Outcome after wait & New observed A \\', r'\midrule']
    choice_changes = []
    for (seed, scenario), group in sorted(wait_groups.items()):
        choose = lambda w: next((d for d in (30, 40, 60) if group[w, d]['Y']), None)
        before, after = choose(0), choose(19)
        if before != after:
            delayed = group[19, before]['label']
            choice_changes.append({'seed': seed, 'scenario': scenario, 'before': before, 'after': after, 'delayed_label': delayed})
            selection_lines.append(f"{seed}{'L' if scenario == 'left_loss' else 'R'} & {before} & {delayed} & {after if after else 'None'} " + r'\\')
    tables['selection-change'] = table('lrrr', selection_lines)
    interface_faults={};interface_healthy={};interface_pairs={};interface_pose={}
    imethods=('fixed','refresh','retarget');inames=('Fixed','Refresh only','Retarget')
    ifull=[r'Load & Wait & Policy & $C$ & $Y$ & Dual & Healthy $Y$ & Wait (s) & Total (s) & Miss (cm) \\',r'\midrule']
    for load in ('native','density150'):
        interface_faults[load]={};interface_healthy[load]={};interface_pairs[load]={}
        for window,wname in [('actual','Actual'),('1000ms','1 s')]:
            faults=[r for r in scored['interface'] if r['load']==load and r['window']==window and r['scenario']!='sham']
            interface_faults[load][window]={};interface_healthy[load][window]={}
            for method,name in zip(imethods,inames):
                group=[r for r in faults if r['method']==method]
                healthy=[r for r in scored['interface'] if r['load']==load and r['window']==window and r['scenario']=='sham' and r['method']==method]
                check(len(group)==40 and len(healthy)==20,'Interface denominator mismatch')
                m=interface_metrics(group);h=interface_metrics(healthy)
                interface_faults[load][window][method]=m;interface_healthy[load][window][method]=h
                aggregated.append({'cohort':'interface','load':load,'window':window,'method':method,**metrics(group)})
                ifull.append(f"{'Native' if load=='native' else 'Heavy'} & {wname} & {name} & {m['C']}/40 & {m['Y']}/40 & {m['ever_dual_regrasp']}/40 & {h['Y']}/20 & {m['wait']:.3f} & {m['total']:.3f} & {m['close_worst_miss_cm']:.2f} "+r'\\')
            interface_pairs[load][window]={f'{a}_vs_{b}':interface_pair(faults,a,b) for a,b in [('retarget','refresh'),('refresh','fixed'),('retarget','fixed')]}
        p=[o for o in interface_observations if o['load']==load and o['scenario']!='sham' and o['method']=='retarget']
        errors=np.concatenate([o['handle_error_mm'] for o in p])
        entry_errors=np.concatenate([o['entry_handle_error_mm'] for o in p])
        interface_pose[load]={'observations':len(p),'handle_error_mm_median':float(np.median(errors)),
            'handle_error_mm_p95':float(np.quantile(errors,.95)),
            'entry_handle_error_mm_median':float(np.median(entry_errors)),
            'entry_handle_error_mm_p95':float(np.quantile(entry_errors,.95)),
            'fixed_worst_target_error_cm_mean':float(np.mean([o['fixed_miss_cm'] for o in p])),
            'retarget_worst_target_error_cm_mean':float(np.mean([o['retarget_miss_cm'] for o in p]))}
    imain=[r'Wait & Policy & Native $Y$ & Heavy $Y$ & Healthy $Y$ & Total (s) \\',r'\midrule']
    for window,wname in [('actual','Actual'),('1000ms','1 s')]:
        for method,name in zip(imethods,inames):
            n=interface_faults['native'][window][method];h=interface_faults['density150'][window][method];healthy=interface_healthy['density150'][window][method]
            imain.append(f"{wname if method=='fixed' else ''} & {name} & {n['Y']}/40 & {h['Y']}/40 & {healthy['Y']}/20 & {h['total']:.3f} "+r'\\')
    tables['interface-retarget']=table('llrrrr',imain)
    tables['interface-full']=table('lllrrrrrrr',ifull)
    wait_lines = [r'Policy & $C$ & $Y$ & Total (s) & Changed\\', r'\midrule']
    wait_metrics = {}
    wait_modes = ('cached', 'refreshed', 'wait_conditioned')
    for w in ('actual', '950ms'):
        wait_metrics[w] = {}
        for method, name in zip(wait_modes, ('Cached', 'Refreshed', 'Wait history')):
            group = [r for r in scored['wait_decision'] if r['window'] == w and r['method'] == method and r['scenario'] != 'sham']
            m = metrics(group)
            m['changed_initial_choice'] = sum(r['candidate'] != r['initial_candidate'] for r in group)
            wait_metrics[w][method] = m
            aggregated.append({'cohort': 'wait_decision', 'load': '', 'window': w, 'method': method, **{k:v for k,v in m.items() if k != 'changed_initial_choice'}})
            if w == '950ms':
                wait_lines.append(f"{name} & {m['C']}/60 & {m['Y']}/60 & {m['total']:.3f} & {m['changed_initial_choice']} " + r'\\')
    tables['wait-decisions'] = table('lrrrr', wait_lines)
    wait_reference = json.loads((HERE / 'reference/wait-decisions.json').read_text())
    wait_pairs = {}
    wait_healthy = {}
    for w in ('actual', '950ms'):
        wait_pairs[w] = {}
        wait_healthy[w] = {}
        for a,b in (('cached','refreshed'), ('cached','wait_conditioned'), ('refreshed','wait_conditioned')):
            pair = paired([r for r in scored['wait_decision'] if r['window'] == w and r['scenario'] != 'sham'], a, b, seed=20261110)
            old = wait_reference['paired_methods'][w][a+'_vs_'+b]['common_protected_cost']
            check(np.allclose([pair['reduction_fraction'], *pair['ci95']], [old['reduction_fraction'], *old['reduction_ci95']], atol=1e-10, rtol=0), 'Waiting paired interval mismatch')
            wait_pairs[w][a+'_vs_'+b] = pair
        for method in wait_modes:
            m, old = wait_metrics[w][method], wait_reference['fault_results'][w][method]
            check(m['C'] == old['C'] and m['Y'] == old['Y'] and m['changed_initial_choice'] == old['changed_initial_choice'], 'Waiting count mismatch')
            close(m['total'], old['mean_total_seconds'], 'Waiting mean time mismatch')
            healthy = [r for r in scored['wait_decision'] if r['window'] == w and r['method'] == method and r['scenario'] == 'sham']
            wait_healthy[w][method] = metrics(healthy)
            check(len(healthy) == 20 and sum(r['Y'] for r in healthy) == wait_reference['healthy_controls'][w][method]['Y'], 'Waiting healthy control mismatch')
    wait_effects = {}
    for method in wait_modes:
        groups = defaultdict(dict)
        for r in scored['wait_decision']:
            if r['method'] == method and r['scenario'] != 'sham':
                groups[r['seed'],r['scenario']][r['window']] = r
        seeds = sorted({k[0] for k in groups})
        delta = np.array([np.mean([int(g['950ms']['Y'])-int(g['actual']['Y']) for k,g in groups.items() if k[0] == s]) for s in seeds])
        draws = np.random.default_rng(20261110).integers(0, len(seeds), (2000,len(seeds)))
        wait_effects[method] = {'Y_difference':float(delta.mean()), 'ci95':np.quantile(delta[draws].mean(1),[.025,.975]).tolist()}
        old = wait_reference['paired_waits'][method]
        check(np.allclose([wait_effects[method]['Y_difference'], *wait_effects[method]['ci95']], [old['Y_difference'], *old['Y_difference_ci95']], atol=1e-12, rtol=0), 'Waiting effect interval mismatch')
    comparisons = {}
    pilot_report=None;history_report=None
    if 'controller_pilot' in scored:
        pilot_report=sampling.summarize(scored['controller_pilot'])
        history_report={'diagnostic_replays':len(history_rows),'new_independent_policy_trials':0,'rows':history_rows}
        check(len(history_rows)==10,'Incomplete selected historical replay diagnostics')
        reference_pilot=json.loads((HERE/'reference/pilot-results.json').read_text())
        for group in ('fault_results','healthy_controls'):
            for key,methods in pilot_report[group].items():
                for method,r in methods.items():
                    old=reference_pilot[group][key][method]
                    for metric in ('episodes','C','Y'):check(r[metric]==old[metric],'Pilot count mismatch: '+metric)
                    check(r['native_Y']==old['native500_Y'],'Pilot native verdict mismatch')
                    for metric in ('total','wait','skill','hold'):close(r[metric],old[metric],'Pilot time mismatch')
        for group in ('executor_effects','waiting_effects'):
            for key,measures in pilot_report[group].items():
                for measure,r in measures.items():
                    old=reference_pilot[group][key]['differences'][measure]
                    check(np.allclose([r['difference'],*r['ci95']],[old['difference'],*old['ci95']],atol=1e-10,rtol=0),'Pilot paired interval mismatch')
        history_expected=json.loads((HERE/'reference/history-native.json').read_text())
        reference_by_key={r['name']:r for r in history_expected['rows']}
        for r in history_rows:
            old=reference_by_key[r['name']]
            for metric in ('primary_gap_ms','native_gap_ms','native_suffix_gap_ms','native_Y','native_history_only','phase_Y','phase_Y_count'):
                check(r[metric]==old[metric],'Historical native diagnostic mismatch: '+metric)
        tables.update(sampling.tables(pilot_report,history_report))
    controller_report=None
    if 'controller' in scored:
        controller_report=controller.summarize(scored['controller'])
        tables.update(controller.tables(controller_report))
        for load in ('native','density150'):
            for window in ('actual','1000ms'):
                for method in controller.METHODS:
                    group=[r for r in scored['controller'] if r['load']==load and r['window']==window and r['method']==method and r['scenario']!='sham']
                    aggregated.append({'cohort':'controller','load':load,'window':window,'method':method,**metrics(group)})
        expected_controller=json.loads((HERE/'reference/controller-results.json').read_text())
        for load in ('native','density150'):
            for window in ('actual','1000ms'):
                for key in ('fault_results','healthy_controls'):
                    for method in controller.METHODS:
                        current=controller_report[key][load][window][method]
                        expected=expected_controller[key][load][window][method]
                        for metric,value in expected.items():
                            if isinstance(value,float):close(current[metric],value,'Controller metric mismatch: '+metric)
                            else:check(current[metric]==value,'Controller metric mismatch: '+metric)
                a=controller_report['paired_methods'][load][window];b=expected_controller['paired_methods'][load][window]
                for metric,value in b.items():
                    if value is None:check(a[metric] is None,'Controller empty paired subset mismatch')
                    else:check(np.allclose(a[metric],value,atol=1e-10,rtol=0),'Controller paired interval mismatch: '+metric)
            for method in controller.METHODS:
                a=controller_report['paired_waits'][load][method];b=expected_controller['paired_waits'][load][method]
                for metric,value in b.items():
                    if isinstance(value,dict):check(a[metric]==value,'Controller retry mismatch')
                    else:check(np.allclose(a[metric],value,atol=1e-10,rtol=0),'Controller waiting interval mismatch: '+metric)
            a=controller_report['waiting_method_interaction'][load]
            b=expected_controller['waiting_method_interaction'][load]
            for metric,value in b.items():
                check(np.allclose(a[metric],value,atol=1e-10,rtol=0),'Controller interaction interval mismatch: '+metric)
    for w in ('actual', '1000ms', '2000ms'):
        comparisons['handover-rule-global-' + w] = paired([r for r in scored['handover'] if r['window'] == w and r['scenario'] != 'nominal'], 'rule', 'R3')
    scope_faults = [r for r in scored['scope'] if r['scenario'] != 'sham']
    state_groups = defaultdict(list)
    for r in scope_faults:
        state_groups[(r['seed'], r['scenario'])].append(r)
    common = {k for k, g in state_groups.items() if all(r['Y'] for r in g)}
    comparisons['scope-rule-mlp-common4'] = paired(scope_faults, 'protected', 'gated_direct', common)
    comparisons['sft-fixed60'] = paired(scored['supervision'], 'ce', 'fixed60')
    for load in ('native', 'density150'):
        for w in ('actual', '1000ms'):
            comparisons[f'load-rule-global-{load}-{w}'] = paired([r for r in scored['load'] if r['load'] == load and r['window'] == w and r['scenario'] != 'sham'], 'protected', 'global')
    reference = json.loads((HERE / 'reference/frozen-results.json').read_text())
    ir=json.loads((HERE/'reference/interface-results.json').read_text())
    for load in ('native','density150'):
        for window in ('actual','1000ms'):
            for method in imethods:
                for current,expected in [(interface_faults[load][window][method],ir['fault_results'][load][window][method]),
                                         (interface_healthy[load][window][method],ir['healthy_controls'][load][window][method])]:
                    for k in ('episodes','C','Y','ever_dual_regrasp','close_dual_grasp','setdown','pretarget_support_violations','completed_with_prior_violation'):check(current[k]==expected[k],'Interface outcome mismatch: '+k)
                    for k in ('total','wait','skill','hold','path_m','close_worst_miss_cm'):close(current[k],expected[k],'Interface metric mismatch: '+k)
            for key,current in interface_pairs[load][window].items():
                expected=ir['paired_effects'][load][window][key]
                for k in ('Y_difference','Y_difference_ci95','all_attempt_time_difference'):
                    check(np.allclose(current[k],expected[k],atol=1e-10,rtol=0),'Interface paired effect mismatch: '+k)
        for k in interface_pose[load]:close(interface_pose[load][k],ir['pose_quality'][load][k],'Interface pose diagnostic mismatch: '+k)
    regrasp_reference = json.loads((HERE / 'reference/regrasp-attribution.json').read_text())
    regrasp = {}
    for load in ('native', 'density150'):
        for window in ('actual', '1000ms'):
            rows = [r for r in scored['load'] if r.get('load') == load and r.get('window') == window
                    and 'ever_dual_regrasp' in r]
            m = {'episodes': len(rows), 'setdown_verified': sum(r['table'] for r in rows),
                 'ever_dual_regrasp': sum(r['ever_dual_regrasp'] for r in rows),
                 'settle_tilt_median_deg': float(np.median([r['settle_tilt_deg'] for r in rows])),
                 'close_miss_range_cm': [100 * fn(r['close_miss_m'] for r in rows) for fn in (min, max)]}
            old = regrasp_reference['groups'][f'{load}/{window}/unilateral']
            check(m['setdown_verified'] == old['setdown_verified'] and m['ever_dual_regrasp'] == old['ever_both_grasps_after_reclose'], 'Regrasp count mismatch')
            close(m['settle_tilt_median_deg'], old['global_table_settle_tilt_deg']['median'], 'Regrasp pose mismatch')
            check(np.allclose(m['close_miss_range_cm'], [100 * old['global_close_max_handle_eef_error_m'][k] for k in ('minimum', 'maximum')], atol=1e-10, rtol=0), 'Regrasp error mismatch')
            regrasp[f'{load}/{window}'] = m
    for w in ('actual', '1000ms', '2000ms'):
        a = comparisons['handover-rule-global-' + w]
        b = reference['handover']['baseline_paired_fault_comparisons'][w]['rule_vs_R3']['total_seconds']
        check(np.allclose([a['reduction_fraction'], *a['ci95']], [b['reduction_fraction'], *b['reduction_ci95']], atol=1e-10, rtol=0), 'Handover paired interval mismatch')
    for key, b in [('scope-rule-mlp-common4', reference['scope']['all_four_common_safe']['paired']['protected_vs_gated_direct']['total_post_fault_seconds']),
                   ('sft-fixed60', reference['supervision']['paired']['ce_vs_fixed60']['total_post_fault_seconds'])]:
        a = comparisons[key]
        check(np.allclose([a['reduction_fraction'], *a['ci95']], [b['reduction_fraction'], *b['ci95']], atol=1e-10, rtol=0), 'Paired interval mismatch: ' + key)
    for load in ('native', 'density150'):
        for w in ('actual', '1000ms'):
            a = comparisons[f'load-rule-global-{load}-{w}']
            b = reference['load']['paired_methods'][load][w]['protected_vs_global']['common_protected_cost']
            check(np.allclose([a['reduction_fraction'], *a['ci95']], [b['reduction_fraction'], *b['reduction_ci95']], atol=1e-10, rtol=0), 'Load paired interval mismatch')
    prospective_reports={}
    for cohort in ('wait_hold','geometry'):
        if cohort not in scored:continue
        report=prospective.summarize(scored[cohort]);prospective_reports[cohort]=report
        tables.update(prospective.tables(report,cohort=='geometry'))
        expected=json.loads((HERE/'reference'/(cohort+'-results.json')).read_text())
        for key in ('episodes','independent_seeds','native_samples','fault_results','healthy_controls','paired_effects'):
            check(report[key]==expected[key],'Prospective report mismatch: '+cohort+'/'+key)
        for hold,windows in report['fault_results'].items():
            for window,methods in windows.items():
                for method,values in methods.items():
                    aggregated.append({'cohort':cohort,'hold_mode':hold,'wait_steps':window,'method':method,**values})
    holding_reports={};handover_native=[];budget_curves={}
    for cohort,name in [('holding_components','holding-components'),('independent_visual','independent-visual')]:
        if cohort not in scored:continue
        report=holding.summarize(scored[cohort]);holding_reports[cohort]=report
        expected=json.loads((HERE/'reference'/(cohort+'-results.json')).read_text())
        for key in ('episodes','independent_seeds','method','native_samples','fault_results','healthy_controls','paired_effects'):
            check(report[key]==expected[key],'Holding report mismatch: '+cohort+'/'+key)
        tables[name]=holding.table(report)
        for hold,windows in report['fault_results'].items():
            for step,values in windows.items():
                aggregated.append({'cohort':cohort,'hold_mode':hold,'wait_steps':step,'method':report['method'],**values})
    if holding is not None:
        expected=json.loads((HERE/'reference/handover-native.json').read_text())
        for original in expected['reports']:
            current=holding.native_handover(HERE/'data/handover-native'/(original['name']+'.npz'))
            for key,value in current.items():check(value==original[key],'Native handover mismatch: '+key)
            handover_native.append({'name':original['name'],**current})
        check(len(handover_native)==7,'Incomplete native handover replay set')
        for hold in ('registered','measured_pose'):
            for step in (1,20):
                group=[r for r in scored['wait_hold'] if r['scenario']!='sham' and r['method']=='closed_loop'
                       and r['hold_mode']==hold and r['wait_steps']==step]
                times=sorted({0.,45.,*[r['total'] for r in group]})
                budget_curves[f'{hold}/{step}']={'episodes':len(group),'seconds':times,
                    'protected_within_budget':[sum(r['native_Y'] and r['total']<=t for r in group)/len(group) for t in times]}
    paired_reports={};prior_mechanism=None
    if paired_audit is not None:
        prior_mechanism=paired_audit.prior_summary({'mechanism':scored['holding_components'],'independent':scored['independent_visual']})
        expected=json.loads((HERE/'reference/prior-mechanism.json').read_text())
        check(prior_mechanism==expected,'Prior mechanism/transition reanalysis mismatch')
        for cohort,name in [('paired_holding','paired-holding'),('paired_placement','paired-placement')]:
            if cohort not in scored:continue
            report=paired_audit.summarize(scored[cohort]);report.update(paired_audit.check_pairing(paired_states[cohort]))
            if cohort=='paired_holding':
                e=report['effects']
                report['placement_trigger']=bool(e['closed_loop']['position_main_wait']['native_Y']['ci95'][0]>0 and
                    e['executor_position_wait']['native_Y']['ci95'][0]>0 and e['visual_servo']['position_main_wait']['native_Y']['ci95'][0]<=0)
                check(report['matched_prehold_groups']==60 and report['matched_posthold_method_pairs']==480,'Paired physical states mismatch')
                tables['paired-holding-main']=paired_audit.compact_table(report)
            else:
                baseline=[r for r in scored['paired_holding'] if r['method']=='closed_loop' and r['hold_mode'] in ('registered','measured_position')]
                report['matched_baseline_episodes']=len(baseline)
                report['effects']=paired_audit.effects(scored[cohort]+baseline)
                baseline_states=[r for r in paired_states['paired_holding'] if r['method']=='closed_loop' and r['hold_mode'] in ('registered','measured_position')]
                paircheck=paired_audit.check_pairing(paired_states[cohort]+baseline_states)
                check(paircheck['matched_prehold_groups']==60 and paircheck['matched_posthold_method_pairs']==240,'Placement physical pairing mismatch')
                report['baseline_pairing']=paircheck
            expected=json.loads((HERE/'reference'/(cohort+'-results.json')).read_text())
            for key in ('episodes','independent_seeds','native_samples','conditions','effects','transitions','matched_prehold_groups','matched_posthold_method_pairs'):
                check(report[key]==expected[key],'Paired study report mismatch: '+cohort+'/'+key)
            if cohort=='paired_holding':check(report['placement_trigger']==expected['placement_trigger'],'Placement gate mismatch')
            tables[name]=paired_audit.table(report);paired_reports[cohort]=report
            for method,holds in report['conditions'].items():
                for hold,windows in holds.items():
                    for step,g in windows.items():
                        aggregated.append({'cohort':cohort,'method':method,'hold_mode':hold,'wait_steps':step,
                            **g['faults'],'healthy_Y':g['healthy']['native_Y'],'healthy_episodes':g['healthy']['episodes']})
    handover_holding_report=None;control_reference_diagnostics=None
    if handover_holding is not None:
        handover_holding_report=handover_holding.summarize(scored['handover_holding'])
        handover_holding_report.update(handover_holding.check_pairing(handover_holding_states))
        handover_holding_report['timing_sensitivity']=handover_holding.timing_sensitivity(scored['handover_holding'],handover_holding_states)
        expected=json.loads((HERE/'reference/handover-holding-results.json').read_text())
        check(handover_holding_report==expected,'Handover holding report mismatch')
        check(handover_holding_report['matched_prehold_groups']==3*handover_holding_report['initialized_seeds'] and
            handover_holding_report['posthold_path_pairs']==18*handover_holding_report['initialized_seeds'],
            'Handover physical pairing mismatch')
        tables['handover-holding']=handover_holding.table(handover_holding_report)
        tables['handover-holding-main']=handover_holding.table(handover_holding_report,True)
        control_reference_diagnostics=handover_holding.reference_summary(scored['holding_components'])
        expected=json.loads((HERE/'reference/control-reference-diagnostics.json').read_text())
        check(control_reference_diagnostics==expected,'Existing reference diagnostics mismatch')
        for method,holds in handover_holding_report['conditions'].items():
            for hold,waits in holds.items():
                for step,c in waits.items():
                    aggregated.append({'cohort':'handover_holding','method':method,'hold_mode':hold,
                        'wait_steps':step,**c['faults'],'healthy_Y500':c['healthy']['Y500'],
                        'healthy_episodes':c['healthy']['episodes']})
    execution_error_report=None;lifting_smooth_report=None
    if execution_error is not None:
        execution_error_report=execution_error.summarize(execution_error_rows)
        expected=json.loads((HERE/'reference/execution-error-results.json').read_text())
        check(execution_error_report==expected,'First-close vector/stage analysis mismatch')
        tables.update(execution_error.tables(execution_error_report))
        (out/'execution-error-cases.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in execution_error_rows))
    if lifting_smooth is not None:
        lifting_smooth_report=lifting_smooth.summarize(scored['lifting_smooth'])
        lifting_smooth_report.update(lifting_smooth.pairing(lifting_smooth_states),pose_fits_recomputed=lifting_smooth_fits)
        expected=json.loads((HERE/'reference/lifting-smooth-results.json').read_text())
        check(lifting_smooth_report==expected,'Smooth lifting report mismatch')
        tables['lifting-smooth']=lifting_smooth.table(lifting_smooth_report)
    public_policy_report=None;public_policy_input_tensors=public_policy_rgb_histories=public_policy_rgb_frames=0
    if public_policy is not None:
        public_policy_report=public_policy.summarize(scored['public_transport'])
        public_policy_report.update(public_policy.check_pairing(public_policy_states))
        interruptions=json.loads((HERE/'data/public-policy-prefix-interruptions-all.json').read_text())['attempts']
        check(len(interruptions)==60 and all(r['C'] is None and not r['holding_was_applied'] for r in interruptions),
              'Public pre-intervention diagnostic accounting mismatch')
        public_policy_report['preintervention_interrupted_checks']=len(interruptions)
        expected=json.loads((HERE/'reference/public-policy-results.json').read_text())
        check(public_policy_report==expected,'Native public-policy report mismatch')
        tables.update(public_policy.tables(public_policy_report))
        baselines={(r['method'],r['seed']):r for r in public_policy_states if r['hold_mode']=='unpaused'}
        for info in json.loads((HERE/'figures/public-policy/inputs.json').read_text()):
            method=info['method'];boundary=str(info['boundary'])
            seeds=json.loads((HERE/'data/public-policy-protocol.json').read_text())['formal_seeds']
            with np.load(HERE/'figures/public-policy'/info['file'],allow_pickle=False) as arrays:
                for i,seed in enumerate(seeds):
                    hashes=baselines[method,seed]['prefix_observation_sha256'][boundary]
                    for key,digest in hashes.items():
                        a=np.ascontiguousarray(arrays[key][i]);check(hashlib.sha256(a.tobytes()).hexdigest()==digest,'Public policy input history mismatch')
                        public_policy_input_tensors+=1
                        if key.endswith('_image'):
                            public_policy_rgb_histories+=1;public_policy_rgb_frames+=a.shape[0]
        for c in public_policy_report['conditions']:aggregated.append({'cohort':'public_transport',**c})
    for name, content in tables.items():
        (out / 'tables' / f'{name}.tex').write_text(content)
        check(content.split() == (HERE / 'reference/tables' / f'{name}.tex').read_text().split(), 'Frozen table mismatch: ' + name)
    write_csv(out / 'main-results.csv', aggregated)
    result = {'episodes_recomputed': {k: len(v) for k, v in scored.items()},
              'total_executions': sum(map(len, scored.values()))-(handover_holding_report['initialization_failures'] if handover_holding_report else 0),
              'total_recorded_attempts':sum(map(len,scored.values())), 'tables_match': list(tables),
              'frozen_mlp_decisions_recomputed': mlp_count, 'paired_costs': comparisons,
              'waiting_policy_decisions_recomputed': waiting_count, 'waiting_policy_metrics': wait_metrics,
              'waiting_policy_paired_costs': wait_pairs,
              'waiting_policy_healthy_controls': wait_healthy, 'waiting_protection_effects': wait_effects,
              'interface_pose_fits_recomputed':len(interface_observations),'interface_fault_results':interface_faults,
              'interface_rgb_examples_recomputed':image_example_count,
              'interface_healthy_controls':interface_healthy,'interface_paired_effects':interface_pairs,'interface_pose_quality':interface_pose,
              'regrasp_diagnostic': regrasp,
              'choice_changes': choice_changes, 'handover_labels': dict(Counter(r['label'] for r in scored['handover'])),
              'handover_boundary_case_role_loss_samples': next(r['role_loss_samples'] for r in scored['handover'] if (r['seed'], r['scenario'], r['method'], r['window']) == (40017, 'object_rotation', 'proprio', '2000ms')),
              'scope_common_protected_states': len(common)}
    result['evaluation_layers'] = layer_summary
    result['controller_report']=controller_report
    result['controller_pose_fits_recomputed']=controller_fits
    result['pilot_report']=pilot_report
    result['pilot_pose_fits_recomputed']=pilot_fits
    result['history_native_diagnostics']=history_report
    result['prospective_reports']=prospective_reports
    result['prospective_pose_fits_recomputed']=prospective_fits
    result['prospective_rgb_examples_recomputed']=prospective_rgb_examples
    result['holding_reports']=holding_reports
    result['holding_pose_fits_recomputed']=holding_fits
    result['landmark_estimates_recomputed']=landmark_fits
    result['landmark_rgb_examples_recomputed']=landmark_rgb_examples
    result['handover_native_diagnostics']=handover_native
    result['protected_budget_curves']=budget_curves
    result['paired_reports']=paired_reports
    result['prior_mechanism']=prior_mechanism
    result['paired_pose_fits_recomputed']=paired_fits
    result['paired_landmark_estimates_recomputed']=paired_landmark_fits
    result['handover_holding_report']=handover_holding_report
    result['control_reference_diagnostics']=control_reference_diagnostics
    result['execution_error_report']=execution_error_report
    result['lifting_smooth_report']=lifting_smooth_report
    result['lifting_smooth_pose_fits_recomputed']=lifting_smooth_fits
    result['public_policy_report']=public_policy_report
    result['public_policy_input_tensors_recomputed']=public_policy_input_tensors
    result['public_policy_rgb_histories_checked']=public_policy_rgb_histories
    result['public_policy_rgb_frames_checked']=public_policy_rgb_frames
    result['holding_native_history_cases']=[{k:r[k] for k in ('seed','scenario','hold_mode','wait_steps','native_gap_ms','native_suffix_gap_ms')}
        for r in scored.get('holding_components',[]) if r['scenario']!='sham' and r['native_history_only']]
    write_csv(out / 'evaluation-layer-cases.csv', layers)
    (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    # Per-execution output permits independent inspection of all denominators.
    with (out / 'scored-episodes.jsonl').open('w') as f:
        for cohort, rows in scored.items():
            for row in rows:
                f.write(json.dumps({'cohort': cohort, **row}) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
