"""Finite-horizon supervision of the first future keeper-grasp loss."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


SUPPORT_TIME_CODES = 'ABCDEFG'
SUPPORT_TIME_INSTRUCTION = """Predict WHEN the keeper gripper will FIRST lose its handle during the remaining specified recovery.
Two Panda arms lift a handled pot. Robot 0 is left; robot 1 is right. Control rate is 20 Hz.
Images 1-4 show the previous observation and images 5-8 the current observation.
At each time the cameras are external front, external agent, left wrist, right wrist.
The approach/close prefix has ALREADY executed. Use only these images, past measured robot history,
and the candidate's planned hold, relevel duration and final verification. Predict a FUTURE event,
not whether the keeper is grasping now. No true future contacts or task outcomes are provided.
A: no keeper-grasp loss observed through the end of the remaining execution and verification.
   This is survival only over that finite executed window, not indefinite survival or task completion.
B: first loss during the planned post-prefix waiting hold.
C: first loss in relevel progress (0%, 25%].
D: first loss in relevel progress (25%, 50%].
E: first loss in relevel progress (50%, 75%].
F: first loss in relevel progress (75%, 100%].
G: first loss after relevel, during settling or final verification.
Progress is the number of executed relevel control steps divided by the candidate's planned relevel steps.
Answer exactly one letter A, B, C, D, E, F or G."""


def support_time_target(trial):
    """Read supervision from a real rollout; never use it in a model prompt."""
    origin = trial['stage_capture']['h2']['state']['time']
    keeper = 1 if trial['candidate'] == 'R_left' else 0
    future = [f for f in trial['trace'] if f['time'] > origin + 1e-8]
    if not future:
        raise ValueError('a future executed observation window is required')
    relevel = [f for f in future if f['stage'] == 'local_relevel']
    planned = trial['local_steps']['local_relevel_steps']
    if len(relevel) != planned:
        raise ValueError('this diagnostic requires the complete executed relevel candidate')
    first = next((f for f in future if not f['evaluator']['holds'][keeper]), None)
    code, step = 'A', None
    if first is not None:
        if first['stage'] == 'duration_budget_wait':
            code = 'B'
        elif first['stage'] == 'local_relevel':
            step = next(j for j, f in enumerate(relevel, 1) if f is first)
            code = 'CDEF'[(4 * step - 1) // planned]
        elif first['stage'] in ('local_relevel_settle', 'final_stable_hold'):
            code = 'G'
        else:
            raise ValueError('unsupported future loss stage: ' + first['stage'])
    return {'label': code, 'event_observed': first is not None,
        'first_loss_seconds_after_observation': None if first is None else first['time'] - origin,
        'first_loss_stage': None if first is None else first['stage'],
        'first_loss_relevel_step': step,
        'first_loss_relevel_progress': None if step is None else step / planned,
        'observed_horizon_seconds': future[-1]['time'] - origin,
        'right_censored': first is None}


def attach_targets(collection, destination, wait_steps=66):
    """Copy the fixed split/order, adding only training/evaluation targets."""
    collection, destination = Path(collection), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    trials = {}
    for w in (0, wait_steps):
        for line in (collection / f'trials-wait{w}.jsonl').read_text().splitlines():
            r = json.loads(line)
            key = r['seed'], r['scenario'], w, r['local_steps']['local_relevel_steps']
            trials[key] = r
    report = {'source_collection': str(collection.resolve()), 'new_physical_episodes': 0,
        'codes': SUPPORT_TIME_CODES, 'instruction': SUPPORT_TIME_INSTRUCTION, 'splits': {}}
    for split in ('train', 'development'):
        rows = [json.loads(s) for s in (collection / 'dataset' / (split + '.jsonl')).read_text().splitlines()]
        for row in rows:
            key = row['seed'], row['scenario'], row['post_prefix_wait_steps'], row['plan']['local_relevel_steps']
            row['support_time'] = support_time_target(trials[key])
        (destination / (split + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rows))
        report['splits'][split] = {'examples': len(rows),
            'classes': dict(Counter(r['support_time']['label'] for r in rows)),
            'observed_events': sum(r['support_time']['event_observed'] for r in rows),
            'finite_censored_windows': sum(r['support_time']['right_censored'] for r in rows),
            'seeds': sorted({r['seed'] for r in rows})}
    (destination / 'support-time-targets.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def support_time_metrics(records):
    events = [r for r in records if r['label'] != 'A']
    censored = [r for r in records if r['label'] == 'A']
    return {'examples': len(records), 'class_counts': dict(Counter(r['label'] for r in records)),
        'time_bin_correct': sum(r['prediction'] == r['label'] for r in records),
        'event_examples': len(events),
        'event_bin_correct': sum(r['prediction'] == r['label'] for r in events),
        'events_predicted_no_loss': sum(r['prediction'] == 'A' for r in events),
        'censored_examples': len(censored),
        'censored_predicted_loss': sum(r['prediction'] != 'A' for r in censored),
        'confusion': {c: dict(Counter(r['prediction'] for r in records if r['label'] == c)) for c in SUPPORT_TIME_CODES},
        'scope': 'Native auxiliary-token classification; not an admission rule or infinite survival estimate.'}
