"""Reproduce P0 attribution from immutable physical traces; no model updates."""
import json
from pathlib import Path
from auditembodied.p0 import paired_statistics

root = Path(__file__).resolve().parent
read = lambda phase: [json.loads(s) for s in (root / f'{phase}-trials.jsonl').read_text().splitlines()]
core, wait = read('core'), read('intervention')
safe = lambda r: r['success'] and not r['protection_violation']
failures = []
for r in core:
    if r['scenario'] == 'sham' or r['method'] != 'direct' or safe(r):
        continue
    local = r['candidate'] in ('R_left', 'R_right')
    keeper = 1 if r['candidate'] == 'R_left' else 0
    initial = r['post_fault']['holds'][keeper] if local else None
    losses = [f for f in r['trace'] if local and f['phase'] == 'recovery' and not f['evaluator']['holds'][keeper]]
    failures.append({'seed': r['seed'], 'scenario': r['scenario'], 'candidate': r['candidate'],
        'keeper_present_at_start': initial, 'keeper_lost_after_valid_start': bool(initial and losses),
        'terminal_complete': r['task_complete'], 'first_loss_stage': losses[0]['stage'] if losses else None})
groups = {}
for r in wait:
    groups.setdefault((r['seed'], r['scenario'], r['planned_duration_steps']), {})[r['planned_wait_steps']] = r
changes = []
for (seed, scenario, n), g in groups.items():
    if safe(g[0]) and not safe(g[19]):
        r = g[19]
        keeper = 1 if r['candidate'] == 'R_left' else 0
        losses = [f for f in r['trace'] if f['phase'] == 'recovery' and not f['evaluator']['holds'][keeper]]
        changes.append({'seed': seed, 'scenario': scenario, 'duration': n,
            'first_loss_stage': losses[0]['stage'] if losses else None,
            'wait_keeper_loss_steps': sum(f['stage'] == 'inference_wait' for f in losses),
            'keeper_loss_steps': len(losses), 'total_seconds_without_wait': g[0]['total_post_fault_seconds'],
            'total_seconds_with_wait': r['total_post_fault_seconds']})
paired = [{**r, 'scenario': r['scenario'] + '/' + str(r['planned_duration_steps']),
           'method': 'wait0' if r['planned_wait_steps'] == 0 else 'wait19'} for r in wait]
result = {'direct_failure_counts': {'missing_keeper_precondition': sum(r['keeper_present_at_start'] is False for r in failures),
          'keeper_interrupted_after_valid_start': sum(r['keeper_lost_after_valid_start'] for r in failures)},
          'direct_failures': failures, 'wait_safety_losses': changes,
          'wait_loss_distinct_states': len({(r['seed'], r['scenario']) for r in changes}),
          'wait_paired_statistics': paired_statistics(paired, 'wait19', 'wait0'),
          'scope': 'Post-run attribution only; no training, threshold change or rerun.'}
(root / 'attribution.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result['direct_failure_counts']))
