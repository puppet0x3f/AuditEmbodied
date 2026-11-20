"""Post-execution attribution only; never changes the frozen recovery policy."""
import json
from pathlib import Path

import numpy as np

from auditembodied.admissible_mlp import local_precondition
from auditembodied.agentchord_style import AgentChordStyle
from auditembodied.l4_selector import safe
from auditembodied.p0 import save_json


root = Path(__file__).resolve().parent
protocol = json.loads((root / 'protocol.json').read_text())
rows = [json.loads(line) for line in (root / 'trials.jsonl').read_text().splitlines()]
assert len(rows) == protocol['expected_episodes']
faults = [r for r in rows if r['scenario'] != 'sham']
graph = AgentChordStyle(protocol['graph_spec'])
groups = {}
for row in faults:
    groups.setdefault((row['seed'], row['scenario']), {})[row['method']] = row
counts = {}
for method in ('protected', 'gated_direct'):
    starts = [local_precondition(r) for r in faults if r['method'] == method]
    starts = [s for s in starts if s is not None]
    counts[method] = {'local_choices': len(starts),
        'missing_keeper_at_fault': sum(not s['keeper_present_at_start'] for s in starts),
        'keeper_interrupted_after_valid_start': sum(s['keeper_interrupted_after_valid_start'] for s in starts)}
locals_, branches, misses = [], [], []
for row in faults:
    if row['method'] != 'agentchord_style':
        continue
    for event in row.get('skill_events', []):
        candidate = event['candidate']
        if candidate not in ('R_left', 'R_right'):
            continue
        keeper = 1 if candidate == 'R_left' else 0
        frames = [f for f in row['trace'] if f['phase'] == 'recovery' and f.get('skill') == candidate
                  and event['start_time'] < f['time'] <= event['end_time']]
        lost = [f for f in frames if not f['evaluator']['holds'][keeper]]
        item = {'seed': row['seed'], 'scenario': row['scenario'], 'candidate': candidate,
            'keeper_present_at_local_start': event['start_evaluator']['holds'][keeper],
            'local_control_steps': len(frames), 'keeper_loss_steps': len(lost),
            'monitor_triggered': event.get('monitor_index') is not None,
            'terminal_complete': row.get('task_complete', False), 'safe_completion': safe(row)}
        if lost:
            first = lost[0]
            port = {k: first['robot'][k] for k in ('positions', 'fingers')}
            port['reference_positions'] = event['input']['reference_positions']
            width = float(np.abs(first['robot']['fingers'][keeper]).mean())
            name = 'left_keeper_missing' if candidate == 'R_left' else 'right_keeper_missing'
            item.update(first_loss_stage=first['stage'],
                first_loss_seconds_after_fault=first['time']-row['prefix_state']['time'],
                encoder_mean_at_first_loss=width,
                monitor_detects_first_true_loss=graph.monitor(name, port))
            misses.append(item)
        locals_.append(item)
    if len(row.get('graph', {}).get('executed_skills', [])) > 1:
        key = (row['seed'], row['scenario'])
        local_frames = [f for f in row['trace'] if f['phase'] == 'recovery' and
                        f['stage'] != 'inference_wait' and f.get('skill') in ('R_left', 'R_right')]
        direct_frames = [f for f in groups[key]['gated_direct']['trace'] if f['phase'] == 'recovery' and
                         f['stage'] != 'inference_wait']
        identical_local_prefix = ([{k: f[k] for k in ('time', 'stage', 'phase', 'robot', 'evaluator')}
            for f in local_frames] == direct_frames)
        branches.append({'seed': key[0], 'scenario': key[1], 'graph': row['graph'],
            'local_prefix_identical_to_executed_gated_mlp': identical_local_prefix,
            'comparators': {m: {k: r.get(k) for k in ('candidate', 'success', 'task_complete',
                'protection_violation', 'total_post_fault_seconds', 'recovery_path_m')} for m, r in groups[key].items()}})
counts['agentchord_style'] = {'local_attempts': len(locals_),
    'missing_keeper_at_local_start': sum(not x['keeper_present_at_local_start'] for x in locals_),
    'keeper_interrupted_after_valid_start': sum(x['keeper_present_at_local_start'] and x['keeper_loss_steps'] > 0 for x in locals_),
    'first_true_loss_detected_by_encoder_monitor': sum(x['monitor_detects_first_true_loss'] for x in misses)}
save_json(root / 'attribution.json', {'scope': 'offline evaluation from executed traces; no policy or test adaptation',
    'local_starts': counts, 'graph_local_attempts': locals_, 'graph_keeper_loss_cases': misses,
    'graph_multi_skill_cases': branches})
print(json.dumps({'local_starts': counts, 'multi_skill_cases': len(branches)}))
