"""Compare the final auxiliary-time fit with the fixed existing stage_late fit."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from auditembodied.vlm_duration_lora import start_key, timing_contrasts
from auditembodied.vlm_duration_support_time import support_time_metrics
from auditembodied.vlm_duration_temporal import input_spec

root = Path(__file__).resolve().parent
plan = json.loads((root / 'experiment-plan.json').read_text())
baseline = Path(plan['baseline_dir'])
new = root / 'supporttime512'
tau = plan['threshold']
read = lambda path: [json.loads(s) for s in Path(path).read_text().splitlines()]


def metrics(rows):
    accepted = lambda r: r['class_probabilities'][0] >= tau
    groups = defaultdict(list)
    for r in rows:
        groups[r['seed'], r['scenario'], r['wait_steps']].append(r)
    families = []
    for (seed, scenario, wait), group in groups.items():
        assert {r['plan']['local_relevel_steps'] for r in group} == {30, 40, 60}
        admitted = sorted([r for r in group if accepted(r)], key=lambda r: r['plan']['local_relevel_steps'])
        families.append({'seed': seed, 'scenario': scenario, 'W_steps': wait,
            'has_safe_candidate': any(r['label'] == 'A' for r in group),
            'safe_covered': any(r['label'] == 'A' for r in admitted),
            'all_rejected': not admitted,
            'selected_steps': admitted[0]['plan']['local_relevel_steps'] if admitted else None,
            'selected_label': admitted[0]['label'] if admitted else None,
            'cheapest_admitted_safe': bool(admitted and admitted[0]['label'] == 'A')})
    no_safe = [f for f in families if not f['has_safe_candidate']]
    contrasts = timing_contrasts([{**r, 'prediction': r['conditional_prediction']} for r in rows])
    pairs = {}
    for r in rows:
        pairs.setdefault((r['seed'], r['scenario'], r['wait_steps']), {})[r['plan']['local_relevel_steps']] = r
    changed = [g for g in pairs.values() if (g[40]['label'] == 'A') != (g[60]['label'] == 'A')]
    contrasts['changed_pair_both_admission_decisions_correct'] = sum(
        all(accepted(g[n]) == (g[n]['label'] == 'A') for n in (40, 60)) for g in changed)
    return {'examples': len(rows), 'class_counts': dict(Counter(r['label'] for r in rows)),
        'safe_candidates': sum(r['label'] == 'A' for r in rows),
        'safe_admitted': sum(r['label'] == 'A' and accepted(r) for r in rows),
        'false_admissions': sum(r['label'] != 'A' and accepted(r) for r in rows),
        'false_by_class': dict(Counter(r['label'] for r in rows if r['label'] != 'A' and accepted(r))),
        'conditional_class_correct': sum(r['conditional_prediction'] == r['label'] for r in rows),
        'conditional_confusion': {c: dict(Counter(r['conditional_prediction'] for r in rows if r['label'] == c)) for c in 'ABCD'},
        'no_safe_families': len(no_safe), 'no_safe_families_rejected': sum(f['all_rejected'] for f in no_safe),
        'recoverable_families': sum(f['has_safe_candidate'] for f in families),
        'safe_covered_families': sum(f['safe_covered'] for f in families),
        'cheapest_admitted_safe_families': sum(f['cheapest_admitted_safe'] for f in families),
        'family_cases': families, 'timing_40_60': contrasts}


report = {'threshold': tau, 'W_steps': 66, 'W_seconds': 3.3, 'steps': 512,
    'new_physical_episodes': 0, 'variants': {}, 'checks': {},
    'scope': plan['scope'],
    'compute_budget': plan['compute_budget'],
    'support_time_targets': json.loads((root / 'dataset/support-time-targets.json').read_text())}
data = {s: read(root / 'dataset' / (s + '.jsonl')) for s in ('train', 'development')}
report['existing_wait_intervention_coverage'] = {}
for split, rows in data.items():
    wait_pairs = {}
    for row in rows:
        wait_pairs.setdefault((row['seed'], row['scenario'], row['plan']['local_relevel_steps']), {})[row['post_prefix_wait_steps']] = row
    report['existing_wait_intervention_coverage'][split] = {
        'matched_wait_pairs': len(wait_pairs),
        'keeper_loss_presence_flips': sum(g[0]['support_time']['event_observed'] != g[66]['support_time']['event_observed'] for g in wait_pairs.values()),
        'time_bin_flips': sum(g[0]['support_time']['label'] != g[66]['support_time']['label'] for g in wait_pairs.values()),
        'joint_outcome_class_flips': sum(g[0]['label'] != g[66]['label'] for g in wait_pairs.values()),
        'scope': 'Descriptive data coverage only; no sampling, loss or threshold change based on these counts.'}
    original = read(Path(plan['source_collection']) / 'dataset' / (split + '.jsonl'))
    for a, b in zip(original, rows):
        assert a == {k: v for k, v in b.items() if k != 'support_time'}
        for task in ('start', 'outcome'):
            assert input_spec(a, task, 'stage_late') == input_spec(b, task, 'stage_late')
report['checks']['original_primary_inputs_and_labels_unchanged'] = True
scores = {}
flat = []
for name, directory in (('baseline', baseline), ('support_time', new)):
    result = json.loads((directory / 'result.json').read_text())
    protocol = json.loads((directory / 'protocol.json').read_text())
    training = read(directory / 'training.jsonl')
    assert result['completed_steps'] == len(training) == 512
    assert result['frozen_start_unchanged'] and result['data_unchanged']
    assert protocol['final_checkpoint_evaluation_only'] and not list(directory.glob('*before*'))
    assert protocol['init_adapter'] == plan['initial_outcome_adapter']
    assert protocol['fixed_admission_threshold'] == tau and protocol['learning_rate'] == plan['learning_rate']
    assert protocol['train_seeds'] == list(range(40, 46)) and protocol['development_seeds'] == [46, 47]
    assert all(0 < r['gradient_norm_before_clip'] < math.inf and math.isfinite(r['loss']) for r in training)
    if name == 'support_time':
        assert all(math.isfinite(r['support_time_loss']) for r in training)
        assert protocol['support_time_loss_weight'] == .5 and not protocol['support_time_inference']
    scores[name] = {}
    variant = {'training_seconds': result['training_seconds'],
        'peak_allocated_gpu_bytes': result['peak_allocated_gpu_bytes'],
        'primary_presentations': sum(len(r['example_indices']) for r in training),
        'auxiliary_time_presentations': sum(len(r.get('support_time_labels', [])) for r in training), 'cohorts': {}}
    starts = json.loads((directory / 'frozen-start-scores.json').read_text())
    for split, examples in data.items():
        rows = read(directory / ('train-after-predictions.jsonl' if split == 'train' else 'after-predictions.jsonl'))
        assert len(rows) == len(examples)
        for a, b in zip(rows, examples):
            assert all(a[k] == b[k] for k in ('seed', 'scenario', 'image', 'robot', 'plan', 'label', 'wait_steps'))
        scores[name][split] = rows
        for w in ('all', 0, 66):
            m = metrics(rows if w == 'all' else [r for r in rows if r['wait_steps'] == w])
            variant['cohorts'][f'{split}_W{w}'] = m
            flat.append({'variant': name, 'split': split, 'W_steps': w,
                **{k: m[k] for k in ('examples', 'safe_candidates', 'safe_admitted', 'false_admissions',
                    'conditional_class_correct', 'no_safe_families', 'no_safe_families_rejected',
                    'recoverable_families', 'safe_covered_families', 'cheapest_admitted_safe_families')}})
    variant['training_start_max_change'] = max(abs(r['start_valid_score'] - starts[start_key(r)]) for r in scores[name]['train'])
    assert variant['training_start_max_change'] == 0
    report['variants'][name] = variant
time_scores = {}
for split in data:
    rows = read(new / ('train-after-support-time-predictions.jsonl' if split == 'train' else 'after-support-time-predictions.jsonl'))
    assert len(rows) == len(data[split])
    for a, b in zip(rows, data[split]):
        assert a['label'] == b['support_time']['label'] and a['target'] == b['support_time']
    time_scores[split] = rows
    for w in ('all', 0, 66):
        subset = rows if w == 'all' else [r for r in rows if r['post_prefix_wait_steps'] == w]
        report['variants']['support_time']['cohorts'][f'{split}_W{w}']['auxiliary_time'] = support_time_metrics(subset)
report['checks']['same_primary_sampling'] = (new / 'sampling-plan.jsonl').read_bytes() == (baseline / 'sampling-plan.jsonl').read_bytes()
base_training, new_training = read(baseline / 'training.jsonl'), read(new / 'training.jsonl')
report['checks']['primary_step1_CE_identical'] = base_training[0]['classification_loss'] == new_training[0]['classification_loss']
report['checks']['actual_primary_updates_identical'] = all(
    all(a[k] == b[k] for k in ('example_indices', 'labels', 'sample_kind', 'ranked_pair', 'task'))
    for a, b in zip(base_training, new_training))
report['checks']['start_probability_max_difference'] = max(abs(a['start_valid_score'] - b['start_valid_score'])
    for split in data for a, b in zip(scores['baseline'][split], scores['support_time'][split]))
frozen = json.loads((root / 'frozen-inputs-before.json').read_text())
report['checks']['old_start_threshold_files_unchanged'] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == d for p, d in frozen.items())
assert report['checks']['same_primary_sampling'] and report['checks']['actual_primary_updates_identical']
assert report['checks']['primary_step1_CE_identical'] and report['checks']['start_probability_max_difference'] == 0
assert report['checks']['old_start_threshold_files_unchanged']
old = report['variants']['baseline']['cohorts']['development_W66']
current = report['variants']['support_time']['cohorts']['development_W66']
report['gate'] = {'false_admissions_decrease': current['false_admissions'] < old['false_admissions'],
    'safe_admission_does_not_decrease': current['safe_admitted'] >= old['safe_admitted'],
    'zero_false_admissions': current['false_admissions'] == 0,
    'all_seven_safe_candidates_admitted': current['safe_admitted'] == 7,
    'passed': current['false_admissions'] == 0 and current['safe_admitted'] == 7,
    'no_safe_family_test_available': current['no_safe_families'] > 0}
report['matched_W_false_admissions'] = []
time_map = {(r['seed'], r['scenario'], r['post_prefix_wait_steps'], r['local_relevel_steps']): r for r in time_scores['development']}
for a, b in zip(scores['baseline']['development'], scores['support_time']['development']):
    if b['wait_steps'] == 66 and b['label'] != 'A' and b['class_probabilities'][0] >= tau:
        k = b['seed'], b['scenario'], 66, b['plan']['local_relevel_steps']
        t = time_map[k]
        report['matched_W_false_admissions'].append({'seed': b['seed'], 'scenario': b['scenario'],
            'relevel_steps': b['plan']['local_relevel_steps'], 'label': b['label'],
            'baseline_score': a['class_probabilities'][0], 'new_score': b['class_probabilities'][0],
            'native_prediction': b['conditional_prediction'], 'auxiliary_true_time': t['label'],
            'auxiliary_predicted_time': t['prediction']})
previous = json.loads((baseline.parent / 'assessment.json').read_text())
report['reused_physical_availability_and_costs'] = previous['physical_availability']
report['serving_check'] = json.loads((root / 'serving-check.json').read_text())
(root / 'assessment.json').write_text(json.dumps(report, indent=2) + '\n')
with (root / 'comparison.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
    writer.writeheader()
    writer.writerows(flat)
decision = {'status': 'FINAL512_SUPPORT_TIME_AUXILIARY_COMPLETED', 'gate': report['gate'],
    'checkpoint_promoted': False, 'independent_closed_loop': False, 'new_physical_episodes': 0,
    'old_route_start_threshold_retained': report['checks']['old_start_threshold_files_unchanged'],
    'assessment': str((root / 'assessment.json').resolve())}
(root / 'decision.json').write_text(json.dumps(decision, indent=2) + '\n')
print(json.dumps({'gate': report['gate'], 'comparison': flat, 'new_auxiliary': result.get('support_time')}, indent=2))
