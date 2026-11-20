"""Fixed train-only gate, followed by optional final512 development comparison."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from auditembodied.vlm_duration_lora import joint_marginal_metrics, timing_contrasts


def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def metrics(rows, tau):
    accept = lambda r: r['class_probabilities'][0] >= tau
    groups = defaultdict(list)
    for r in rows:
        groups[r['seed'], r['scenario'], r['wait_steps']].append(r)
    cases = []
    for (seed, scenario, wait), group in groups.items():
        assert {r['plan']['local_relevel_steps'] for r in group} == {30, 40, 60}
        chosen = sorted([r for r in group if accept(r)], key=lambda r: r['plan']['local_relevel_steps'])
        cases.append({'seed': seed, 'scenario': scenario, 'W_steps': wait,
            'has_safe_candidate': any(r['label'] == 'A' for r in group),
            'safe_covered': any(r['label'] == 'A' for r in chosen),
            'all_rejected': not chosen, 'cheapest_admitted_safe': bool(chosen and chosen[0]['label'] == 'A'),
            'selected_steps': chosen[0]['plan']['local_relevel_steps'] if chosen else None,
            'selected_label': chosen[0]['label'] if chosen else None})
    no_safe = [r for r in cases if not r['has_safe_candidate']]
    m = {'examples': len(rows), 'class_counts': dict(Counter(r['label'] for r in rows)),
        'native_correct': sum(r['conditional_prediction'] == r['label'] for r in rows),
        'native_confusion': {c: dict(Counter(r['conditional_prediction'] for r in rows if r['label'] == c)) for c in 'ABCD'},
        'safe_candidates': sum(r['label'] == 'A' for r in rows),
        'safe_admitted': sum(r['label'] == 'A' and accept(r) for r in rows),
        'false_admitted': sum(r['label'] != 'A' and accept(r) for r in rows),
        'no_safe_families': len(no_safe), 'no_safe_families_rejected': sum(r['all_rejected'] for r in no_safe),
        'recoverable_families': sum(r['has_safe_candidate'] for r in cases),
        'safe_covered_families': sum(r['safe_covered'] for r in cases),
        'cheapest_admitted_safe_families': sum(r['cheapest_admitted_safe'] for r in cases),
        'family_cases': cases, 'marginals': joint_marginal_metrics(rows)}
    m['timing_40_60'] = timing_contrasts([{**r, 'prediction': r['conditional_prediction']} for r in rows])
    return m


def frozen_unchanged(root):
    digests = json.loads((root / 'frozen-inputs-before.json').read_text())
    return all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == d for p, d in digests.items())


def assess_diagnostic(root):
    root = Path(root)
    plan = json.loads((root / 'experiment-plan.json').read_text())
    directory = root / 'diagnostic128'
    result = json.loads((directory / 'result.json').read_text())
    protocol = json.loads((directory / 'protocol.json').read_text())
    updates = read(directory / 'training.jsonl')
    rows = read(directory / 'train-after-predictions.jsonl')
    expected = read(root / 'diagnostic-data/train.jsonl')
    assert result['completed_steps'] == len(updates) == 128 and len(rows) == 30
    assert result['frozen_start_unchanged'] and result['data_unchanged']
    assert protocol['train_only'] and protocol['development_examples'] == 0
    assert not (root / 'diagnostic-data/development.jsonl').exists()
    assert not (directory / 'after-predictions.jsonl').exists()
    assert protocol['init_adapter'] == plan['initial_outcome_adapter']
    assert protocol['process_loss_weight'] == .5 and protocol['fixed_admission_threshold'] == plan['threshold']
    assert sum(len(r['example_indices']) for r in updates) == 256
    assert Counter(r['sample_kind'] for r in updates) == Counter(duration_family=64, uniform_single=64)
    assert all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm_before_clip']) for r in updates)
    assert all('support_time_loss' not in r for r in updates)
    for a, b in zip(rows, expected):
        assert all(a[k] == b[k] for k in ('seed', 'scenario', 'image', 'robot', 'plan', 'label', 'wait_steps'))
    current = metrics(rows, plan['threshold'])
    gate_spec = plan['diagnostic']['fit_gate']
    actual = {**{k: current[k] for k in ('native_correct', 'safe_admitted', 'false_admitted', 'no_safe_families', 'no_safe_families_rejected')},
        'process_correct': current['marginals']['process_correct']}
    checks = {'training_steps_and_presentations_match': True, 'development_not_read': True,
        'frozen_start_unchanged': result['frozen_start_unchanged'], 'old_start_threshold_files_unchanged': frozen_unchanged(root)}
    assert checks['old_start_threshold_files_unchanged']
    baselines = {}
    for name, key in [('previous_aftergrasp', 'baseline_dir'), ('previous_time_auxiliary', 'previous_time_aux_dir')]:
        prior = read(Path(plan[key]) / 'train-after-predictions.jsonl')
        subset = [prior[i] for i in plan['diagnostic']['original_training_indices']]
        for a, b in zip(subset, expected):
            assert all(a[k] == b[k] for k in ('seed', 'scenario', 'plan', 'label', 'robot', 'image'))
        baselines[name] = metrics(subset, plan['threshold'])
    report = {'scope': 'Final128 TRAIN-only fit diagnosis. Prior512 results restricted to the same30 TRAIN rows, not same-budget retraining controls.',
        'steps': 128, 'presentations': 256, 'threshold': plan['threshold'], 'metrics': current,
        'gate_expected': gate_spec, 'gate_actual': actual, 'passed': all(actual[k] == v for k, v in gate_spec.items()),
        'checks': checks, 'comparisons_on_same_training_subset': baselines,
        'training_seconds': result['training_seconds'], 'peak_allocated_gpu_bytes': result['peak_allocated_gpu_bytes'],
        'errors': [{k: r[k] for k in ('seed', 'scenario', 'wait_steps', 'plan', 'label', 'conditional_prediction', 'class_probabilities')}
            for r in rows if r['conditional_prediction'] != r['label'] or (r['class_probabilities'][0] >= plan['threshold']) != (r['label'] == 'A')]}
    (root / 'diagnostic-gate.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    root = Path(__file__).resolve().parent
    plan = json.loads((root / 'experiment-plan.json').read_text())
    diagnostic = json.loads((root / 'diagnostic-gate.json').read_text())
    report = {'scope': plan['scope'], 'diagnostic': diagnostic,
        'new_physical_episodes': 0, 'threshold': plan['threshold'], 'W_steps': 66, 'W_seconds': 3.3,
        'old_start_threshold_files_unchanged': frozen_unchanged(root), 'full_fit_started': (root / 'full512/protocol.json').exists()}
    if diagnostic['passed']:
        result = json.loads((root / 'full512/result.json').read_text())
        updates = read(root / 'full512/training.jsonl')
        assert result['completed_steps'] == len(updates) == 512
        assert sum(len(r['example_indices']) for r in updates) == 1024
        assert result['frozen_start_unchanged'] and result['data_unchanged']
        report['full512'] = {}
        for split, filename in [('train', 'train-after-predictions.jsonl'), ('development', 'after-predictions.jsonl')]:
            report['full512'][split] = {}
            for name, directory in [('previous_aftergrasp', Path(plan['baseline_dir'])),
                                    ('previous_time_auxiliary', Path(plan['previous_time_aux_dir'])), ('joint_family', root / 'full512')]:
                rows = read(directory / filename)
                report['full512'][split][name] = {f'W{w}': metrics(rows if w == 'all' else [r for r in rows if r['wait_steps'] == w], plan['threshold'])
                    for w in ('all', 0, 66)}
    else:
        assert not report['full_fit_started'] and not (root / 'full-data').exists()
    if (root / 'serving-check.json').exists():
        report['serving_check'] = json.loads((root / 'serving-check.json').read_text())
    assert report['old_start_threshold_files_unchanged']
    (root / 'assessment.json').write_text(json.dumps(report, indent=2) + '\n')
    decision = {'status': 'DIAGNOSTIC_PASSED_FULL512_COMPLETED' if diagnostic['passed'] else 'DIAGNOSTIC128_FAILED_STOPPED_BEFORE_DEVELOPMENT',
        'diagnostic_passed': diagnostic['passed'], 'full_fit_started': report['full_fit_started'],
        'development_evaluated': diagnostic['passed'], 'checkpoint_promoted': False, 'new_physical_episodes': 0,
        'independent_closed_loop': False, 'old_route_start_threshold_retained': report['old_start_threshold_files_unchanged'],
        'assessment': str((root / 'assessment.json').resolve())}
    (root / 'decision.json').write_text(json.dumps(decision, indent=2) + '\n')
    print(json.dumps({'decision': decision, 'diagnostic_actual': diagnostic['gate_actual']}, indent=2))


if __name__ == '__main__':
    main()
