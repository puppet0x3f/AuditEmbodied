"""Assess only the preselected 24 training rows, without loading development data."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from auditembodied.vlm_duration_lora import fixed_admission_metrics, timing_contrasts

root = Path(__file__).resolve().parent
plan = json.loads((root / 'experiment-plan.json').read_text())
threshold = plan['threshold']
read = lambda p: [json.loads(s) for s in p.read_text().splitlines()]
initial = load_file(str(Path(plan['initial_outcome_adapter']) / 'adapter_model.safetensors'))
frozen = json.loads((root / 'frozen-inputs-before.json').read_text())
unchanged = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in frozen.items())
assessment = {'scope': 'Training-fit diagnostic on the same 24 examples; no generalization claim',
    'threshold': threshold, 'development_read': False, 'new_physical_rollouts': 0,
    'frozen_old_start_and_threshold_files_unchanged': unchanged, 'runs': {}}
comparison = []
for name in ('current128', 'boundary128'):
    directory = root / name
    if not (directory / 'result.json').exists():
        continue
    protocol = json.loads((directory / 'protocol.json').read_text())
    result = json.loads((directory / 'result.json').read_text())
    assert protocol['train_only'] and protocol['development_examples'] == 0
    assert protocol['evaluation_splits'] == ['train'] and list(protocol['data_sha256']) == ['train.jsonl']
    assert protocol['steps'] == result['completed_steps'] == 128
    assert result['development'] is None and result['frozen_start_unchanged'] and result['data_unchanged']
    training = read(directory / 'training.jsonl')
    final = load_file(str(directory / 'adapter' / 'adapter_model.safetensors'))
    assert final.keys() == initial.keys()
    deltas = [(final[k].float() - initial[k].float()) for k in final]
    stats = {'parameters_with_nonzero_change': sum(bool(torch.count_nonzero(v)) for v in deltas),
        'parameter_tensors': len(deltas), 'maximum_absolute_change': max(float(v.abs().max()) for v in deltas),
        'l2_change': sum(float((v * v).sum()) for v in deltas) ** .5,
        'finite_nonzero_gradient_steps': sum(0 < r['gradient_norm_before_clip'] < float('inf') for r in training)}
    run = {'optimizer_steps': len(training), 'training_seconds': result['training_seconds'],
        'peak_allocated_gpu_bytes': result['peak_allocated_gpu_bytes'], 'weight_updates': stats,
        'frozen_start_unchanged_in_memory': result['frozen_start_unchanged'], 'splits': {}}
    for phase in ('before', 'after'):
        records = read(directory / f'train-{phase}-predictions.jsonl')
        metrics = fixed_admission_metrics(records, threshold)
        native = [{**r, 'prediction': r['conditional_prediction']} for r in records]
        metrics['strict_40_60'] = timing_contrasts(native)
        families = defaultdict(list)
        for row in records:
            families[(row['seed'], row['scenario'], row['wait_steps'])].append(row)
        metrics['families'] = [{'seed': k[0], 'scenario': k[1], 'wait_steps': k[2],
            'labels_30_40_60': [r['label'] for r in sorted(v, key=lambda r: r['plan']['local_relevel_steps'])],
            'native_30_40_60': [r['conditional_prediction'] for r in sorted(v, key=lambda r: r['plan']['local_relevel_steps'])],
            'admitted_steps': [r['plan']['local_relevel_steps'] for r in v if r['class_probabilities'][0] >= threshold]}
            for k, v in families.items()]
        metrics['minimum_positive_start_score'] = min(r['start_valid_score'] for r in records if r['label'] == 'A')
        metrics['positive_score_range'] = [min(r['class_probabilities'][0] for r in records if r['label'] == 'A'),
            max(r['class_probabilities'][0] for r in records if r['label'] == 'A')]
        metrics['maximum_non_A_score'] = max(r['class_probabilities'][0] for r in records if r['label'] != 'A')
        run['splits'][phase] = metrics
        comparison.append({'run': name, 'phase': phase,
            **{k: metrics[k] for k in ('native_correct', 'native_A_vs_non_A_correct', 'safe_admitted',
                'false_admitted', 'non_A_rejected', 'no_safe_families_rejected', 'passed')}})
    before = read(directory / 'train-before-predictions.jsonl')
    after = read(directory / 'train-after-predictions.jsonl')
    run['start_probability_max_change'] = max(abs(a['start_valid_score'] - b['start_valid_score']) for a, b in zip(before, after))
    assessment['runs'][name] = run
assert unchanged
if 'boundary128' in assessment['runs']:
    a, b = [root / name / 'sampling-plan.jsonl' for name in ('current128', 'boundary128')]
    assessment['matched_sampling_plan'] = a.read_bytes() == b.read_bytes()
    before_a, before_b = [read(root / name / 'train-before-predictions.jsonl') for name in ('current128', 'boundary128')]
    assessment['matched_initial_probability_max_error'] = max(abs(x - y) for a, b in zip(before_a, before_b)
        for x, y in zip(a['class_probabilities'], b['class_probabilities']))
current = assessment['runs']['current128']['splits']['after']
if current['passed']:
    assessment['decision'] = 'CURRENT_LOSS_TINY_TRAINING_GATE_PASSED_NO_BOUNDARY_COMPARISON_NEEDED'
elif current['native_A_vs_non_A_correct'] != 24:
    assessment['decision'] = 'NATIVE_A_NON_A_FIT_INCOMPLETE_DO_NOT_ATTRIBUTE_FAILURE_ONLY_TO_ADMISSION_BOUNDARY'
elif 'boundary128' not in assessment['runs']:
    assessment['decision'] = 'NATIVE_A_NON_A_FIT_PASSED_BOUNDARY_MISALIGNED_RUN_MATCHED_BOUNDARY_COMPARISON'
elif assessment['runs']['boundary128']['splits']['after']['passed']:
    assessment['decision'] = 'MATCHED_BOUNDARY_LOSS_PASSED_TINY_TRAINING_GATE_NOT_GENERALIZATION'
else:
    assessment['decision'] = 'BOUNDARY_COMPARISON_DID_NOT_PASS_TINY_TRAINING_GATE'
(root / 'assessment.json').write_text(json.dumps(assessment, indent=2) + '\n')
with (root / 'comparison.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(comparison[0])); writer.writeheader(); writer.writerows(comparison)
print(json.dumps({'decision': assessment['decision'], 'comparison': comparison}, indent=2))
