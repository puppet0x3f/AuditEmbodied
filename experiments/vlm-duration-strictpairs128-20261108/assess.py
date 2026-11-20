"""Compare the same 24 training rows and final 128-step checkpoints only."""
import csv
import hashlib
import json
from pathlib import Path

from auditembodied.vlm_duration_lora import fixed_admission_metrics, timing_contrasts

root=Path(__file__).resolve().parent
plan=json.loads((root/'experiment-plan.json').read_text())
read=lambda p:[json.loads(line) for line in p.read_text().splitlines()]
directories={'previous_random_pairs':Path(plan['baseline_run']),'strict128':root/'strict128','boundary128':root/'boundary128'}
reference=read(directories['previous_random_pairs']/'train-before-predictions.jsonl')
assessment={'scope':'Same24 training examples only; not a development or closed-loop test',
    'threshold':plan['threshold'],'development_read':False,'physical_rollouts':0,'runs':{}}
comparison=[]
for name,directory in directories.items():
    result=json.loads((directory/'result.json').read_text())
    protocol=json.loads((directory/'protocol.json').read_text())
    assert result['completed_steps']==128 and result['development'] is None
    assert result['data_unchanged'] and result['frozen_start_unchanged']
    assert protocol['evaluation_splits']==['train'] and protocol['development_examples']==0
    assert protocol['init_adapter']==plan['initial_outcome_adapter']
    assert protocol['data_sha256']['train.jsonl']==plan['data_sha256']
    before=read(directory/'train-before-predictions.jsonl')
    rows=read(directory/'train-after-predictions.jsonl')
    training=read(directory/'training.jsonl')
    native=[{**r,'prediction':r['conditional_prediction']} for r in rows]
    timing=timing_contrasts(native)
    m=fixed_admission_metrics(rows,plan['threshold'])
    for k in ('examples','positive_examples','negative_examples'):
        if k=='examples':m[k]=len(rows)
    m.update(training_seconds=result['training_seconds'],peak_allocated_gpu_bytes=result['peak_allocated_gpu_bytes'],
        optimizer_steps=len(training),presentations=sum(len(x['labels']) for x in training),
        loss_weight=protocol.get('admission_loss_weight',0.),
        strict_pair_order_correct=timing['changed_pair_order_correct'],
        strict_pair_classes_both_correct=timing['changed_pair_both_classes_correct'],
        strict_pair_count=timing['changed_safety_pairs'],
        safe_A_score_range=[min(r['class_probabilities'][0] for r in rows if r['label']=='A'),
            max(r['class_probabilities'][0] for r in rows if r['label']=='A')],
        maximum_non_A_score=max(r['class_probabilities'][0] for r in rows if r['label']!='A'),
        start_probability_max_change=max(abs(a['start_valid_score']-b['start_valid_score']) for a,b in zip(before,rows)),
        initial_probability_max_error=max(abs(x-y) for a,b in zip(before,reference)
            for x,y in zip(a['class_probabilities'],b['class_probabilities'])),
        finite_nonzero_gradient_steps=sum(0<r['gradient_norm_before_clip']<float('inf') for r in training),
        class_errors=[{'seed':r['seed'],'wait_steps':r['wait_steps'],'relevel_steps':r['plan']['local_relevel_steps'],
            'label':r['label'],'prediction':r['conditional_prediction'],'joint_A_score':r['class_probabilities'][0]}
            for r in rows if r['label']!=r['conditional_prediction']],
        candidate_scores=[{'seed':r['seed'],'scenario':r['scenario'],'wait_steps':r['wait_steps'],
            'relevel_steps':r['plan']['local_relevel_steps'],'label':r['label'],
            'prediction':r['conditional_prediction'],'joint_A_score':r['class_probabilities'][0],
            'accepted':r['class_probabilities'][0]>=plan['threshold']} for r in rows])
    assert m['optimizer_steps']==128 and m['presentations']==160
    assert m['initial_probability_max_error']==m['start_probability_max_change']==0
    assessment['runs'][name]=m
    comparison.append({'run':name,**{k:m[k] for k in ('native_correct','native_A_vs_non_A_correct',
        'safe_admitted','false_admitted','non_A_rejected','no_safe_families_rejected',
        'strict_pair_order_correct','strict_pair_classes_both_correct','passed')}})
strict=read(directories['strict128']/'sampling-plan.jsonl')
old=read(directories['previous_random_pairs']/'sampling-plan.jsonl')
assessment['only_11_action_pair_slots_changed']=sum(a!=b for a,b in zip(old,strict))==11 and all(
    a==b for a,b in zip(old,strict) if a['kind']!='action_contrast')
assessment['strict_pair_presentations']=json.loads((directories['strict128']/'protocol.json').read_text())['sampling_design']['strict_40_60_pair_presentations']
assessment['strict_and_boundary_sampling_identical']=(directories['strict128']/'sampling-plan.jsonl').read_bytes()==(directories['boundary128']/'sampling-plan.jsonl').read_bytes()
frozen=json.loads((root/'frozen-inputs-before.json').read_text())
assessment['frozen_old_start_and_threshold_files_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==d for p,d in frozen.items())
assert assessment['only_11_action_pair_slots_changed'] and assessment['strict_and_boundary_sampling_identical']
assert assessment['frozen_old_start_and_threshold_files_unchanged']
assessment['decision']='BOUNDARY_TINY_FIT_PASSED_NOT_GENERALIZATION' if assessment['runs']['boundary128']['passed'] else 'BOUNDARY_TINY_FIT_NOT_PASSED'
(root/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
with (root/'comparison.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(comparison[0]));writer.writeheader();writer.writerows(comparison)
print(json.dumps({'decision':assessment['decision'],'comparison':comparison},indent=2))
