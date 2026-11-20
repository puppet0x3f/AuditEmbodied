"""Final matched-stage comparison; separate W feasibility from discrimination."""
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path

from auditembodied.vlm_duration_lora import start_key,timing_contrasts

root=Path(__file__).resolve().parent;plan=json.loads((root/'experiment-plan.json').read_text());tau=plan['threshold']
read=lambda p:[json.loads(s) for s in Path(p).read_text().splitlines()]
physical=json.loads((root/'collection/physical-report.json').read_text())
data={s:read(root/'collection/dataset'/(s+'.jsonl')) for s in ('train','development')}
raw=[]
for w in (0,plan['W_steps']):raw.extend(read(root/'collection'/f'trials-wait{w}.jsonl'))


def metrics(rows):
    accepted=lambda r:r['class_probabilities'][0]>=tau
    positives=[r for r in rows if r['label']=='A'];valid=[r for r in rows if r['label']!='E']
    groups=defaultdict(list)
    for r in rows:groups[r['seed'],r['scenario'],r['wait_steps']].append(r)
    families=[]
    for (s,q,w),group in groups.items():
        assert {r['plan']['local_relevel_steps'] for r in group}=={30,40,60}
        admitted=sorted([r for r in group if accepted(r)],key=lambda r:r['plan']['local_relevel_steps'])
        safe=any(r['label']=='A' for r in group)
        families.append({'seed':s,'scenario':q,'post_prefix_wait_steps':w,'has_safe_candidate':safe,
            'admitted_steps':[r['plan']['local_relevel_steps'] for r in admitted],
            'safe_covered':any(r['label']=='A' for r in admitted),
            'cheapest_admitted_safe':bool(admitted and admitted[0]['label']=='A'),
            'all_rejected':not admitted})
    no_safe=[f for f in families if not f['has_safe_candidate']]
    # W is part of the action, so 40/60 pairs must have the same W.
    contrasts=[]
    for w in sorted({r['wait_steps'] for r in valid}):
        contrasts.append(timing_contrasts([{**r,'prediction':r['conditional_prediction']} for r in valid if r['wait_steps']==w]))
    return {'examples':len(rows),'class_counts':dict(Counter(r['label'] for r in rows)),
        'safe_candidates':len(positives),'safe_admitted':sum(accepted(r) for r in positives),
        'false_admissions':sum(accepted(r) for r in rows if r['label']!='A'),
        'false_by_class':dict(Counter(r['label'] for r in rows if r['label']!='A' and accepted(r))),
        'conditional_class_correct':sum(r['conditional_prediction']==r['label'] for r in valid),
        'valid_outcomes':len(valid),'conditional_confusion':{c:dict(Counter(r['conditional_prediction'] for r in valid if r['label']==c)) for c in 'ABCD'},
        'no_safe_families':len(no_safe),'no_safe_families_rejected':sum(f['all_rejected'] for f in no_safe),
        'recoverable_families':sum(f['has_safe_candidate'] for f in families),
        'safe_covered_families':sum(f['safe_covered'] for f in families),
        'cheapest_admitted_safe_families':sum(f['cheapest_admitted_safe'] for f in families),
        'family_cases':families,'timing_by_W':contrasts}


report={'threshold':tau,'W_steps':plan['W_steps'],'W_seconds':plan['W_seconds'],
    'physical_episodes':96,'physical_pairing_valid':physical['valid'],'physical_availability':{},'variants':{},
    'scope':'Matched forced-candidate diagnostic with supplied repair roles and fixed initial19-step hold; heldout seeds46,47, not an independent closed-loop success test.'}
for split,seeds in (('train',range(40,46)),('development',(46,47))):
    for w in (0,plan['W_steps']):
        cases=[c for c in physical['cases'] if c['seed'] in seeds and c['post_prefix_wait_steps']==w]
        episodes=[r for r in raw if r['seed'] in seeds and r['post_prefix_wait_steps']==w]
        report['physical_availability'][f'{split}_W{w}']={'safe_candidates':sum(v=='A' for c in cases for v in c['labels'].values()),
            'recoverable_families':sum('A' in c['labels'].values() for c in cases),'families':len(cases),
            'prefix_guard_violating_families':sum(c['prefix_protection']['keeper_grasp_lost_steps']>0 or
                c['prefix_protection']['deep_interarm_contact_steps']>0 or c['prefix_protection']['max_unsupported_seconds']>.1 for c in cases),
            'mean_total_seconds':sum(r['total_post_fault_seconds'] for r in episodes)/len(episodes),
            'mean_inference_wait_seconds':sum(r['inference_wait_seconds'] for r in episodes)/len(episodes)}
flat=[];scores={}
for mode in ('stage_early','stage_late'):
    directory=root/(mode+'512');result=json.loads((directory/'result.json').read_text())
    protocol=json.loads((directory/'protocol.json').read_text());training=read(directory/'training.jsonl')
    assert result['completed_steps']==len(training)==512 and result['frozen_start_unchanged'] and result['data_unchanged']
    assert not list(directory.glob('*before*')) and protocol['final_checkpoint_evaluation_only']
    assert protocol['temporal_mode']==mode and protocol['init_adapter']==plan['initial_outcome_adapter']
    assert protocol['learning_rate']==plan['learning_rate'] and protocol['fixed_admission_threshold']==tau
    assert all(0<r['gradient_norm_before_clip']<float('inf') for r in training)
    assert protocol['train_seeds']==list(range(40,46)) and protocol['development_seeds']==[46,47]
    starts=json.loads((directory/'frozen-start-scores.json').read_text())
    variant={'training_seconds':result['training_seconds'],'peak_allocated_gpu_bytes':result['peak_allocated_gpu_bytes'],
        'presentations':sum(len(r['example_indices']) for r in training),'cohorts':{}}
    scores[mode]={}
    for split,examples in data.items():
        records=read(directory/('train-after-predictions.jsonl' if split=='train' else 'after-predictions.jsonl'))
        assert len(records)==len(examples)
        for a,b in zip(records,examples):assert all(a[k]==b[k] for k in ('seed','scenario','image','robot','plan','label','wait_steps'))
        scores[mode][split]=records
        for w in ('all',0,plan['W_steps']):
            subset=records if w=='all' else [r for r in records if r['wait_steps']==w]
            m=metrics(subset);variant['cohorts'][f'{split}_W{w}']=m
            flat.append({'mode':mode,'split':split,'W_steps':w,**{k:m[k] for k in ('examples','safe_candidates','safe_admitted','false_admissions',
                'conditional_class_correct','valid_outcomes','no_safe_families','no_safe_families_rejected',
                'recoverable_families','safe_covered_families','cheapest_admitted_safe_families')}})
    variant['training_start_probability_max_change']=max(abs(r['start_valid_score']-starts[start_key(r)]) for r in scores[mode]['train'])
    assert variant['training_start_probability_max_change']==0
    report['variants'][mode]=variant
report['same_sampling_plan']=(root/'stage_early512/sampling-plan.jsonl').read_bytes()==(root/'stage_late512/sampling-plan.jsonl').read_bytes()
report['start_probability_max_difference']=max(abs(a['start_valid_score']-b['start_valid_score']) for split in data
    for a,b in zip(scores['stage_early'][split],scores['stage_late'][split]))
frozen=json.loads((root/'frozen-inputs-before.json').read_text())
report['old_start_threshold_files_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==d for p,d in frozen.items())
assert report['same_sampling_plan'] and report['start_probability_max_difference']==0 and report['old_start_threshold_files_unchanged']
for name,digest in plan['data_sha256'].items():assert hashlib.sha256((root/'collection/dataset'/name).read_bytes()).hexdigest()==digest
old=report['variants']['stage_early']['cohorts'][f'development_W{plan["W_steps"]}']
new=report['variants']['stage_late']['cohorts'][f'development_W{plan["W_steps"]}']
report['matched_W_comparison']={'false_admissions_decrease':new['false_admissions']<old['false_admissions'],
    'safe_admission_does_not_decrease':new['safe_admitted']>=old['safe_admitted'],
    'no_safe_rejection_test_available':new['no_safe_families']>0,
    'no_safe_rejection_improves':new['no_safe_families_rejected']>old['no_safe_families_rejected'] if new['no_safe_families'] else None}
report['serving_check']=json.loads((root/'serving-check.json').read_text())
(root/'assessment.json').write_text(json.dumps(report,indent=2)+'\n')
with (root/'comparison.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
print(json.dumps({'comparison':report['matched_W_comparison'],'heldout':[r for r in flat if r['split']=='development']},indent=2))
