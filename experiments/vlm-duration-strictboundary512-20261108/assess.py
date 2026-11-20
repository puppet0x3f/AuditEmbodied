"""Compare final outcomes at the frozen threshold using recorded candidates."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from auditembodied.vlm_duration_lora import fixed_admission_metrics, start_key, timing_contrasts

root=Path(__file__).resolve().parent
plan=json.loads((root/'experiment-plan.json').read_text());threshold=plan['threshold']
read=lambda p:[json.loads(s) for s in Path(p).read_text().splitlines()]
data={split:read(Path(plan['data_dir'])/(split+'.jsonl')) for split in ('train','development')}
key=lambda r:json.dumps([r['seed'],r['scenario'],r['image'],r['robot'],r['plan']],sort_keys=True)


def metrics(rows):
    valid=[r for r in rows if r['label']!='E']
    m=fixed_admission_metrics(rows,threshold)
    m['examples']=len(rows);m['class_counts']=dict(Counter(r['label'] for r in rows))
    m['joint_class_correct']=sum(r['prediction']==r['label'] for r in rows)
    m['valid_conditional_class_correct']=sum(r.get('conditional_prediction',r['prediction'])==r['label'] for r in valid)
    m['valid_outcome_examples']=len(valid)
    m['native_confusion']={c:dict(Counter(r.get('conditional_prediction',r['prediction']) for r in valid if r['label']==c)) for c in 'ABCD'}
    m['native_A_vs_non_A_correct']=sum((r.get('conditional_prediction',r['prediction'])=='A')==(r['label']=='A') for r in valid)
    m['false_by_class']=dict(Counter(r['label'] for r in rows if r['label']!='A' and r['class_probabilities'][0]>=threshold))
    m['strict_40_60']=timing_contrasts([{**r,'prediction':r.get('conditional_prediction',r['prediction'])} for r in valid])
    groups=defaultdict(list)
    for r in valid:
        groups[(r['image'],json.dumps(r['robot'],sort_keys=True),r['plan']['repair_arm'],r['plan']['keeper_arm'])].append(r)
    families=[]
    for group in groups.values():
        accepted=[r for r in group if r['class_probabilities'][0]>=threshold]
        safe=[r for r in group if r['label']=='A']
        cost=lambda r:sum(r['plan'][k] for k in ('local_approach_steps','local_close_steps','local_relevel_steps'))
        chosen=min(accepted,key=cost) if accepted else None
        families.append({'seed':group[0]['seed'],'scenario':group[0]['scenario'],
            'wait_steps':group[0].get('wait_steps',0),'has_safe_candidate':bool(safe),'all_rejected':not accepted,
            'safe_candidate_covered':any(r['label']=='A' for r in accepted),
            'chosen_safe':bool(chosen and chosen['label']=='A'),
            'admitted_relevel_steps':[r['plan']['local_relevel_steps'] for r in accepted]})
    no_safe=[f for f in families if not f['has_safe_candidate']]
    m.update(no_safe_families=len(no_safe),no_safe_families_rejected=sum(f['all_rejected'] for f in no_safe),
        families=len(families),recoverable_families=sum(f['has_safe_candidate'] for f in families),
        safe_covered_families=sum(f['safe_candidate_covered'] for f in families),
        cheapest_admitted_safe_families=sum(f['chosen_safe'] for f in families),family_cases=families)
    # Prerequisite-invalid E rows are reported separately from conditional A-D fit.
    m['native_correct']=m['valid_conditional_class_correct']
    return m


def align(scores,examples,require_all=True):
    lookup={key(r):r for r in scores};aligned=[]
    for example in examples:
        r=lookup.get(key(example))
        if r is None:
            if require_all:raise ValueError('historical candidate mismatch')
            continue
        assert r['label']==example['label']
        aligned.append({**example,**r})
    return aligned


model=root/'model512';result=json.loads((model/'result.json').read_text())
protocol=json.loads((model/'protocol.json').read_text());updates=read(model/'training.jsonl')
assert result['completed_steps']==len(updates)==512
assert [r['step'] for r in updates]==list(range(1,513))
assert result['frozen_start_unchanged'] and result['data_unchanged']
assert protocol['final_checkpoint_evaluation_only'] and not list(model.glob('*before*'))
assert protocol['admission_loss_weight']==.1 and protocol['admission_logit_margin']==1.
assert protocol['strict_timing_coverage'] and protocol['fixed_admission_threshold']==threshold
assert protocol['init_adapter']==plan['initial_outcome_adapter']
assert sum(len(r['labels']) for r in updates)==640
for name,digest in plan['data_sha256'].items():assert hashlib.sha256((Path(plan['data_dir'])/name).read_bytes()).hexdigest()==digest

variants={};current={split:read(model/('train-after-predictions.jsonl' if split=='train' else 'after-predictions.jsonl')) for split in data}
variants['strict_boundary512']=current
for name,folder in (('previous_static512','temporal_static512'),('previous_paired512','temporal_paired512'),('previous_mixed512','previous_mixed512')):
    directory=Path(plan['comparison_runs'][folder])
    variants[name]={split:read(directory/('train-after-predictions.jsonl' if split=='train' else 'after-predictions.jsonl')) for split in data}

old=Path('outputs/vlm-duration-lora-riskreplay512-20261108');old_data=Path('outputs/l4-duration-newstates-20261108/dataset')
wait=read('outputs/l4-duration-wait-matrix-20261108/predictions.jsonl')
variants['frozen_old_route']={}
for split in data:
    ex=read(old_data/(split+'.jsonl'));pred=read(old/('train-after-predictions.jsonl' if split=='train' else 'after-predictions.jsonl'))
    assert len(ex)==len(pred)
    for a,b in zip(ex,pred):assert all(a[k]==b[k] for k in ('seed','scenario','plan','label'))
    variants['frozen_old_route'][split]=[{**a,**b} for a,b in zip(ex,pred)]+[r for r in wait if r['split']==split]

report={'threshold':threshold,'scope':'Final checkpoint only; recorded-candidate train/development diagnostic, not a physical policy test',
    'training_seconds':result['training_seconds'],'peak_allocated_gpu_bytes':result['peak_allocated_gpu_bytes'],
    'optimizer_steps':len(updates),'presentations':640,'frozen_start_unchanged_in_memory':result['frozen_start_unchanged'],
    'data_unchanged':result['data_unchanged'],'only_final_checkpoint_evaluated':True,'cohorts':{}}
flat=[]
for name,splits in variants.items():
    report['cohorts'][name]={}
    for split,scores in splits.items():
        matched=align(scores,data[split],require_all=name not in ('previous_mixed512','frozen_old_route'))
        for source in ('all',*sorted({r['source'] for r in matched})):
            subset=matched if source=='all' else [r for r in matched if r['source']==source]
            m=metrics(subset);report['cohorts'][name][split+'_'+source]=m
            flat.append({'model':name,'split':split,'source':source,
                **{k:m[k] for k in ('examples','positive_examples','safe_admitted','false_admitted',
                    'no_safe_families','no_safe_families_rejected','valid_outcome_examples','valid_conditional_class_correct',
                    'recoverable_families','safe_covered_families')}})
qcache=json.loads((model/'frozen-start-scores.json').read_text())
report['train_start_cache_max_change']=max(abs(r['start_valid_score']-qcache[start_key(r)]) for r in current['train'])
previous=variants['previous_paired512']
report['frozen_start_vs_previous_max_change']=max(abs(a['start_valid_score']-b['start_valid_score'])
    for split in data for a,b in zip(align(current[split],data[split]),align(previous[split],data[split])))
frozen=json.loads((root/'frozen-inputs-before.json').read_text())
report['frozen_old_start_and_threshold_files_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in frozen.items())
assert report['train_start_cache_max_change']==report['frozen_start_vs_previous_max_change']==0
assert report['frozen_old_start_and_threshold_files_unchanged']
target=report['cohorts']['strict_boundary512']['development_existing_wait']
report['development_reference_gate']={'false_admissions_at_most_2':target['false_admitted']<=2,
    'safe_admitted_at_least_13':target['safe_admitted']>=13,
    'both_no_safe_families_rejected':target['no_safe_families']==target['no_safe_families_rejected']==2}
report['development_reference_gate_passed']=all(report['development_reference_gate'].values())
(root/'assessment.json').write_text(json.dumps(report,indent=2)+'\n')
with (root/'comparison.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
print(json.dumps({'gate':report['development_reference_gate'],'post_wait_development':[
    r for r in flat if r['split']=='development' and r['source']=='existing_wait'],
    'current_all':[r for r in flat if r['model']=='strict_boundary512' and r['source']=='all']},indent=2))
