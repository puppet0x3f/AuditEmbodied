"""Compare one final-step outcome adapter at the unchanged working point."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out=Path(__file__).resolve().parent
read=lambda p:[json.loads(s) for s in p.read_text().splitlines()]
setup=json.loads((out/'dataset'/'data-report.json').read_text())
result=json.loads((out/'result.json').read_text())
threshold=setup['threshold']


def candidate_metrics(rows):
    positive=[r for r in rows if r['label']=='A']
    admitted=[r for r in rows if r['class_probabilities'][0]>=threshold]
    safe=sum(r['label']=='A' for r in admitted)
    return {'examples':len(rows),'true_safe_candidates':len(positive),'admitted_candidates':len(admitted),
        'safe_candidates_admitted':safe,'safe_coverage':safe/len(positive) if positive else None,
        'false_admissions':sum(r['label']!='A' for r in admitted),
        'false_admissions_by_class':dict(Counter(r['label'] for r in admitted if r['label']!='A')),
        'process_violating_candidates_admitted':sum(r['label'] in 'BD' for r in admitted),
        'incomplete_candidates_admitted':sum(r['label'] in 'CD' for r in admitted),
        'native_class_accuracy':sum(r['prediction']==r['label'] for r in rows)/len(rows)}


def local_family_metrics(rows):
    groups={}
    for r in rows:
        p=r['plan']
        if (p['kind']!='local' or p['local_approach_steps']!=16 or p['local_close_steps']!=22 or
            p['local_relevel_steps'] not in (30,40,60)):continue
        # Same seed can contain h0 and h1; image/proprioception distinguish them.
        key=(r['image'],json.dumps(r['robot'],sort_keys=True),p['repair_arm'],p['keeper_arm'])
        groups.setdefault(key,{})[p['local_relevel_steps']]=r
    cases=[]
    for key,group in groups.items():
        if set(group)!={30,40,60}:continue
        ref=group[30];safe=[n for n,r in group.items() if r['label']=='A']
        admitted=sorted(n for n,r in group.items() if r['class_probabilities'][0]>=threshold)
        selected=admitted[0] if admitted else None
        cases.append({'seed':ref['seed'],'scenario':ref['scenario'],'image':ref['image'],
            'safe_durations':safe,'admitted_durations':admitted,'selected_duration':selected,
            'selected_safe':selected is not None and selected in safe,
            'safe_duration_covered':any(n in safe for n in admitted),
            'no_safe_duration':not safe,'all_rejected':not admitted})
    no_safe=[c for c in cases if c['no_safe_duration']]
    return {'complete_local_families':len(cases),'physically_recoverable_families':sum(bool(c['safe_durations']) for c in cases),
        'recoverable_families_with_safe_admission':sum(c['safe_duration_covered'] for c in cases),
        'offline_cheapest_choice_safe_families':sum(c['selected_safe'] for c in cases),
        'abstained_families':sum(c['all_rejected'] for c in cases),
        'no_safe_local_families':len(no_safe),'no_safe_local_families_fully_rejected':sum(c['all_rejected'] for c in no_safe),
        'cases':cases,'scope':'Logged candidate families with supplied roles; offline admission and ranking, not a new policy rollout.'}


cohorts={};candidate_rows=[];rejection_rows=[];start_errors=[];reference_error=0.
for split,after_file in (('train','train-after-predictions.jsonl'),('development','after-predictions.jsonl')):
    examples=read(out/'dataset'/f'{split}.jsonl')
    before=read(out/f'{split}-reference-before.jsonl');after=read(out/after_file)
    assert len(examples)==len(before)==len(after)
    for example,a,b in zip(examples,before,after):
        assert all(example[k]==a[k]==b[k] for k in ('seed','scenario','plan','label','image','robot'))
        start_errors.append(abs(a['start_valid_score']-b['start_valid_score']))
        candidate_rows.append({'split':split,'period':example['observation_period'],'seed':example['seed'],
            'scenario':example['scenario'],'image':example['image'],'plan_kind':example['plan']['kind'],
            'repair_arm':example['plan'].get('repair_arm'),'relevel_steps':example['plan'].get('local_relevel_steps'),
            'label':example['label'],'before_prediction':a['prediction'],'after_prediction':b['prediction'],
            'before_A_score':a['class_probabilities'][0],'after_A_score':b['class_probabilities'][0],
            'before_admitted':a['class_probabilities'][0]>=threshold,'after_admitted':b['class_probabilities'][0]>=threshold})
    for period in ('original','post_wait'):
        indices=[i for i,r in enumerate(examples) if r['observation_period']==period]
        a=[before[i] for i in indices];b=[after[i] for i in indices]
        fa,fb=local_family_metrics(a),local_family_metrics(b)
        cohorts[f'{period}_{split}']={'before':candidate_metrics(a),'after':candidate_metrics(b),
            'local_families_before':fa,'local_families_after':fb}
        for ca,cb in zip(fa['cases'],fb['cases']):
            assert ca['image']==cb['image'] and ca['seed']==cb['seed'] and ca['scenario']==cb['scenario']
            if not ca['no_safe_duration']:continue
            rejection_rows.append({'split':split,'period':period,'seed':ca['seed'],'scenario':ca['scenario'],
                'before_admitted_durations':ca['admitted_durations'],'after_admitted_durations':cb['admitted_durations'],
                'before_all_rejected':ca['all_rejected'],'after_all_rejected':cb['all_rejected']})
    if split=='train':
        actual=read(out/'train-before-predictions.jsonl')
        assert len(actual)==len(before)
        reference_error=max(abs(x-y) for a,b in zip(before,actual) for x,y in zip(a['class_probabilities'],b['class_probabilities']))

training=read(out/'training.jsonl')
assert len(training)==512 and [r['step'] for r in training]==list(range(1,513))
assert result['completed_steps']==512 and result['frozen_start_unchanged'] and result['data_unchanged']
frozen=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in setup['frozen_files_sha256'].items())
comparison={'threshold':threshold,'completed_optimizer_steps':512,'threshold_changed':False,
    'frozen_files_unchanged':frozen,'frozen_start_unchanged':result['frozen_start_unchanged'],
    'max_start_score_change':max(start_errors),'training_before_reference_max_error':reference_error,
    'data_unchanged':result['data_unchanged'],'training_seconds':result['training_seconds'],
    'peak_allocated_gpu_bytes':result['peak_allocated_gpu_bytes'],
    'candidate_presentations':sum(len(r['example_indices']) for r in training),
    'training_class_presentations':dict(Counter(c for r in training for c in r['labels'])),
    'unpaired_risk_updates':sum(r['sample_kind'] in ('new_B_pair','C_pair','process_pair') and not r['paired'] for r in training),
    'cohorts':cohorts,'no_safe_local_comparisons':rejection_rows,
    'scope':'Fixed-step final model, unchanged threshold and start adapter. Same logged observations before/after. No physical collection, threshold search, checkpoint selection or independent policy evaluation.'}
assert frozen and max(start_errors)==0. and reference_error<1e-6
(out/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
for name,rows in (('candidate-comparison.csv',candidate_rows),('no-safe-local-comparison.csv',rejection_rows)):
    with (out/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
flat=[]
for cohort,values in cohorts.items():
    for when in ('before','after'):flat.append({'cohort':cohort,'checkpoint':when,**values[when]})
with (out/'comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(11.5,7.5))
order=['original_train','post_wait_train','original_development','post_wait_development']
labels=['Original\ntrain','Post-wait\ntrain','Original\ndev','Post-wait\ndev'];x=np.arange(4);w=.34
colors=['#a5a5a5','#2a8c85']
for j,when in enumerate(('before','after')):
    position=x+(j-.5)*w
    false=[cohorts[k][when]['false_admissions'] for k in order]
    cover=[100*cohorts[k][when]['safe_coverage'] for k in order]
    axes[0,0].bar(position,false,w,color=colors[j],label=when.capitalize())
    axes[0,1].bar(position,cover,w,color=colors[j],label=when.capitalize())
    for i,v in enumerate(false):axes[0,0].text(position[i],v+.12,str(v),ha='center',fontsize=9)
    for i,v in enumerate(cover):
        m=cohorts[order[i]][when]
        axes[0,1].text(position[i],v+1,f"{m['safe_candidates_admitted']}/{m['true_safe_candidates']}",ha='center',fontsize=8)
axes[0,0].set(xticks=x,xticklabels=labels,ylabel='Falsely admitted candidates',title='Fixed-threshold false admissions')
axes[0,0].set_ylim(0,max(cohorts[k][v]['false_admissions'] for k in order for v in ('before','after'))+2)
axes[0,0].legend(frameon=False)
axes[0,1].set(xticks=x,xticklabels=labels,ylim=(0,114),ylabel='Safe candidate coverage (%)',title='Coverage at the same threshold')

for j,when in enumerate(('before','after')):
    for i,split in enumerate(('train','development')):
        f=cohorts[f'post_wait_{split}'][f'local_families_{when}'];position=i+(j-.5)*w
        reject=f['no_safe_local_families_fully_rejected'];chosen=f['offline_cheapest_choice_safe_families']
        axes[1,0].bar(position,reject,w,color=colors[j]);axes[1,0].text(position,reject+.05,f"{reject}/{f['no_safe_local_families']}",ha='center')
        axes[1,1].bar(position,chosen,w,color=colors[j]);axes[1,1].text(position,chosen+.15,f"{chosen}/{f['complete_local_families']}",ha='center')
        if j==0:axes[1,1].plot([i-.4,i+.4],[f['physically_recoverable_families']]*2,color='#b65555',linestyle='--',linewidth=1.2)
axes[1,0].set(xticks=[0,1],xticklabels=['Post-wait train','Post-wait dev'],ylim=(0,2.5),ylabel='Families with all durations rejected',title='No safe local duration: rejection')
axes[1,1].set(xticks=[0,1],xticklabels=['Post-wait train','Post-wait dev'],ylim=(0,17),ylabel='Safe offline choices',title='Cheapest admitted local plan (offline)')
fig.suptitle('512-step outcome LoRA | frozen start LoRA and threshold\nSame 196 train / 80 development examples; no policy rollout',fontsize=12)
fig.tight_layout(rect=(0,0,1,.92))
for suffix in ('png','pdf'):fig.savefig(out/f'postwait-training-comparison.{suffix}',dpi=180,bbox_inches='tight')
print(json.dumps({'threshold':threshold,'frozen_unchanged':frozen,'cohorts':{k:{w:v[w] for w in ('before','after')} for k,v in cohorts.items()},'no_safe':rejection_rows},indent=2))
