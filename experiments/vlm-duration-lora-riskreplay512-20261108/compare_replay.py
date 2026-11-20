"""Compare risk replay with its initial checkpoint on the unchanged data."""
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parents[1]))
from auditembodied.vlm_duration_lora import prediction_metrics, timing_contrasts

load=lambda p:json.loads(p.read_text())
read=lambda p:[json.loads(s) for s in p.read_text().splitlines()]
baseline=root.parent/'vlm-duration-lora-newstates512-20261108'
config=load(root/'protocol.json')
result=load(root/'result.json')
assert result['completed_steps']==512 and result['data_unchanged'] and result['frozen_start_unchanged']
assert config['data_sha256']==load(baseline/'protocol.json')['data_sha256']
design=config['sampling_design']


def frontier(records):
    out=[]
    for budget in range(4):
        best={'false_safe_budget':budget,'true_safe_accepted':0,'false_safe_accepted':0,'threshold':None}
        for threshold in sorted({r['class_probabilities'][0] for r in records}):
            chosen=[r for r in records if r['class_probabilities'][0]>=threshold]
            fp=sum(r['label']!='A' for r in chosen);tp=sum(r['label']=='A' for r in chosen)
            if fp<=budget and tp>best['true_safe_accepted']:
                best.update(true_safe_accepted=tp,false_safe_accepted=fp,threshold=threshold)
        out.append(best)
    return out


def gate(records,threshold=None):
    rs=records if threshold is None else [{**r,'prediction':'A' if r['class_probabilities'][0]>=threshold else 'rejected'} for r in records]
    timing=timing_contrasts(rs)['cases']
    flips=[r for r in timing if r['changed_safety']]
    controls=[r for r in timing if not r['changed_safety']]
    flip=len(flips)==1 and flips[0]['both_safety_decisions_correct']
    aa=sum(r['both_safety_decisions_correct'] for r in controls)
    process=sum(r['label'] in 'BD' and r['prediction']=='A' for r in rs)
    tp=sum(r['label']==r['prediction']=='A' for r in rs)
    return {'passed':bool(flip and aa>=5 and process==0 and tp>=28),'threshold':threshold,
            'critical_flip_correct':bool(flip),'AA_controls_both_accepted':aa,'process_false_accepts':process,
            'safe_accepted':tp,'other_false_accepts':sum(r['label'] not in 'ABD' and r['prediction']=='A' for r in rs)}


summary,table,timing_rows,targets={},[],[],{}
for name,folder in [('before',baseline),('replay',root)]:
    train=read(folder/'train-after-predictions.jsonl');dev=read(folder/'after-predictions.jsonl')
    summary[name]={}
    for split,records in [('train',train),('old_train',[r for r in train if r['seed']<8]),
                          ('new_train',[r for r in train if r['seed']>=8]),('development',dev)]:
        m=prediction_metrics(records)
        local=[r for r in records if r['plan']['kind']=='local' and r['label']!='E']
        m['valid_local_accuracy']={'examples':len(local),'correct':sum(r['label']==r['prediction'] for r in local)}
        m['local_incomplete']={c:{'examples':sum(r['label']==c for r in local),'correct':sum(r['label']==c and r['prediction']==c for r in local),
            'false_safe':sum(r['label']==c and r['prediction']=='A' for r in local)} for c in 'CD'}
        controls=[r for r in m['timing_40_60']['cases'] if not r['changed_safety']]
        m['AA_controls']={'pairs':len(controls),'both_correct':sum(r['both_safety_decisions_correct'] for r in controls)}
        m['posthoc_frontier']=frontier(records)
        summary[name][split]=m
        table.append({'method':name,'split':split,'examples':len(records),'accuracy':m['accuracy'],
            'false_safe':m['false_safe_accepts'],'safe_recall':m['safe_recall'],'safe_AUROC':m['safe_probability_auroc'],
            'process_false_safe':m['process_violations']['accepted'],'local_pair_correct':m['within_state_local_ranking']['correct'],
            'local_pairs':m['within_state_local_ranking']['pairs'],'AA_both_correct':m['AA_controls']['both_correct'],'AA_pairs':len(controls)})
        timing_rows.extend({'method':name,'split':split,**r} for r in m['timing_40_60']['cases'])
    targets[name]={kind:[{'index':i,'seed':train[i]['seed'],'scenario':train[i]['scenario'],'label':train[i]['label'],
                         'prediction':train[i]['prediction'],'A_score':train[i]['class_probabilities'][0]} for i in design[key]]
                   for kind,key in [('B','hard_B_indices'),('C','hard_C_indices'),('A','hard_A_indices')]}
dev=read(root/'after-predictions.jsonl')
points=[gate(dev,t) for t in sorted({r['class_probabilities'][0] for r in dev})]
passing=[r for r in points if r['passed']]
gates={'default_argmax':gate(dev),'passing_posthoc_points':passing,
       'proposed_point':max(passing,key=lambda r:(r['safe_accepted'],-r['other_false_accepts'])) if passing else None,
       'deployed':False,'note':'Thresholds are post-hoc on development, not independent policy evidence.'}
(root/'development-gate.json').write_text(json.dumps(gates,indent=2)+'\n')
training=read(root/'training.jsonl')
assert [r['step'] for r in training]==list(range(1,513))
report={'completed_steps':512,'additional_training':True,'new_physical_episodes':0,'data_unchanged':True,
    'frozen_start_unchanged':True,'training_seconds':result['training_seconds'],
    'all_losses_finite':all(math.isfinite(r['loss']) for r in training),
    'actual_update_counts':dict(Counter(r['sample_kind'] for r in training)),
    'presentations':sum(len(r['labels']) for r in training),
    'class_presentations':dict(Counter(c for r in training for c in r['labels'])),
    'targets':targets,'comparison':summary,'development_gate':gates,
    'note':'This run continues the previous checkpoint for 512 extra steps and changes sampling; no equal-total-budget or component-wise causal claim.'}
(root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
for filename,rows in [('comparison.csv',table),('timing-40-60-comparison.csv',timing_rows)]:
    with (root/filename).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for r in table:print(json.dumps(r))
print('targets',json.dumps(targets));print('gate',json.dumps(gates))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
fig,axes=plt.subplots(1,3,figsize=(14,4.1),constrained_layout=True)
for name,offset,color in [('before',-.18,'#8c949e'),('replay',.18,'#009e73')]:
    cases=targets[name]['B']+targets[name]['C']
    axes[0].bar(np.arange(len(cases))+offset,[r['A_score'] for r in cases],.34,label=name,color=color)
    d=summary[name]['development']
    bars=axes[1].bar(np.arange(2)+offset,[d['safe_recall'],d['process_violations']['accepted']/3],.34,label=name,color=color)
    axes[1].bar_label(bars,labels=[f"{round(d['safe_recall']*40)}/40",f"{d['process_violations']['accepted']}/3"],padding=3,fontsize=8)
    p=next(r for r in d['timing_40_60']['cases'] if r['changed_safety'])
    axes[2].plot([40,60],[p['p_safe_40'],p['p_safe_60']],'o-',label=name,color=color)
    for x,y,label in [(40,p['p_safe_40'],p['prediction_40']),(60,p['p_safe_60'],p['prediction_60'])]:
        axes[2].annotate(label,(x,y),xytext=(3,5),textcoords='offset points')
axes[0].set(xticks=range(5),xticklabels=['B8L30','B10R30','B10R40','C2L','C7L'],ylim=(0,1.1),ylabel='Uncalibrated A score (lower is correct)',title='Targeted training failures')
axes[1].set(xticks=range(2),xticklabels=['True safe accepted','Process unsafe accepted'],ylim=(0,1.15),ylabel='Fraction within each group',title='Development coverage and process errors')
axes[2].set(xticks=[40,60],ylim=(-.05,1.1),xlabel='Relevel steps',ylabel='Uncalibrated joint A score',title='Development seed 23 left: true B/A')
for ax in axes:ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8)
fig.savefig(root/'risk-replay-comparison.png',dpi=200)
fig.savefig(root/'risk-replay-comparison.pdf')
