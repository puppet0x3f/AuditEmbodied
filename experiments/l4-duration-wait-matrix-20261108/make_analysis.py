"""Paired physical hold effects and frozen-model duration ranking."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
import numpy as np

out=Path(__file__).resolve().parent
report=json.loads((out/'report.json').read_text())
predictions=[json.loads(s) for s in (out/'predictions.jsonl').read_text().splitlines()]
cases=sorted(report['cases'],key=lambda r:(r['split']!='train',r['seed'],r['scenario']))
groups={}
for r in predictions:groups.setdefault((r['seed'],r['scenario']),{})[r['plan']['local_relevel_steps']]=r
analysis={'valid':report['valid'],'physical_episodes':report['physical_episodes'],
    'by_split':report['by_split'],'ranking_by_split':{},'hold_label_transitions':{},'strict40_60':{}}
for split in ('train','development'):
    subset=[c for c in cases if c['split']==split]
    ranking=report['ranking_by_split'][split]
    analysis['ranking_by_split'][split]={k:v for k,v in ranking.items() if k!='paired_contrasts'}
    transitions={}
    for c in subset:
        for n in ('30','40','60'):
            key=c['zero_wait_labels_by_relevel'][n]+'->'+c['labels_by_relevel'][n]
            transitions[key]=transitions.get(key,0)+1
    analysis['hold_label_transitions'][split]=transitions
    strict=[]
    for c in subset:
        pair=groups[c['seed'],c['scenario']];a,b=pair[40],pair[60]
        if (a['label']=='A')==(b['label']=='A'):continue
        safe,unsafe=(a,b) if a['label']=='A' else (b,a)
        margin=safe['joint_A_score']-unsafe['joint_A_score']
        strict.append({'seed':c['seed'],'scenario':c['scenario'],'safe_steps':safe['plan']['local_relevel_steps'],
            'unsafe_steps':unsafe['plan']['local_relevel_steps'],'score_margin':margin,
            'correctly_ranked':margin>0,'unsafe_admitted':unsafe['accepted'],'safe_admitted':safe['accepted']})
    analysis['strict40_60'][split]={'pairs':len(strict),'correctly_ranked':sum(p['correctly_ranked'] for p in strict),
        'unsafe_candidates_admitted':sum(p['unsafe_admitted'] for p in strict),
        'safe_candidates_admitted':sum(p['safe_admitted'] for p in strict),'cases':strict}
analysis['no_safe_duration_states']=[{k:c[k] for k in ('seed','scenario','split','labels_by_relevel')} for c in cases if not c['safe_relevel_steps']]
analysis['short30_unsafe_longer_safe_cases']=[{k:c[k] for k in ('seed','scenario','split','labels_by_relevel')} for c in cases
    if c['labels_by_relevel']['30']!='A' and any(c['labels_by_relevel'][n]=='A' for n in ('40','60'))]
both_safe=[c for c in cases if c['safe_relevel_steps'] and c['labels_by_relevel']['60']=='A']
oracle_cost=float(np.mean([c['total_seconds_by_relevel'][str(c['minimal_safe_relevel'])] for c in both_safe]))
fixed_cost=float(np.mean([c['total_seconds_by_relevel']['60'] for c in both_safe]))
analysis['offline_safe_duration_cost_potential']={'paired_states':len(both_safe),'fixed60_mean_seconds':fixed_cost,
    'minimum_safe_mean_seconds':oracle_cost,'mean_saved_seconds':fixed_cost-oracle_cost,
    'scope':'Offline envelope of actually executed safe candidates on the same states; includes the 0.95s hold and final hold. Not learned-policy performance.'}
(out/'analysis-summary.json').write_text(json.dumps(analysis,indent=2)+'\n')

colors=['#2a8c85','#e6a23c','#9b9b9b','#b65555','#8860a9']
cmap=ListedColormap(colors);norm=BoundaryNorm(np.arange(6)-.5,5)
fig,axes=plt.subplots(1,3,figsize=(11.5,9),gridspec_kw={'width_ratios':[1,1,1.35]})
for ax,field,title in zip(axes[:2],('zero_wait_labels_by_relevel','labels_by_relevel'),('Existing no-wait executions','New 0.95s-hold executions')):
    values=np.array([['ABCDE'.index(c[field][n]) for n in ('30','40','60')] for c in cases])
    ax.imshow(values,cmap=cmap,norm=norm,aspect='auto')
    ax.set(xticks=[0,1,2],xticklabels=['30','40','60'],xlabel='Relevel steps',title=title,
        yticks=np.arange(len(cases)),yticklabels=[str(c['seed'])+('L' if c['scenario']=='left_loss' else 'R') for c in cases])
    ax.axhline(15.5,color='black',linewidth=1.5)
    for i,c in enumerate(cases):
        for j,n in enumerate(('30','40','60')):ax.text(j,i,c[field][n],ha='center',va='center',color='white',fontsize=8)
axes[0].set_ylabel('Training seeds 8-15 | Development seeds 20-23')
axes[1].set_ylabel('Whole-episode outcomes include every hold step')
axes[1].yaxis.set_label_position('right')

x=np.arange(5);width=.34
for offset,(split,color) in enumerate((('train','#4c78a8'),('development','#e6a23c'))):
    subset=[c for c in cases if c['split']==split]
    numbers=[sum(c['labels_by_relevel'][str(n)]=='A' for c in subset) for n in (30,40,60)]
    numbers.append(sum(bool(c['safe_relevel_steps']) for c in subset))
    numbers.append(report['ranking_by_split'][split]['offline_selected_safe_whole_states'])
    axes[2].bar(x+(offset-.5)*width,numbers,width,label=split.capitalize(),color=color)
    for j,v in enumerate(numbers):axes[2].text(j+(offset-.5)*width,v+.15,str(v),ha='center',fontsize=9)
axes[2].set(xticks=x,xticklabels=['30','40','60','Any safe\nplan','Frozen\nmodel\n(offline)'],ylim=(0,18),ylabel='Safely completed states',title='Physical envelope and frozen selection')
axes[2].tick_params(axis='x',labelsize=8)
axes[2].spines[['top','right']].set_visible(False)
axes[2].legend(frameon=False)
fig.suptitle('Matched duration interventions after a fixed inference-like hold\n72 new executions; frozen Qwen3-VL assessment; no training',fontsize=13)
legend=[Patch(facecolor=c,label=l) for c,l in zip(colors,('A: safe completion','B: completed, guard failed','C: incomplete, guards passed','D: incomplete, guard failed','E: invalid start'))]
fig.legend(handles=legend,loc='lower center',ncol=3,frameon=False,fontsize=9,bbox_to_anchor=(.5,.01))
fig.subplots_adjust(left=.08,right=.98,bottom=.15,top=.88,wspace=.5)
for extension in ('png','pdf'):fig.savefig(out/f'wait-duration-results.{extension}',dpi=180,bbox_inches='tight')
print(json.dumps({k:analysis[k] for k in ('valid','hold_label_transitions','strict40_60')},indent=2))
