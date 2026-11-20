import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out=Path(__file__).resolve().parent
report=json.loads((out/'assessment.json').read_text())
physical=json.loads((out/'collection/report.json').read_text())
baseline=report['baseline'];variants=report['variants']
colors=['#aaaaaa','#4c78a8','#2a8c85'];labels=['Frozen old','Static 512','Temporal 512'];x=np.arange(3)
candidates=[baseline['wait_development']]+[variants[m]['cohorts']['development_existing_wait']['candidates'] for m in ('static','paired')]
families=[baseline['wait_families']]+[variants[m]['cohorts']['development_existing_wait']['families'] for m in ('static','paired')]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(11,7.6))
for ax,values,title,ylabel,top in [
    (axes[0,0],[m['false_admissions'] for m in candidates],'Post-wait dev: false admissions','Candidates (out of 24)',None),
    (axes[0,1],[m['safe_admitted'] for m in candidates],'Post-wait dev: safe coverage','Accepted safe candidates (out of 15)',17),
    (axes[1,0],[m['no_safe_states_fully_rejected'] for m in families],'No safe local duration: all rejected','States (out of 2)',2.6),
    (axes[1,1],[m['safe_choice_states'] for m in families],'Cheapest admitted local plan (offline)','Safe choices (out of 8 states)',9)]:
    ax.bar(x,values,color=colors)
    for i,v in enumerate(values):ax.text(i,v+.08,str(v),ha='center')
    ax.set(xticks=x,xticklabels=labels,title=title,ylabel=ylabel)
    ax.set_ylim(0,top if top else max(values)+2)
axes[0,1].axhline(13,color='#b65555',linestyle='--',linewidth=1)
axes[1,1].axhline(6,color='#b65555',linestyle='--',linewidth=1)
fig.suptitle('Same expanded training data and sample schedule | 512 steps each\nFrozen original route and start LoRA | unchanged admission threshold',fontsize=12)
fig.tight_layout(rect=(0,0,1,.92))
for ext in ('png','pdf'):fig.savefig(out/f'temporal-comparison.{ext}',dpi=180,bbox_inches='tight')

fig,axes=plt.subplots(1,2,figsize=(10,4.3))
for i,kind in enumerate(('continuation_labels','whole_episode_labels')):
    ax=axes[i]
    for n,color in zip(('30','40','60'),('#4c78a8','#e6a23c','#2a8c85')):
        values=[sum(c[kind].get(n)=='A' for c in physical['cases'] if c['wait_steps']==w) for w in (0,10,20)]
        ax.plot([0,.5,1.],values,'o-',label=n+' relevel steps',color=color)
    ax.set(xlabel='Executed hold (seconds)',ylabel='Safely completed train fault states',ylim=(0,17),
        title='Future from h1' if i==0 else 'Whole episode including hold')
    ax.legend(frameon=False,fontsize=8)
fig.suptitle('144 new physical interventions | 16 fault states | paired prefixes')
fig.tight_layout(rect=(0,0,1,.91))
for ext in ('png','pdf'):fig.savefig(out/f'physical-wait-interventions.{ext}',dpi=180,bbox_inches='tight')
